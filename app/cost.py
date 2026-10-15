"""Per-query cost metering for Prism (Gemini models, search grounding, SerpApi)."""

import collections
import contextvars
import inspect
import json
import os
from typing import Any, AsyncIterator, Optional

DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "image_output": 0.0},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00, "image_output": 0.0},
    "gemini-3.1-flash-image": {"input": 0.50, "output": 3.00, "image_output": 60.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "image_output": 0.0},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "image_output": 0.0},
}

PRICES: dict[str, dict[str, float]] = {k: dict(v) for k, v in DEFAULT_PRICES.items()}

GROUNDING_PER_QUERY = 0.014
GROUNDING_PER_PROMPT_GEMINI2 = 0.035


def serpapi_per_search() -> float:
  val = os.environ.get("SERPAPI_COST_PER_SEARCH")
  if val is not None:
    try:
      return float(val)
    except ValueError:
      pass
  return 0.015


def vertex_search_per_query() -> float:
  val = os.environ.get("VERTEX_SEARCH_COST_PER_QUERY")
  if val is not None:
    try:
      return float(val)
    except ValueError:
      pass
  return 0.004


def get_prices() -> dict[str, dict[str, float]]:
  prices = {k: dict(v) for k, v in PRICES.items()}
  raw = os.environ.get("PRICING_JSON")
  if raw:
    try:
      override = json.loads(raw)
      if isinstance(override, dict):
        for m, p in override.items():
          if isinstance(p, dict):
            prices[m] = {
                "input": float(p.get("input", 0.0)),
                "output": float(p.get("output", 0.0)),
                "image_output": float(p.get("image_output", 0.0)),
            }
    except Exception:
      pass
  return prices


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
  if obj is None:
    return default
  if isinstance(obj, dict):
    return obj.get(key, default)
  try:
    return object.__getattribute__(obj, key)
  except (AttributeError, TypeError):
    return default


def _extract_image_tokens(usage: Any) -> int:
  details = _safe_get(usage, "candidates_tokens_details") or []
  total = 0
  for item in details:
    mod = _safe_get(item, "modality")
    if mod is not None and "IMAGE" in str(mod).upper():
      cnt = _safe_get(item, "token_count") or 0
      total += int(cnt)
  return total


def _extract_tokens(usage: Any) -> tuple[int, int, int]:
  """Returns (input_tokens, text_output_tokens, image_output_tokens)."""
  if usage is None:
    return 0, 0, 0
  prompt = _safe_get(usage, "prompt_token_count") or 0
  tool_use = _safe_get(usage, "tool_use_prompt_token_count") or 0
  input_tokens = int(prompt) + int(tool_use)

  image_tokens = _extract_image_tokens(usage)
  candidates = _safe_get(usage, "candidates_token_count") or 0
  thoughts = _safe_get(usage, "thoughts_token_count") or 0
  text_output_tokens = max(0, int(candidates) - image_tokens) + int(thoughts)

  return input_tokens, text_output_tokens, image_tokens


def _extract_grounding_queries(candidates: Any) -> tuple[int, bool]:
  cands = candidates or []
  g_queries = 0
  for cand in cands:
    gm = _safe_get(cand, "grounding_metadata")
    if gm:
      w = _safe_get(gm, "web_search_queries") or []
      img = _safe_get(gm, "image_search_queries") or []
      g_queries += len(w) + len(img)
  return g_queries, (g_queries > 0)


class Meter:
  """Collects token and search usage for a single request."""

  def __init__(self):
    self.genai_calls: list[dict] = []
    self.gemini3_grounding_queries: int = 0
    self.gemini2_grounded_prompts: int = 0
    self.serpapi_calls: list[str] = []
    self.vertex_search_queries: int = 0

  def add_genai(
      self,
      model: str,
      usage: Any,
      grounding_queries: int = 0,
      grounded: bool = False,
  ) -> None:
    in_tok, text_out_tok, img_tok = _extract_tokens(usage)
    clean_model = model.removeprefix("models/").split("/")[-1] if model else ""
    prices = get_prices()

    norm_model = model
    if norm_model in prices:
      rate = prices[norm_model]
      is_estimated = False
    elif clean_model in prices:
      rate = prices[clean_model]
      is_estimated = False
    else:
      rate = prices.get("gemini-3.8-flash", DEFAULT_PRICES["gemini-3.8-flash"])
      is_estimated = True

    usd = (
        (in_tok / 1_000_000.0) * rate["input"]
        + (text_out_tok / 1_000_000.0) * rate["output"]
        + (img_tok / 1_000_000.0) * rate.get("image_output", 0.0)
    )

    self.genai_calls.append({
        "model": model or "unknown",
        "in_tokens": in_tok,
        "text_out_tokens": text_out_tok,
        "image_tokens": img_tok,
        "usd": usd,
        "is_estimated": is_estimated,
    })

    if clean_model.startswith("gemini-2"):
      if grounded:
        self.gemini2_grounded_prompts += 1
    else:
      self.gemini3_grounding_queries += int(grounding_queries or 0)

  def add_serpapi(self, engine: str) -> None:
    self.serpapi_calls.append(str(engine or "google"))

  def add_vertex_search(self, count: int = 1) -> None:
    self.vertex_search_queries += int(count)

  def summary(self) -> dict:
    items = []
    grouped_genai = collections.defaultdict(list)
    for call in self.genai_calls:
      grouped_genai[call["model"]].append(call)

    for model, calls in grouped_genai.items():
      in_total = sum(c["in_tokens"] for c in calls)
      out_total = sum(c["text_out_tokens"] for c in calls)
      img_total = sum(c["image_tokens"] for c in calls)
      usd_total = sum(c["usd"] for c in calls)
      is_estimated = any(c["is_estimated"] for c in calls)

      cnt = len(calls)
      detail = (
          f"{cnt} call{'s' if cnt != 1 else ''} · "
          f"{in_total:,} in / {out_total:,} out tokens"
      )
      if img_total > 0:
        detail += f" · {img_total:,} image tokens"

      item = {
          "label": model,
          "detail": detail,
          "usd": round(usd_total, 6),
      }
      if is_estimated:
        item["estimated"] = True
      items.append(item)

    if self.gemini3_grounding_queries > 0 or self.gemini2_grounded_prompts > 0:
      grounding_usd = (
          self.gemini3_grounding_queries * GROUNDING_PER_QUERY
          + self.gemini2_grounded_prompts * GROUNDING_PER_PROMPT_GEMINI2
      )
      if self.gemini3_grounding_queries and self.gemini2_grounded_prompts:
        detail = (
            f"{self.gemini3_grounding_queries} search queries · "
            f"{self.gemini2_grounded_prompts} grounded prompt"
            f"{'s' if self.gemini2_grounded_prompts != 1 else ''}"
        )
      elif self.gemini3_grounding_queries:
        detail = (
            f"{self.gemini3_grounding_queries} search "
            f"{'query' if self.gemini3_grounding_queries == 1 else 'queries'}"
        )
      else:
        detail = (
            f"{self.gemini2_grounded_prompts} grounded prompt"
            f"{'s' if self.gemini2_grounded_prompts != 1 else ''}"
        )
      items.append({
          "label": "Google Search grounding",
          "detail": detail,
          "usd": round(grounding_usd, 6),
      })

    if self.serpapi_calls:
      tot = len(self.serpapi_calls)
      counts = collections.Counter(self.serpapi_calls)
      parts = [f"{eng} ×{cnt}" for eng, cnt in sorted(counts.items())]
      detail = f"{tot} search{'es' if tot != 1 else ''} ({', '.join(parts)})"
      usd = tot * serpapi_per_search()
      items.append({
          "label": "SerpApi",
          "detail": detail,
          "usd": round(usd, 6),
      })

    if self.vertex_search_queries > 0:
      cnt = self.vertex_search_queries
      price = vertex_search_per_query()
      items.append({
          "label": "Vertex AI Search",
          "detail": f"{cnt} query" if cnt == 1 else f"{cnt} queries",
          "usd": round(cnt * price, 6),
      })

    total_usd = round(sum(it["usd"] for it in items), 6)
    rate = serpapi_per_search()
    note = (
        "List prices (USD), free tiers ignored; "
        f"SerpApi at ${rate:g}/search."
    )
    if self.vertex_search_queries > 0:
      vs_rate = vertex_search_per_query()
      note = (
          "List prices (USD), free tiers ignored; "
          f"SerpApi at ${rate:g}/search; "
          f"Vertex AI Search ${vs_rate:g}/query (Enterprise)."
      )
    return {
        "total_usd": total_usd,
        "items": items,
        "note": note,
    }


_current: contextvars.ContextVar[Optional[Meter]] = contextvars.ContextVar(
    "_current", default=None
)


def start() -> Meter:
  m = Meter()
  _current.set(m)
  return m


def current() -> Optional[Meter]:
  return _current.get()


def record_serpapi(engine: str) -> None:
  m = _current.get()
  if m is not None:
    m.add_serpapi(engine)


def record_vertex_search(count: int = 1) -> None:
  m = _current.get()
  if m is not None:
    m.add_vertex_search(count)


async def _stream_with_meter(
    stream: Any, model: str, meter: Meter
) -> AsyncIterator[Any]:
  last_usage = None
  max_g_queries = 0
  any_grounded = False
  recorded = False
  try:
    async for chunk in stream:
      u = _safe_get(chunk, "usage_metadata")
      if u is not None:
        last_usage = u
      c_queries, grounded = _extract_grounding_queries(
          _safe_get(chunk, "candidates")
      )
      if c_queries > max_g_queries:
        max_g_queries = c_queries
      if grounded:
        any_grounded = True
      yield chunk
  finally:
    if not recorded:
      recorded = True
      meter.add_genai(model, last_usage, max_g_queries, any_grounded)


class _InstrumentedModelHandler:
  """Intercepts generate_content calls on Client.aio.models."""

  def __init__(self, orig_gc: Any, orig_gcs: Any):
    self._orig_gc = orig_gc
    self._orig_gcs = orig_gcs

  async def generate_content(self, *args, **kwargs) -> Any:
    res = self._orig_gc(*args, **kwargs)
    if inspect.iscoroutine(res):
      res = await res
    meter = current()
    if meter is not None:
      model = kwargs.get("model") or (args[0] if args else "")
      usage = _safe_get(res, "usage_metadata")
      cands = _safe_get(res, "candidates")
      g_queries, grounded = _extract_grounding_queries(cands)
      meter.add_genai(model, usage, g_queries, grounded)
    return res

  async def generate_content_stream(self, *args, **kwargs) -> Any:
    active_meter = current()
    res = self._orig_gcs(*args, **kwargs)
    if inspect.iscoroutine(res):
      res = await res
    if active_meter is None:
      return res

    model = kwargs.get("model") or (args[0] if args else "")
    return _stream_with_meter(res, model, active_meter)


def instrument(client: Any) -> Any:
  """Wraps generate_content and generate_content_stream to track cost."""
  if _safe_get(client, "_cost_instrumented", False):
    return client
  aio = _safe_get(client, "aio")
  models = _safe_get(aio, "models")
  if models is None:
    return client

  handler = _InstrumentedModelHandler(
      models.generate_content, models.generate_content_stream
  )
  models.generate_content = handler.generate_content
  models.generate_content_stream = handler.generate_content_stream
  try:
    object.__setattr__(client, "_cost_instrumented", True)
  except (AttributeError, TypeError):
    pass
  return client
