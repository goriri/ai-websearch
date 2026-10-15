"""Search orchestration: turns a natural-language query into a stream of
interleaved text + image events.

Two modes:

* ``search`` (default): a Gemini text model grounded with Google Search writes
  the answer and inserts image *slots* (``[[IMAGE: query | caption]]``). Each
  slot is resolved concurrently while the text is still streaming: real web
  images are found with Gemini's Google image-search grounding, downloaded,
  visually verified by a small Gemini model and inlined. Fallbacks: Wikimedia
  Commons, then (optionally) Gemini image generation.
* ``visual``: Gemini's native interleaved text+image output (a Gemini image
  model, also grounded with Google Search) generates illustrations inline.
* ``aimode``: one SerpApi Google AI Mode call returns the answer, citations,
  images and related questions; no Gemini calls (see ``aimode.py``).
"""

import asyncio
import base64
import datetime
import json
import logging
import re
import time
from typing import Awaitable, Callable, Literal, Optional
from urllib.parse import urlparse

import httpx
import pydantic
from google.genai import types

from app import aimode, images, providers
from app.config import settings
from app.genai_client import client
from app.textproc import (
    SlotEvent,
    SlotMarkerParser,
    Support,
    TextEvent,
    clean_title,
    insert_citations,
    slot_token,
    strip_slot_markers,
)

log = logging.getLogger(__name__)

Emit = Callable[[str, dict], Awaitable[None]]


class Turn(pydantic.BaseModel):
  role: Literal["user", "model"]
  text: str = pydantic.Field(max_length=20000)


class SearchRequest(pydantic.BaseModel):
  query: str = pydantic.Field(min_length=1)
  mode: Literal["search", "visual", "aimode"] = "search"
  arch: Optional[Literal["gemini", "thirdparty", "aimode"]] = None
  model: Literal["fast", "pro"] = "fast"
  images: Literal["auto", "web", "ai"] = "web"
  history: list[Turn] = pydantic.Field(default_factory=list, max_length=20)
  # Optional user-supplied image as a data URL (multimodal query).
  image: Optional[str] = None


SEARCH_SYSTEM_PROMPT = """\
You are Prism, an intelligent multimodal search engine. Today is {today}.

Research the user's request with Google Search, then write a helpful, accurate,
well-organized answer in GitHub-flavored Markdown that reads like a rich,
magazine-style search result with text and images interleaved.

Structure:
- Open with a direct 1-3 sentence answer (no heading before it).
- Then organize details under short `###` headings; use bullet lists, numbered
  steps or a compact table when they genuinely help (e.g. comparisons).
- Be concrete: names, numbers, dates, prices, locations. Prefer recent info.
- Do not add a "Sources" or "References" section; the UI shows sources.

Images: insert image slots that the system will fill with real photos from the
web. Write each slot ALONE on its own line, exactly in this form:
[[IMAGE: <image search query> | <short caption>]]
Rules for slots:
- Place a slot right after the paragraph or section it illustrates, never
  inside a list, table or heading, and never as the very first line.
- The query must be specific, visual and self-contained (include proper nouns,
  place, model, year), e.g. "Angels Landing chains section Zion" rather than
  "the trail". The caption (max ~12 words) says what the image shows.
- Use {min_slots}-{max_slots} slots for visual topics (places, products,
  animals, food, people, art, how-to, events, design). Use 0-1 for abstract,
  numeric, coding or purely factual questions. Each slot must show something
  different.
- If the user attached an image, use it to understand the request.
"""

VISUAL_SYSTEM_PROMPT = """\
You are Prism Visual, a search assistant that explains topics with text and
generated illustrations interleaved. Today is {today}.

Use Google Search to get the facts right, then write a clear, well-structured
Markdown answer (start with a 1-2 sentence direct answer, then `###` sections).
After each key section, generate ONE illustrative image (photo-realistic scene,
diagram, product shot, step illustration, etc.) that helps the reader
understand that section. Generate {min_slots}-{max_slots} images total. Never put
text inside images unless it is a labeled diagram. Never refer to the images by
number or name in the text (no "image 1", "see image below"); the images sit
right next to the text they illustrate. Do not add a sources section.
"""


def _today() -> str:
  return datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y")


def _decode_upload(data_url: str) -> types.Part:
  m = re.match(r"^data:(image/[\w.+-]+);base64,(.+)$", data_url, re.S)
  if not m:
    raise ValueError("Attached image must be a base64 data URL.")
  raw = base64.b64decode(m.group(2), validate=False)
  if len(raw) > settings.max_upload_bytes:
    raise ValueError("Attached image is too large (max 7 MB).")
  try:
    jpeg, _, _ = images.normalize_image(raw, 1536)
  except ValueError as e:
    raise ValueError(f"Could not read attached image ({e}).") from e
  return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


def build_contents(req: SearchRequest) -> list[types.Content]:
  contents = []
  for turn in req.history[-8:]:
    text = re.sub(r"@@(SLOT|CITE):[^@]*@@", "", strip_slot_markers(turn.text))
    if text.strip():
      contents.append(
          types.Content(role=turn.role, parts=[types.Part.from_text(text=text)])
      )
  parts = []
  if req.image:
    parts.append(_decode_upload(req.image))
  parts.append(types.Part.from_text(text=req.query))
  contents.append(types.Content(role="user", parts=parts))
  return contents


_TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title>", re.I | re.S)


async def _resolve_source(url: str) -> tuple[str, str]:
  """Resolves a grounding redirect link to (final_url, page_title).

  Vertex grounding chunks only carry the domain as the title, so we peek at
  the first bytes of the page to show a real title. Best effort, time-boxed.
  """
  final, title = url, ""
  try:
    async with asyncio.timeout(3.0):
      async with images.http().stream(
          "GET", url, headers={"Accept": "text/html,*/*;q=0.5"}
      ) as r:
        final = str(r.url)
        if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
          buf = b""
          async for part in r.aiter_bytes():
            buf += part
            m = _TITLE_RE.search(buf)
            if m or len(buf) > 96_000:
              break
          if m:
            enc = r.encoding or "utf-8"
            title = clean_title(m.group(1).decode(enc, errors="replace"))
            title = re.sub(r"\s+", " ", title)[:160]
  except (httpx.HTTPError, TimeoutError, LookupError):
    pass
  if "vertexaisearch" in (urlparse(final).hostname or ""):
    final = url
  return final, title


async def _sources_from_grounding(gm: Optional[types.GroundingMetadata]):
  """Returns (sources, supports, search_entry_point_html, queries)."""
  if gm is None:
    return [], [], "", []
  sources = []
  for i, ch in enumerate(gm.grounding_chunks or []):
    if ch.web and ch.web.uri:
      sources.append({
          "n": i + 1,
          "title": clean_title(ch.web.title) or (ch.web.domain or ""),
          "url": ch.web.uri,
          "domain": ch.web.domain or clean_title(ch.web.title),
      })
  resolved = await asyncio.gather(*(_resolve_source(s["url"]) for s in sources))
  for s, (url, title) in zip(sources, resolved):
    s["url"] = url
    host = urlparse(url).hostname or ""
    if host and "vertexaisearch" not in host:
      s["domain"] = host.removeprefix("www.")
    if title:
      s["title"] = title
  supports = [
      Support(
          text=(sup.segment.text if sup.segment else "") or "",
          chunk_indices=list(sup.grounding_chunk_indices or []),
      )
      for sup in (gm.grounding_supports or [])
  ]
  sep = ""
  if gm.search_entry_point and gm.search_entry_point.rendered_content:
    sep = gm.search_entry_point.rendered_content
  queries = list(gm.web_search_queries or [])
  return sources, supports, sep, queries


async def related_questions(query: str, answer: str) -> list[str]:
  try:
    resp = await client().aio.models.generate_content(
        model=settings.lite_model,
        contents=(
            f"User searched: {query}\n\nAnswer shown (truncated):\n"
            f"{answer[:2500]}\n\nSuggest 4 short, diverse follow-up searches"
            " the user is likely to want next (max 10 words each, same"
            " language as the user)."
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        ),
    )
    out = json.loads(resp.text)
    return [str(q).strip() for q in out if str(q).strip()][:4]
  except Exception as e:  # pylint: disable=broad-except
    log.info("related questions failed: %s", e)
    return []


def _thinking(model: str, pro: bool) -> Optional[types.ThinkingConfig]:
  if model.startswith("gemini-2"):
    return types.ThinkingConfig(thinking_budget=0 if not pro else -1)
  return types.ThinkingConfig(thinking_level="HIGH" if pro else "LOW")


_CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\()")


def citations_from_brackets(markdown: str, n_sources: int) -> str:
  """Converts model-written [1] / [2, 3] citations into @@CITE tokens."""
  def repl(m):
    nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
    nums = [n for n in nums if 1 <= n <= n_sources]
    return f"@@CITE:{','.join(map(str, nums))}@@" if nums else m.group(0)
  return _CITE_RE.sub(repl, markdown)


def third_party_context(results: list[dict]) -> str:
  lines = ["Web search results (cite them inline as [n]):"]
  for i, r in enumerate(results, 1):
    lines.append(f"[{i}] {r['title']} ({r['domain']})\n{r['snippet']}\n{r['url']}")
  return "\n\n".join(lines)


THIRD_PARTY_RULES = (
    "\nYou are given numbered web search results from a search API instead of a"
    " search tool. Base the answer on them (and well-established knowledge);"
    " cite supporting results inline right after the sentence as [n] or [n, m]."
    " Don't invent sources.\n"
)


async def run_search(req: SearchRequest, emit: Emit) -> None:
  image_tasks: list[asyncio.Task] = []
  try:
    await _run_search(req, emit, image_tasks)
  finally:
    # If the client went away (task cancelled), stop pending image work too.
    for t in image_tasks:
      if not t.done():
        t.cancel()


async def _run_search(
    req: SearchRequest, emit: Emit, image_tasks: list[asyncio.Task]
) -> None:
  t0 = time.monotonic()
  pro = req.model == "pro"
  primary = settings.pro_model if pro else settings.answer_model
  models = [primary] + ([settings.fallback_model] if settings.fallback_model != primary else [])
  contents = build_contents(req)
  sysprompt = SEARCH_SYSTEM_PROMPT.format(
      today=_today(), min_slots=2, max_slots=min(5, settings.max_image_slots)
  )

  await emit("status", {"stage": "search", "message": "Searching the web"})

  web = "serpapi" if req.arch == "thirdparty" else ("gemini" if req.arch == "gemini" else settings.web_provider)
  image_provider = "serpapi" if req.arch == "thirdparty" else ("gemini" if req.arch == "gemini" else settings.image_provider)

  # Third-party web search (WEB_PROVIDER=serpapi): fetch results first and pass
  # them to Gemini as context instead of using the Google Search tool.
  tp_results: Optional[list[dict]] = None
  if web in providers.WEB_PROVIDERS:
    try:
      tp_results = await providers.WEB_PROVIDERS[web](req.query)
    except Exception as e:  # pylint: disable=broad-except
      log.warning("%s web search failed (%s); using Gemini grounding", web, e)
  if tp_results:
    sysprompt = sysprompt.replace(
        "Research the user's request with Google Search, then write",
        "Using the provided web search results, write") + THIRD_PARTY_RULES
    last = contents[-1]
    last.parts.append(types.Part.from_text(text=third_party_context(tp_results)))
  search_tools = None if tp_results else [types.Tool(google_search=types.GoogleSearch())]

  sem = asyncio.Semaphore(settings.image_concurrency)
  used_images: set[str] = set()
  assembled: list[str] = []
  raw_text: list[str] = []
  n_slots = 0

  async def fill_slot(slot_id: str, slot: SlotEvent) -> None:
    async with sem:
      try:
        img = await images.resolve_slot(
            slot.query, slot.caption, req.query, req.images, used_images,
            provider=image_provider,
        )
      except Exception:  # pylint: disable=broad-except
        log.exception("slot %s failed", slot_id)
        img = None
    if img is None:
      await emit("image_failed", {"id": slot_id})
    else:
      await emit("image", {"id": slot_id, "caption": slot.caption} | img.to_event())

  gm = None
  used_model = None
  for model in models:
    parser = SlotMarkerParser()
    emitted_any = False
    try:
      stream = await client().aio.models.generate_content_stream(
          model=model,
          contents=contents,
          config=types.GenerateContentConfig(
              system_instruction=sysprompt,
              tools=search_tools,
              thinking_config=_thinking(model, pro),
          ),
      )

      async def handle(events):
        nonlocal emitted_any, n_slots
        for ev in events:
          if isinstance(ev, TextEvent):
            if ev.text:
              emitted_any = True
              assembled.append(ev.text)
              await emit("text", {"delta": ev.text})
          elif isinstance(ev, SlotEvent):
            if n_slots >= settings.max_image_slots:
              continue
            n_slots += 1
            slot_id = f"s{n_slots}"
            emitted_any = True
            assembled.append(slot_token(slot_id))
            if n_slots == 1:
              await emit("status", {"stage": "images", "message": "Finding images"})
            await emit("slot", {"id": slot_id, "query": ev.query, "caption": ev.caption})
            image_tasks.append(asyncio.create_task(fill_slot(slot_id, ev)))

      async for chunk in stream:
        for cand in chunk.candidates or []:
          if cand.grounding_metadata:
            gm = cand.grounding_metadata
          for part in (cand.content.parts if cand.content else None) or []:
            if part.text and not part.thought:
              raw_text.append(part.text)
              await handle(parser.feed(part.text))
      await handle(parser.flush())
      used_model = model
      break
    except Exception as e:  # pylint: disable=broad-except
      log.exception("answer model %s failed", model)
      if emitted_any or model == models[-1]:
        await emit("error", {"message": f"The answer model failed: {e}"})
        break
      await emit("status", {"stage": "retry", "message": "Retrying with a backup model"})

  answer_md = "".join(assembled)
  if tp_results:
    sources = [{"n": i, "title": r["title"] or r["domain"], "url": r["url"],
                "domain": r["domain"]} for i, r in enumerate(tp_results, 1)]
    sep, queries = "", [req.query]
    final_md = citations_from_brackets(answer_md, len(sources))
  else:
    sources, supports, sep, queries = await _sources_from_grounding(gm)
    final_md = insert_citations(answer_md, supports)
  await emit("final", {
      "markdown": final_md,
      "sources": sources,
      "search_entry_point": sep,
      "queries": queries,
      "model": used_model,
  })

  related_task = asyncio.create_task(
      related_questions(req.query, strip_slot_markers("".join(raw_text)))
  )
  if image_tasks:
    await asyncio.gather(*image_tasks, return_exceptions=True)
  await emit("related", {"questions": await related_task})
  await emit("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})


async def run_visual(req: SearchRequest, emit: Emit) -> None:
  """Native interleaved text + generated images in a single Gemini call."""
  t0 = time.monotonic()
  contents = build_contents(req)
  await emit("status", {"stage": "search", "message": "Researching and illustrating"})
  assembled: list[str] = []
  n_img = 0
  gm = None
  try:
    stream = await client().aio.models.generate_content_stream(
        model=settings.image_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=VISUAL_SYSTEM_PROMPT.format(
                today=_today(), min_slots=2, max_slots=4
            ),
            response_modalities=["TEXT", "IMAGE"],
            tools=[types.Tool(google_search=types.GoogleSearch())],
            image_config=types.ImageConfig(image_size="1K"),
        ),
    )
    async for chunk in stream:
      for cand in chunk.candidates or []:
        if cand.grounding_metadata:
          gm = cand.grounding_metadata
        for part in (cand.content.parts if cand.content else None) or []:
          if part.thought:
            continue
          if part.text:
            assembled.append(part.text)
            await emit("text", {"delta": part.text})
          elif part.inline_data and part.inline_data.data:
            try:
              jpeg, w, h = await asyncio.to_thread(
                  images.normalize_image, part.inline_data.data,
                  settings.image_max_px,
              )
            except ValueError:
              continue
            n_img += 1
            slot_id = f"g{n_img}"
            assembled.append(slot_token(slot_id))
            await emit("slot", {"id": slot_id, "query": "", "caption": ""})
            await emit("image", {
                "id": slot_id, "caption": "", "src": images.to_data_url(jpeg),
                "width": w, "height": h, "kind": "generated",
                "title": "Generated with Gemini", "domain": "AI-generated",
                "source_url": "", "license": "",
            })
  except Exception as e:  # pylint: disable=broad-except
    log.exception("visual mode failed")
    await emit("error", {"message": f"Visual generation failed: {e}"})

  answer_md = "".join(assembled)
  sources, supports, sep, queries = await _sources_from_grounding(gm)
  await emit("final", {
      "markdown": insert_citations(answer_md, supports),
      "sources": sources,
      "search_entry_point": sep,
      "queries": queries,
      "model": settings.image_model,
  })
  await emit("related", {
      "questions": await related_questions(req.query, answer_md)
  })
  await emit("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})


AI_MODE_MODEL = "serpapi:google_ai_mode"
# Google AI Mode inline images are small gstatic thumbnails (often < 240 px).
AI_MODE_MIN_PX = 100
_SLOT_TOKEN_RE = re.compile(r"\n\n@@SLOT:([\w-]+)@@\n\n")


async def run_ai_mode(req: SearchRequest, emit: Emit) -> None:
  """One SerpApi Google AI Mode call: answer, citations, images, related.

  Makes no Gemini calls. The rendered markdown is streamed as `text` deltas
  (one per paragraph) with a `slot` event at each image position, exactly as
  the client assembles it for the other modes; slot images are downloaded
  concurrently and sent as `image` / `image_failed` events.
  """
  image_tasks: list[asyncio.Task] = []
  try:
    await _run_ai_mode(req, emit, image_tasks)
  finally:
    for t in image_tasks:
      if not t.done():
        t.cancel()


async def _run_ai_mode(
    req: SearchRequest, emit: Emit, image_tasks: list[asyncio.Task]
) -> None:
  t0 = time.monotonic()

  def elapsed() -> dict:
    return {"elapsed_ms": int((time.monotonic() - t0) * 1000)}

  await emit("status", {"stage": "search", "message": "Asking Google AI Mode"})
  try:
    resp = await providers.serpapi_ai_mode(req.query)
  except (providers.ProviderError, httpx.HTTPError, ValueError) as e:
    log.warning("AI Mode request failed: %s", e)
    await emit("error", {"message": f"Google AI Mode failed: {e}"})
    await emit("done", elapsed())
    return
  rendered = aimode.render(resp, max_images=settings.max_image_slots)
  if not rendered.markdown.strip():
    await emit("error", {"message": "Google AI Mode returned no answer."})
    await emit("done", elapsed())
    return

  sem = asyncio.Semaphore(settings.image_concurrency)

  async def fetch(slot: dict) -> None:
    url = slot["image_url"]
    host = urlparse(url).hostname or ""
    cand = images.Candidate(
        image_url=url, kind="web", domain=host, source_url=url
    )
    async with sem:
      try:
        got = await images.download(cand, min_px=AI_MODE_MIN_PX)
      except (httpx.HTTPError, ValueError, OSError) as e:
        log.warning("AI Mode image %s failed: %s", slot["id"], e)
        got = None
    if got is None:
      await emit("image_failed", {"id": slot["id"]})
      return
    jpeg, w, h = got
    img = images.ResolvedImage(
        data_url=images.to_data_url(jpeg), width=w, height=h, kind="web",
        source_url=cand.source_url, title="", domain=host,
    )
    await emit("image", {"id": slot["id"], "caption": slot["caption"]}
               | img.to_event())

  slots = {s["id"]: s for s in rendered.slots}
  pieces = _SLOT_TOKEN_RE.split(rendered.markdown)
  for i, piece in enumerate(pieces):
    if i % 2:  # Slot id.
      slot = slots.get(piece)
      if slot is None:
        continue
      await emit("slot", {"id": piece, "query": "", "caption": slot["caption"]})
      image_tasks.append(asyncio.create_task(fetch(slot)))
      continue
    for delta in re.split(r"(?<=\n\n)", piece):
      if delta:
        await emit("text", {"delta": delta})

  await emit("final", {
      "markdown": rendered.markdown,
      "sources": rendered.sources,
      "search_entry_point": "",
      "queries": [req.query],
      "model": AI_MODE_MODEL,
  })
  if image_tasks:
    await asyncio.gather(*image_tasks, return_exceptions=True)
  await emit("related", {"questions": rendered.related})
  await emit("done", elapsed())
