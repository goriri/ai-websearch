"""Benchmark: can the Prism pipeline use fewer Gemini calls?

Evaluates Gemini call-reduction architectures:
  - current: text model streams answer with slots; per slot: 1 image-search call
    + 1 visual re-rank call (~2N+1 calls, + related questions)
  - current_norerank: current pipeline with visual re-ranking disabled
  - batched: text model streams answer with slots; then ONE image-search call
    for all slot queries + ONE vision "assign" call (3 calls)
  - single: ONE image-model call with web_search+image_search writes the answer
    with slots and retrieves photos; ONE vision "assign" call; an extra
    batched image-search call only if no images came back (2-3 calls)

Each final (caption, image) pair is judged by gemini-3.8-flash (yes/partial/no).

Backs:
  - docs/third-party-comparison.md (Cost per query / calls per query trade-offs)
  - docs/image-search-grounding.md

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").
  - NORERANK: (optional) Set to 1 to run the current_norerank variant.

Approximate cost:
  - current: ~10-15 calls per query (4 queries × 3 runs = ~120-180 Gemini calls)
  - batched: ~3 calls per query (4 queries × 3 runs = ~36 Gemini calls)
  - single: 2-3 calls per query (4 queries × 3 runs = ~24-36 Gemini calls)
  - Judging: 1 call per filled image

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/bench_calls.py --out-dir /tmp
"""
import argparse
import asyncio
import base64
import contextvars
import json
import os
import pathlib
import statistics
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
  sys.exit("Error: GOOGLE_CLOUD_PROJECT environment variable is required.")

from google.genai import types
from app import images, pipeline
from app.config import settings
from app.genai_client import client
from app.textproc import SlotEvent, SlotMarkerParser, clean_title

IMG_MODEL = "gemini-3.1-flash-image"
LITE = "gemini-3.1-flash-lite"
QUERIES = [
    "Best beginner hikes in Zion National Park",
    "Most iconic buildings designed by Zaha Hadid",
    "Birds you can spot in Central Park in spring",
    "Best street foods to try in Bangkok",
]


class CallCounter:
  """Counts Gemini model calls tagged by benchmark variant and query."""

  def __init__(self):
    self.calls = {}
    self.tag = contextvars.ContextVar("tag", default="")
    self.orig_gc = None
    self.orig_gcs = None

  def install(self):
    m = client().aio.models
    self.orig_gc, self.orig_gcs = m.generate_content, m.generate_content_stream
    m.generate_content = self.generate_content
    m.generate_content_stream = self.generate_content_stream

  async def generate_content(self, *a, **kw):
    t = self.tag.get()
    if t:
      self.calls[t] = self.calls.get(t, 0) + 1
    return await self.orig_gc(*a, **kw)

  async def generate_content_stream(self, *a, **kw):
    t = self.tag.get()
    if t:
      self.calls[t] = self.calls.get(t, 0) + 1
    return await self.orig_gcs(*a, **kw)


COUNTER = CallCounter()


def chunks_to_candidates(gm):
  out = []
  for ch in (gm.grounding_chunks or []) if gm else []:
    if ch.image and ch.image.image_uri:
      out.append(
          images.Candidate(
              image_url=ch.image.image_uri,
              source_url=ch.image.source_uri or "",
              title=clean_title(ch.image.title),
              domain=ch.image.domain or "",
          )
      )
  return out


def _short(q, n=6):
  return " ".join(q.split()[:n])


async def _img_search_once(prompt):
  r = await client().aio.models.generate_content(
      model=IMG_MODEL,
      contents=prompt,
      config=types.GenerateContentConfig(
          response_modalities=["TEXT", "IMAGE"],
          tools=[
              types.Tool(
                  google_search=types.GoogleSearch(
                      search_types=types.SearchTypes(
                          image_search=types.ImageSearch()
                      )
                  )
              )
          ],
      ),
  )
  return chunks_to_candidates(r.candidates[0].grounding_metadata)


async def batch_image_search(slots, context):
  """ONE image-search call for all slots; one retry with other wording if empty."""
  subjects = [_short(s.query) for s in slots]
  prompts = [
      (
          "What do these look like? "
          + "; ".join(subjects)
          + ". Look each one up with Google image search and describe what the"
          " real photos show, one sentence each. Answer in text only."
      ),
      (
          "Search Google Images for photos of each of: "
          + "; ".join(subjects)
          + ". Then describe each photo you found in one sentence. Text only."
      ),
  ]
  for p in prompts:
    cands = await _img_search_once(p)
    if cands:
      return cands
  return []


async def download_pool(cands, cap=16):
  seen, uniq = set(), []
  for c in cands:
    if c.image_url not in seen:
      seen.add(c.image_url)
      uniq.append(c)
  uniq = uniq[:cap]
  res = await asyncio.gather(*(images.download(c) for c in uniq))
  return [(c, r[0]) for c, r in zip(uniq, res) if r]


async def assign(slots, pool):
  """ONE multimodal call: map each slot caption to the best candidate (or -1)."""
  if not slots or not pool:
    return {}
  parts = [
      types.Part.from_text(
          text=(
              "Assign photos to the image slots of a search results page. Each"
              " slot needs a photo that accurately shows its caption. Each"
              " photo may be used at most once. Reject photos that are mostly"
              " text, logos, collages, screenshots, or show a different subject"
              " (use -1).\n\nSlots:\n"
              + "\n".join(
                  f"- {s.id}: {s.caption} (search: {s.query})" for s in slots
              )
              + f"\n\nCandidate photos numbered 0..{len(pool) - 1} follow."
          )
      )
  ]
  for i, (c, jpeg) in enumerate(pool):
    small, _, _ = images.normalize_image(jpeg, 320)
    parts += [
        types.Part.from_text(text=f"Photo {i} ({c.domain}):"),
        types.Part.from_bytes(data=small, mime_type="image/jpeg"),
    ]
  r = await client().aio.models.generate_content(
      model=LITE,
      contents=[types.Content(role="user", parts=parts)],
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema={
              "type": "ARRAY",
              "items": {
                  "type": "OBJECT",
                  "properties": {
                      "slot": {"type": "STRING"},
                      "photo": {"type": "INTEGER"},
                  },
                  "required": ["slot", "photo"],
              },
          },
          thinking_config=types.ThinkingConfig(thinking_level="LOW"),
      ),
  )
  used, out = set(), {}
  for a in json.loads(r.text):
    p = int(a["photo"])
    if 0 <= p < len(pool) and p not in used:
      used.add(p)
      out[a["slot"]] = pool[p]
  return out


class Slot:

  def __init__(self, id, ev):
    self.id, self.query, self.caption = id, ev.query, ev.caption


def parse_slots(text):
  p = SlotMarkerParser()
  slots = []
  for ev in list(p.feed(text)) + list(p.flush()):
    if isinstance(ev, SlotEvent):
      slots.append(Slot(f"s{len(slots) + 1}", ev))
  return slots


class CurrentPipelineCollector:
  """Collects streaming events for current pipeline variant."""

  def __init__(self, t0):
    self.t0 = t0
    self.first = None
    self.result = {}
    self.slots = {}

  async def emit(self, ev, d):
    if ev == "text" and self.first is None:
      self.first = time.monotonic() - self.t0
    if ev == "slot":
      self.slots[d["id"]] = d["caption"]
    if ev == "image":
      jpeg = base64.b64decode(d["src"].split(",", 1)[1])
      self.result[d["id"]] = (self.slots[d["id"]], jpeg, d["kind"])


async def v_current(q):
  t0 = time.monotonic()
  collector = CurrentPipelineCollector(t0)
  await pipeline.run_search(
      pipeline.SearchRequest(query=q, images="web"), collector.emit
  )
  return dict(
      ttft=collector.first,
      total=time.monotonic() - t0,
      n_slots=len(collector.slots),
      pairs=list(collector.result.values()),
  )


async def stream_answer(q):
  """Text model answer with slots (same prompt as production)."""
  sysprompt = pipeline.SEARCH_SYSTEM_PROMPT.format(
      today=pipeline._today(), min_slots=2, max_slots=5
  )
  t0 = time.monotonic()
  first = None
  text = ""
  stream = await client().aio.models.generate_content_stream(
      model="gemini-3.8-flash",
      contents=q,
      config=types.GenerateContentConfig(
          system_instruction=sysprompt,
          tools=[types.Tool(google_search=types.GoogleSearch())],
          thinking_config=types.ThinkingConfig(thinking_level="LOW"),
      ),
  )
  async for ch in stream:
    for c in ch.candidates or []:
      for p in (c.content.parts if c.content else None) or []:
        if p.text and not p.thought:
          first = first or (time.monotonic() - t0)
          text += p.text
  return text, first


async def v_batched(q):
  t0 = time.monotonic()
  text, first = await stream_answer(q)
  slots = parse_slots(text)
  pool = (
      await download_pool(await batch_image_search(slots, q)) if slots else []
  )
  m = await assign(slots, pool)
  pairs = [(s.caption, m[s.id][1], "web") for s in slots if s.id in m]
  return dict(
      ttft=first, total=time.monotonic() - t0, n_slots=len(slots), pairs=pairs
  )


SINGLE_SYS = """You are a multimodal search engine. Step 1: run Google WEB search queries to
collect facts. Step 2: run Google IMAGE search queries to find a real photo of each item you will
discuss. You must do both steps. Then write a Markdown answer (1-3 sentence direct answer, then
### sections). Right after each section, on its own line, write an image slot
[[IMAGE: <image search query> | <short caption of what the photo should show>]]. 2-5 slots.
Text only; never generate images."""


async def v_single(q):
  t0 = time.monotonic()
  r = await client().aio.models.generate_content(
      model=IMG_MODEL,
      contents=q,
      config=types.GenerateContentConfig(
          system_instruction=SINGLE_SYS,
          response_modalities=["TEXT", "IMAGE"],
          tools=[
              types.Tool(
                  google_search=types.GoogleSearch(
                      search_types=types.SearchTypes(
                          web_search=types.WebSearch(),
                          image_search=types.ImageSearch(),
                      )
                  )
              )
          ],
      ),
  )
  first = time.monotonic() - t0  # non-streamed: text only available at the end
  c = r.candidates[0]
  text = "".join(p.text or "" for p in c.content.parts if not p.thought)
  gm = c.grounding_metadata
  slots = parse_slots(text)
  cands = chunks_to_candidates(gm)
  both = bool(gm and gm.web_search_queries) and bool(cands)
  if len(cands) < len(
      slots
  ):  # image search didn't run (or too few): one batched fallback call
    cands += await batch_image_search(slots, q)
  pool = await download_pool(cands)
  m = await assign(slots, pool)
  pairs = [(s.caption, m[s.id][1], "web") for s in slots if s.id in m]
  n_web = sum(1 for ch in (gm.grounding_chunks or []) if ch.web) if gm else 0
  return dict(
      ttft=first,
      total=time.monotonic() - t0,
      n_slots=len(slots),
      pairs=pairs,
      both_searches=both,
      web_sources=n_web,
  )


async def judge(caption, jpeg):
  r = await client().aio.models.generate_content(
      model="gemini-3.8-flash",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=(
                          f"Caption: {caption}\nDoes this photo accurately"
                          " show what the caption describes (right"
                          " subject/place/thing)? Answer exactly one word: yes,"
                          " partial, or no."
                      )
                  ),
                  types.Part.from_bytes(
                      data=images.normalize_image(jpeg, 512)[0],
                      mime_type="image/jpeg",
                  ),
              ],
          )
      ],
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(thinking_level="LOW")
      ),
  )
  w = (r.text or "").strip().lower()
  return (
      "yes"
      if w.startswith("yes")
      else "partial"
      if w.startswith("partial")
      else "no"
  )


async def run_variant(name, fn, q):
  tag = f"{name}:{q}"
  COUNTER.tag.set(tag)
  try:
    res = await fn(q)
  except Exception as e:
    return dict(variant=name, q=q, error=str(e)[:200])
  res["calls"] = COUNTER.calls.get(tag, 0)
  COUNTER.tag.set("")
  verdicts = await asyncio.gather(
      *(
          asyncio.create_task(judge(c, j), name="judge")
          for c, j, _ in res.pop("pairs")
      )
  )
  res.update(
      variant=name,
      q=q,
      filled=len(verdicts),
      yes=verdicts.count("yes"),
      partial=verdicts.count("partial"),
      no=verdicts.count("no"),
  )
  return res


async def main():
  parser = argparse.ArgumentParser(
      description="Benchmark fewer-Gemini-calls pipeline variants."
  )
  parser.add_argument(
      "--variants",
      nargs="*",
      default=None,
      choices=["current", "current_norerank", "batched", "single"],
      help="Pipeline variants to test (default: current, or current_norerank if NORERANK=1)",
  )
  parser.add_argument(
      "--norerank",
      action="store_true",
      default=bool(os.environ.get("NORERANK")),
      help="Disable image re-ranking",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "bench_calls.json")

  COUNTER.install()
  if args.norerank:
    object.__setattr__(settings, "rerank_images", False)

  all_funcs = {
      "current": v_current,
      "current_norerank": v_current,
      "batched": v_batched,
      "single": v_single,
  }

  if args.variants:
    variants = {v: all_funcs[v] for v in args.variants}
  else:
    default_name = "current_norerank" if args.norerank else "current"
    variants = {default_name: v_current}

  queries = [f"{q} " + " " * i for i in range(3) for q in QUERIES]  # unique tags per run
  jobs = [
      asyncio.create_task(run_variant(n, f, q))
      for q in queries
      for n, f in variants.items()
  ]
  rows = await asyncio.gather(*jobs)
  with open(out_path, "w") as f:
    json.dump(rows, f, indent=1, default=str)
  for r in rows:
    print(
        json.dumps({
            k: (round(v, 1) if isinstance(v, float) else v)
            for k, v in r.items()
        })
    )
  print("\nSUMMARY")
  for n in variants:
    rs = [r for r in rows if r["variant"] == n and "error" not in r]
    if not rs:
      continue
    tot_slots = sum(r["n_slots"] for r in rs)
    filled = sum(r["filled"] for r in rs)
    yes = sum(r["yes"] for r in rs)
    part = sum(r["partial"] for r in rs)
    print(
        f"{n:8s} calls/query={statistics.mean(r['calls'] for r in rs):.1f} "
        f"ttft={statistics.mean(r['ttft'] or 0 for r in rs):.1f}s"
        f" total={statistics.mean(r['total'] for r in rs):.1f}s "
        f"slots={tot_slots} filled={filled} correct(yes)={yes} partial={part} "
        f"fill_rate={filled / max(tot_slots, 1):.0%}"
        f" precision={yes / max(filled, 1):.0%}"
        + (
            f" both_searches={sum(r.get('both_searches', 0) for r in rs)}/{len(rs)}"
            if n == "single"
            else ""
        )
    )


if __name__ == "__main__":
  asyncio.run(main())
