"""Benchmark image providers on a fixed set of real image slots.

Evaluates image search providers (Gemini image search, SerpApi Google Images,
Openverse, Brave) on a fixed set of image slots derived from production queries.

Workflow:
1. Builds (and caches in bench_slots.json) a slot set by running the production
   answer prompt on 8 queries (gemini-3.8-flash + Google Search grounding).
2. Resolves every slot with each provider through the SAME path as production
   (search -> download -> normalize -> visual re-rank -> dedupe, web only).
3. Judges each final image against its caption with gemini-3.8-flash.

Backs:
  - docs/third-party-comparison.md (Section 1: Image search alone: same 24 image slots)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").
  - SERPAPI_API_KEY: SerpApi key (required to benchmark SerpApi provider).
  - SERPAPI_NO_CACHE: Recommended set to 1 for fair, uncached latency.
  - BRAVE_API_KEY: Brave Search key (optional, to benchmark Brave provider).

Approximate cost:
  - Building slots (one-time, cached): 8 Gemini calls (~24 slots).
  - Per provider on 24 slots: 24 image searches (or SerpApi queries) +
    24 visual re-rank calls + 1 judge call per filled slot (~60-70 calls / provider).

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  export SERPAPI_API_KEY="my-serpapi-key"
  export SERPAPI_NO_CACHE=1
  python scripts/bench/bench_providers.py gemini openverse serpapi --out-dir /tmp
"""
import argparse
import asyncio
import base64
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
from app.textproc import SlotEvent, SlotMarkerParser

QUERIES = [
    "Best beginner hikes in Zion National Park",
    "Most iconic buildings designed by Zaha Hadid",
    "Birds you can spot in Central Park in spring",
    "Best street foods to try in Bangkok",
    "Japandi living room design ideas",
    "Latest flagship ZTE nubia smartphones",
    "How does a jet engine work?",
    "Top things to see in Kyoto in autumn",
]


async def generate_query_slots(q: str, sysprompt: str):
  """Runs the production answer prompt on a single query and extracts slots."""
  r = await client().aio.models.generate_content(
      model="gemini-3.8-flash",
      contents=q,
      config=types.GenerateContentConfig(
          system_instruction=sysprompt,
          tools=[types.Tool(google_search=types.GoogleSearch())],
          thinking_config=types.ThinkingConfig(thinking_level="LOW"),
      ),
  )
  p = SlotMarkerParser()
  evs = list(p.feed(r.text or "")) + list(p.flush())
  return [
      {"q": q, "query": e.query, "caption": e.caption}
      for e in evs
      if isinstance(e, SlotEvent)
  ]


async def build_slots(cache_path: str):
  if os.path.exists(cache_path):
    with open(cache_path) as f:
      return json.load(f)
  sysprompt = pipeline.SEARCH_SYSTEM_PROMPT.format(
      today=pipeline._today(), min_slots=2, max_slots=5
  )
  slots = [
      s
      for ss in await asyncio.gather(
          *(generate_query_slots(q, sysprompt) for q in QUERIES)
      )
      for s in ss
  ]
  with open(cache_path, "w") as f:
    json.dump(slots, f, indent=1)
  return slots


async def judge(caption, data_url):
  jpeg = base64.b64decode(data_url.split(",", 1)[1])
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
                  types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
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


class TimedSearchWrapper:
  """Instruments search_images to record search durations."""

  def __init__(self, orig_search):
    self.orig_search = orig_search
    self.search_times = []

  async def search(self, q, ctx):
    t = time.monotonic()
    out = await self.orig_search(q, ctx)
    self.search_times.append((time.monotonic() - t, len(out)))
    return out


class SlotResolver:
  """Resolves slots with concurrency control and query-level deduplication."""

  def __init__(self, sem):
    self.sem = sem
    self.used_by_q = {}

  async def resolve(self, s):
    async with self.sem:
      t = time.monotonic()
      img = await images.resolve_slot(
          s["query"],
          s["caption"],
          s["q"],
          "web",
          self.used_by_q.setdefault(s["q"], set()),
      )
      dt = time.monotonic() - t
    verdict = await judge(s["caption"], img.data_url) if img else None
    return dict(
        slot=s["query"],
        latency=dt,
        filled=img is not None,
        verdict=verdict,
        domain=img.domain if img else "",
        license=img.license if img else "",
    )


async def run_provider(name, slots):
  object.__setattr__(settings, "image_provider", name)
  object.__setattr__(
      settings, "enable_wikimedia", False
  )  # Isolate the provider itself.
  orig = images.search_images
  wrapper = TimedSearchWrapper(orig)
  images.search_images = wrapper.search

  resolver = SlotResolver(asyncio.Semaphore(6))
  try:
    rows = await asyncio.gather(*(resolver.resolve(s) for s in slots))
  finally:
    images.search_images = orig
  v = [r["verdict"] for r in rows if r["filled"]]
  lat = [r["latency"] for r in rows]
  st = [t for t, _ in wrapper.search_times]
  summary = dict(
      provider=name,
      slots=len(rows),
      filled=sum(r["filled"] for r in rows),
      fill_rate=round(sum(r["filled"] for r in rows) / len(rows), 3),
      correct=v.count("yes"),
      partial=v.count("partial"),
      wrong=v.count("no"),
      correct_of_all_slots=round(v.count("yes") / len(rows), 3),
      precision=round(v.count("yes") / max(len(v), 1), 3),
      search_p50=round(statistics.median(st), 2) if st else None,
      search_p90=round(sorted(st)[int(len(st) * 0.9) - 1], 2) if st else None,
      slot_p50=round(statistics.median(lat), 2),
      slot_p90=round(sorted(lat)[int(len(lat) * 0.9) - 1], 2),
      avg_candidates=(
          round(statistics.mean(n for _, n in wrapper.search_times), 1)
          if wrapper.search_times
          else 0
      ),
      empty_searches=sum(1 for _, n in wrapper.search_times if n == 0),
  )
  return summary, rows


async def main():
  parser = argparse.ArgumentParser(
      description="Benchmark image providers on fixed image slots."
  )
  parser.add_argument(
      "providers",
      nargs="*",
      default=[],
      help="Image providers to benchmark (default: all with credentials)",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output and cache files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  wanted = (
      args.providers
      or ["gemini", "openverse"]
      + (["serpapi"] if settings.serpapi_api_key else [])
      + (["brave"] if settings.brave_api_key else [])
  )
  cache_path = os.path.join(args.out_dir, "bench_slots.json")
  out_path = os.path.join(args.out_dir, "bench_providers.json")

  slots = await build_slots(cache_path)
  print(f"{len(slots)} slots from {len(QUERIES)} queries; providers: {wanted}")
  allres = {}
  for name in wanted:  # Sequential: providers share the global setting.
    summary, rows = await run_provider(name, slots)
    allres[name] = {"summary": summary, "rows": rows}
    print(json.dumps(summary))
  with open(out_path, "w") as f:
    json.dump(allres, f, indent=1)


if __name__ == "__main__":
  asyncio.run(main())
