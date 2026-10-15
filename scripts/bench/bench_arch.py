"""End-to-end architecture benchmark: Gemini pipeline vs SerpApi Google AI Mode.

Compares three architectures:
  - Architecture A (Gemini-native default): streamed answer via gemini-3.8-flash
    with Grounding with Google Search, image slots resolved concurrently via
    gemini-3.1-flash-image image search + gemini-3.1-flash-lite visual re-ranking.
  - Architecture B (Third-party search): SerpApi web results passed to
    gemini-3.8-flash, image slots resolved via SerpApi Google Images.
  - Architecture C (SerpApi Google AI Mode): single non-streaming call to
    SerpApi engine=google_ai_mode.

For each query and architecture records: time to first text, total time,
#sources, #image slots, #images shown, and judged image relevance (image vs the
paragraph it sits under; gemini-3.8-flash judge), plus Gemini/SerpApi call
counts. Uses the production run_search / run_ai_mode code paths.

Backs:
  - docs/third-party-comparison.md (Section 2: End to end: 8 queries)
  - README.md (Three architectures table)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").
  - SERPAPI_API_KEY: SerpApi key (required for Architecture B and C; if unset,
    only Architecture A is evaluated).
  - SERPAPI_NO_CACHE: Set to 1 to bypass SerpApi cache for fair latency numbers.

Approximate cost:
  - Architecture A: ~10 Gemini calls / query (~80 Gemini calls total across 8 queries).
  - Architecture B: ~5 SerpApi searches + ~6 Gemini calls / query (~40 SerpApi + ~48 Gemini calls).
  - Architecture C: 1 SerpApi search / query (8 SerpApi searches total, 0 Gemini calls).
  - Judging: 1 gemini-3.8-flash call per displayed image (~20-40 calls).

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  export SERPAPI_API_KEY="my-serpapi-key"
  export SERPAPI_NO_CACHE=1
  python scripts/bench/bench_arch.py --out-dir /tmp
"""
import argparse
import asyncio
import base64
import json
import os
import pathlib
import re
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
from app import pipeline
from app.config import settings
from app.genai_client import client

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


class SearchCollector:
  """Collects streaming events during search."""

  def __init__(self):
    self.t0 = time.monotonic()
    self.ttft = None
    self.md = ""
    self.slots = []
    self.imgs = {}
    self.final = {}

  async def emit(self, ev, d):
    if ev == "text":
      self.ttft = self.ttft or time.monotonic() - self.t0
      self.md += d["delta"]
    elif ev == "slot":
      self.slots.append((d["id"], self.md))
      self.md += f"@@SLOT:{d['id']}@@"
    elif ev == "image":
      self.imgs[d["id"]] = base64.b64decode(d["src"].split(",", 1)[1])
    elif ev == "final":
      self.final = d


async def judge(context, jpeg):
  r = await client().aio.models.generate_content(
      model="gemini-3.8-flash",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=(
                          f"Text next to the photo:\n{context[-600:]}\n\nIs"
                          " this photo relevant to and a correct illustration"
                          " of that text? One word: yes, partial, or no."
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


async def run(arch, q):
  collector = SearchCollector()
  req = pipeline.SearchRequest(
      query=q,
      images="web",
      mode="aimode" if arch == "aimode" else "search",
      arch=arch if arch in ("gemini", "thirdparty", "aimode", "vertexsearch") else None,
  )
  # "thirdparty": same Prism pipeline, but SerpApi for web results + Google Images.
  runner = pipeline.run_ai_mode if arch == "aimode" else pipeline.run_search
  try:
    await runner(req, collector.emit)
  except Exception as e:
    return dict(arch=arch, q=q, error=str(e)[:200])
  total = time.monotonic() - collector.t0
  ctx = {
      sid: re.sub(r"@@\w+:[^@]*@@", "", before)
      for sid, before in collector.slots
  }
  verdicts = await asyncio.gather(
      *(judge(ctx[sid], jpeg) for sid, jpeg in collector.imgs.items())
  )
  return dict(
      arch=arch,
      q=q,
      ttft=round(collector.ttft or 0, 1),
      total=round(total, 1),
      sources=len(collector.final.get("sources", [])),
      slots=len(collector.slots),
      images=len(collector.imgs),
      yes=verdicts.count("yes"),
      partial=verdicts.count("partial"),
      no=verdicts.count("no"),
      chars=len(collector.md),
  )


async def main():
  parser = argparse.ArgumentParser(
      description="End-to-end architecture benchmark."
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "bench_arch.json")

  archs = ["gemini"] + (
      ["aimode", "thirdparty"] if settings.serpapi_api_key else []
  ) + (
      ["vertexsearch"] if settings.vertex_search_engines else []
  )
  rows = []
  for arch in archs:  # Sequential: "thirdparty" flips global provider settings.
    tp = arch == "thirdparty"
    vs = arch == "vertexsearch"
    object.__setattr__(settings, "web_provider", "serpapi" if tp else "gemini")
    object.__setattr__(
        settings, "image_provider", "serpapi" if tp else ("vertex_search" if vs else "gemini")
    )
    rows += await asyncio.gather(*(run(arch, q) for q in QUERIES))
  with open(out_path, "w") as f:
    json.dump(rows, f, indent=1)
  for arch in archs:
    rs = [r for r in rows if r["arch"] == arch and "error" not in r]
    errs = len([r for r in rows if r["arch"] == arch and "error" in r])
    n_img = sum(r["images"] for r in rs)
    print(
        f"{arch:10s} runs={len(rs)} errors={errs}"
        f" ttft_p50={statistics.median(r['ttft'] for r in rs):.1f}s"
        f" total_p50={statistics.median(r['total'] for r in rs):.1f}s"
        f" sources/q={statistics.mean(r['sources'] for r in rs):.1f}"
        f" images/q={n_img / max(len(rs), 1):.1f}"
        f" relevant(yes)={sum(r['yes'] for r in rs)}/{n_img}"
        f" partial={sum(r['partial'] for r in rs)}"
        f" chars/q={statistics.mean(r['chars'] for r in rs):.0f}"
    )


if __name__ == "__main__":
  asyncio.run(main())
