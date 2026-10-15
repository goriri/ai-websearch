"""Live end-to-end check: verify zero image output tokens in production pipeline.

Runs real searches with diverse image preferences ("auto", "ai", "web") and
verifies that with default settings:
  1. No Gemini call produces billable image output tokens.
  2. Every image displayed is an authentic web/Wikimedia photo.
  3. Image generation fallback remains disabled.

Backs:
  - docs/image-search-grounding.md (Section 5: Measured Evidence — Slot Fill
    Accuracy and Zero Image Output Tokens)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").

Approximate cost:
  - 4 production search queries: ~40 Gemini calls total across text answer,
    image search, and visual re-ranking.

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/live_no_generation_check.py --out-dir /tmp
"""
import argparse
import asyncio
import collections
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
  sys.exit("Error: GOOGLE_CLOUD_PROJECT environment variable is required.")

from app import pipeline
from app.config import settings
from app.genai_client import client

CALLS = []
m = client().aio.models
_gc, _gcs = m.generate_content, m.generate_content_stream


def record(model, usage):
  img = (
      sum(
          d.token_count or 0
          for d in (usage.candidates_tokens_details or [])
          if "IMAGE" in str(d.modality)
      )
      if usage
      else 0
  )
  CALLS.append(
      (model, (usage.candidates_token_count or 0) if usage else 0, img)
  )


async def gc(*a, **kw):
  r = await _gc(*a, **kw)
  record(kw.get("model"), r.usage_metadata)
  return r


async def _wrap_stream_with_recording(stream, model):
  last = None
  async for ch in stream:
    last = ch.usage_metadata or last
    yield ch
  record(model, last)


async def gcs(*a, **kw):
  stream = await _gcs(*a, **kw)
  return _wrap_stream_with_recording(stream, kw.get("model"))


m.generate_content, m.generate_content_stream = gc, gcs

QUERIES = [
    ("Best beginner hikes in Zion National Park", "auto"),
    ("Most iconic buildings designed by Zaha Hadid", "ai"),
    ("Birds you can spot in Central Park in spring", "auto"),
    ("Best street foods to try in Bangkok", "web"),
]


class LiveCheckCollector:
  """Collects slot and image metrics during live searches."""

  def __init__(self):
    self.slots = 0
    self.failed = 0
    self.kinds = collections.Counter()

  async def emit(self, ev, d):
    if ev == "slot":
      self.slots += 1
    elif ev == "image":
      self.kinds[d["kind"]] += 1
    elif ev == "image_failed":
      self.failed += 1


async def main():
  parser = argparse.ArgumentParser(
      description="Live verification of zero image output tokens."
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "live_no_generation_check.json")

  print(
      "allow_image_generation =",
      settings.allow_image_generation,
      "| image_search_max_tokens =",
      settings.image_search_max_tokens,
  )
  collector = LiveCheckCollector()
  for q, pref in QUERIES:
    await pipeline.run_search(
        pipeline.SearchRequest(query=q, images=pref), collector.emit
    )
  by_model = collections.defaultdict(lambda: [0, 0, 0])
  for model, out, img in CALLS:
    by_model[model][0] += 1
    by_model[model][1] += out
    by_model[model][2] += img
  print(
      f"slots={collector.slots} images_by_kind={dict(collector.kinds)}"
      f" unfilled={collector.failed}"
  )
  for model, (n, out, img) in by_model.items():
    print(
        f"  {model:26s} calls={n:3d} output_tokens={out:6d}"
        f" image_output_tokens={img}"
    )
  total_img_tokens = sum(c[2] for c in CALLS)
  print("TOTAL image output tokens:", total_img_tokens)

  result = {
      "slots": collector.slots,
      "images_by_kind": dict(collector.kinds),
      "unfilled": collector.failed,
      "calls_by_model": {
          model: {"calls": n, "output_tokens": out, "image_output_tokens": img}
          for model, (n, out, img) in by_model.items()
      },
      "total_image_output_tokens": total_img_tokens,
  }
  with open(out_path, "w") as f:
    json.dump(result, f, indent=1)


if __name__ == "__main__":
  asyncio.run(main())
