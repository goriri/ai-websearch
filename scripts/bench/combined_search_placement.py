"""Experiment: unified web + image search in one call and section placement.

Evaluates an experimental pattern where a single call to gemini-3.1-flash-image
is instructed to execute both Google web search and Google image search, write
an answer divided into sections, and insert {{image_N}} placement markers.

Measures:
  1. How often both web and image searches actually execute.
  2. Number of web and image chunks returned in grounding metadata.
  3. Whether {{image_N}} markers land under the matching section heading
     (demonstrates the chunk mapping disconnect described in docs).

Backs:
  - docs/image-search-grounding.md (Section 6.1: Chunk Mapping Disconnect)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").

Approximate cost:
  - 4 queries × trials (default: 2) = 8 Gemini calls to gemini-3.1-flash-image.

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/combined_search_placement.py --trials 2 --out-dir /tmp
"""
import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
  sys.exit("Error: GOOGLE_CLOUD_PROJECT environment variable is required.")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

SYS = """You are a multimodal search engine. You MUST use BOTH tools:
1) Google web search to research facts, and
2) Google image search to find real photos for each item you discuss.
Write a Markdown answer with one `###` section per item. At the end of each
section, on its own line, place the retrieved photo of THAT item as {{image_N}}
(the retrieved image's name). 3-5 images. Never generate images."""

QS = [
    "best beginner hikes in Zion National Park",
    "most iconic buildings designed by Zaha Hadid",
    "birds you can spot in Central Park in spring",
    "best street foods to try in Bangkok",
]


def clean(s):
  return (
      re.sub("<[^>]+>", "", s or "")
      .replace("&#39;", "'")
      .replace("&amp;", "&")
  )


async def run(q, trial):
  t = time.time()
  try:
    r = await client.aio.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=q,
        config=types.GenerateContentConfig(
            system_instruction=SYS,
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
  except Exception as e:
    return {"q": q, "error": str(e)[:200]}
  c = r.candidates[0]
  text = "".join(p.text or "" for p in (c.content.parts or []) if not p.thought)
  gm = c.grounding_metadata
  chunks = (gm.grounding_chunks if gm else None) or []
  img = {i: ch.image for i, ch in enumerate(chunks) if ch.image}
  web = [i for i, ch in enumerate(chunks) if ch.web]
  img_list = list(img.values())
  # Evaluate placement: for each marker, find the heading above it and compare
  # with the title of (a) chunk with same overall index, (b) Nth image chunk.
  placements = []
  for m in re.finditer(r"\{\{\s*image_(\d+)(?:\.\w+)?\s*\}\}", text):
    n = int(m.group(1))
    heads = re.findall(r"^#{2,4}\s*(.+)$", text[: m.start()], re.M)
    head = heads[-1].strip() if heads else "(none)"
    by_chunk = (
        clean(chunks[n].image.title)
        if n < len(chunks) and chunks[n].image
        else None
    )
    by_nth = clean(img_list[n].title) if n < len(img_list) else None
    placements.append({
        "n": n,
        "section": head[:60],
        "chunk[n]": (by_chunk or "")[:60],
        "nth_image": (by_nth or "")[:60],
    })
  return {
      "q": q,
      "trial": trial,
      "latency": round(time.time() - t, 1),
      "web_q": len((gm.web_search_queries or []) if gm else []),
      "img_q": len((gm.image_search_queries or []) if gm else []),
      "web_chunks": len(web),
      "image_chunks": len(img),
      "supports": len((gm.grounding_supports or []) if gm else []),
      "placements": placements,
  }


async def main():
  parser = argparse.ArgumentParser(
      description="Experiment: combined web+image search and placement accuracy."
  )
  parser.add_argument(
      "--trials",
      type=int,
      default=2,
      help="Trials per query (default: 2)",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "combined_search_placement.json")

  res = await asyncio.gather(*(run(q, t) for q in QS for t in range(args.trials)))
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1, default=str)
  for r in res:
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
  asyncio.run(main())
