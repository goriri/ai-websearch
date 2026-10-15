"""Benchmark: verify that max_output_tokens cap prevents image generation.

Measures whether the image-search call to gemini-3.1-flash-image ever produces
(billable) image output tokens, and whether a max_output_tokens cap (e.g. 128
or 64 tokens) prevents image generation while still returning full grounding
chunks.

Backs:
  - docs/image-search-grounding.md (Section 2: The Exact Image Search Request,
    Section 5: Measured Evidence — Zero Image Output Tokens)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").

Approximate cost:
  - 3 variants × 10 queries × reps (default: 1) = 30 calls to gemini-3.1-flash-image.

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/image_search_no_generation.py 1 --out-dir /tmp
"""
import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys

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

QUERIES = [
    "Angels Landing chains section Zion",
    "Heydar Aliyev Center Baku exterior",
    "Blackburnian warbler Central Park",
    "Pad kra pao Bangkok street food",
    "Merdeka 118 tower Kuala Lumpur",
    "Japandi living room interior",
    "Sagrada Familia interior columns",
    "Riverside Walk trail Zion",
    "London Aquatics Centre roof",
    "Mango sticky rice Thailand",
]
CUR = (
    "What does this look like: {q}? Look it up with Google image search and"
    " describe in two sentences what the real photos you found show. Answer in"
    " text only."
)
SHORT = (
    "What does this look like: {q}? Look it up with Google image search, then"
    " reply with one short sentence. Text only."
)
VARIANTS = {
    "current": dict(prompt=CUR, max_tokens=None),
    "short+cap256": dict(prompt=SHORT, max_tokens=256),
    "short+cap64": dict(prompt=SHORT, max_tokens=64),
}


def modality_tokens(details):
  out = {}
  for d in details or []:
    out[str(d.modality).split(".")[-1]] = d.token_count
  return out


async def one(vname, q):
  v = VARIANTS[vname]
  try:
    r = await client.aio.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=v["prompt"].format(q=q),
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            max_output_tokens=v["max_tokens"],
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
  except Exception as e:
    return dict(v=vname, q=q, error=str(e)[:150])
  c = r.candidates[0]
  gm = c.grounding_metadata
  u = r.usage_metadata
  return dict(
      v=vname,
      q=q,
      finish=str(c.finish_reason).split(".")[-1],
      img_chunks=(
          sum(1 for ch in (gm.grounding_chunks or []) if ch.image) if gm else 0
      ),
      gen_parts=sum(
          1
          for p in (c.content.parts if c.content else None) or []
          if p.inline_data
      ),
      out_tokens=u.candidates_token_count,
      thoughts=u.thoughts_token_count,
      out_by_modality=modality_tokens(u.candidates_tokens_details),
      prompt_tokens=u.prompt_token_count,
      tool_tokens=u.tool_use_prompt_token_count,
  )


async def main():
  parser = argparse.ArgumentParser(
      description="Benchmark max_output_tokens capping on image search."
  )
  parser.add_argument(
      "reps",
      nargs="?",
      type=int,
      default=1,
      help="Repetitions per variant/query (default: 1)",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "image_search_no_generation.json")

  rows = await asyncio.gather(
      *(
          one(v, q)
          for v in VARIANTS
          for q in QUERIES
          for _ in range(args.reps)
      )
  )
  for v in VARIANTS:
    rs = [r for r in rows if r["v"] == v and "error" not in r]
    errs = [r for r in rows if r["v"] == v and "error" in r]
    img_tok = sum(r["out_by_modality"].get("IMAGE", 0) for r in rs)
    print(
        f"{v:14s} runs={len(rs)} errors={len(errs)} "
        f"with_results={sum(1 for r in rs if r['img_chunks'])}/{len(rs)} "
        f"avg_chunks={statistics.mean(r['img_chunks'] for r in rs):.1f} "
        f"generated_parts={sum(r['gen_parts'] for r in rs)}"
        f" image_out_tokens={img_tok} "
        f"avg_out_tokens={statistics.mean((r['out_tokens'] or 0) for r in rs):.0f} "
        f"avg_thoughts={statistics.mean((r['thoughts'] or 0) for r in rs):.0f} "
        f"finish={sorted({r['finish'] for r in rs})}"
    )
    for e in errs[:2]:
      print("   ERR", e["error"])
  with open(out_path, "w") as f:
    json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
  asyncio.run(main())
