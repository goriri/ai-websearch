"""Image-search grounding support matrix across Gemini models on Vertex AI.

Probes 16 text models and 5 image models across 4 tiers:
  T0 basic      : plain text call (is the model servable here?)
  T1 image-only : google_search(search_types=[image_search]) with a "what does X look like" prompt
  T2 both/plain : search_types=[web_search, image_search], no system prompt (N runs)
  T3 both/steps : same + step-by-step system instruction (N runs)
Image models are tried with TEXT-only first, then TEXT+IMAGE (recording which works).

Backs:
  - docs/image-search-grounding.md (Section 1: Model Support Matrix)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").

Approximate cost:
  - Probes 21 models. Text models fail fast on T1 (1 small prompt). Image models
    execute ~3 queries each for N repetitions (default N=3).
  - Total: ~50-80 Gemini API calls.

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/imgsearch_matrix.py 3 --out-dir /tmp
"""
import argparse
import asyncio
import json
import os
import pathlib
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

TEXT_MODELS = [
    "gemini-1.5-pro-002",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro",
    "gemini-2.5-pro-exp-03-25",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-4-argon",
    "gemini-omni-1.1-flash-preview",
]
IMAGE_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image",
    "gemini-3.1-flash-lite-image",
    "gemini-nano-banana-2.1",
]

STEP_SYS = (
    "Step 1: run Google WEB search queries to collect facts and sources. Step"
    " 2: run Google IMAGE search queries to find real photos of the things you"
    " discuss. You must do both steps before answering. Then answer in"
    " Markdown. Text only."
)
Q_IMG = (
    "What does the Sagrada Familia look like? Look it up with Google image"
    " search and describe what the real photos show."
)
Q_BOTH = [
    "How tall is Merdeka 118 and what does it look like?",
    "Best beginner hikes in Zion National Park",
    "What are the most iconic buildings designed by Zaha Hadid?",
]

locations = tuple(dict.fromkeys([LOCATION, "global", "us-central1"]))
clients = {
    loc: genai.Client(vertexai=True, project=PROJECT, location=loc)
    for loc in locations
}
sem = asyncio.Semaphore(10)


def tool(web, img):
  st = types.SearchTypes(
      web_search=types.WebSearch() if web else None,
      image_search=types.ImageSearch() if img else None,
  )
  return [types.Tool(google_search=types.GoogleSearch(search_types=st))]


def summarize(r):
  c = r.candidates[0] if r.candidates else None
  gm = c.grounding_metadata if c else None
  chunks = (gm.grounding_chunks or []) if gm else []
  img_idx = {i for i, ch in enumerate(chunks) if ch.image}
  sups = (gm.grounding_supports or []) if gm else []
  return dict(
      web_q=len((gm.web_search_queries or []) if gm else []),
      img_q=len((gm.image_search_queries or []) if gm else []),
      web_chunks=sum(1 for ch in chunks if ch.web),
      img_chunks=len(img_idx),
      supports=len(sups),
      supports_on_images=sum(
          1 for s in sups if set(s.grounding_chunk_indices or []) & img_idx
      ),
      gen_images=sum(
          1
          for p in ((c.content.parts if c and c.content else None) or [])
          if p.inline_data
      ),
  )


async def call(model, loc, contents, cfg):
  async with sem:
    t = time.time()
    r = await clients[loc].aio.models.generate_content(
        model=model, contents=contents, config=cfg
    )
    return r, time.time() - t


def err(e):
  s = str(e)
  for key in (
      "image_search is not supported",
      "response_modalities",
      "not found",
      "NOT_FOUND",
      "not supported",
      "INVALID_ARGUMENT",
      "PERMISSION_DENIED",
      "RESOURCE_EXHAUSTED",
  ):
    if key in s:
      i = s.find("'message': '")
      return s[i + 12 : i + 140].split("'")[0] if i >= 0 else key
  return s[:120]


async def _run_search_types(model, loc, mods, sysmsg, q):
  try:
    r, dt = await call(
        model,
        loc,
        q,
        types.GenerateContentConfig(
            system_instruction=sysmsg,
            response_modalities=mods,
            tools=tool(True, True),
        ),
    )
    s = summarize(r)
    s["mode"] = (
        "both"
        if (s["web_q"] or s["web_chunks"])
        and (s["img_q"] or s["img_chunks"])
        else "web"
        if (s["web_q"] or s["web_chunks"])
        else "image"
        if (s["img_q"] or s["img_chunks"])
        else "none"
    )
    s["latency"] = round(dt, 1)
    return s
  except Exception as e:
    return {"mode": "error", "err": err(e)}


async def probe(model, is_image, reps):
  out = {"model": model, "kind": "image" if is_image else "text"}
  # T0: availability (+ pick location + modalities)
  loc = None
  mods_options = [["TEXT"], ["TEXT", "IMAGE"]] if is_image else [None]
  for l in ("global", "us-central1"):
    if l not in clients:
      continue
    try:
      await call(
          model,
          l,
          "Say hi in two words.",
          types.GenerateContentConfig(
              response_modalities=mods_options[-1] if is_image else None
          ),
      )
      loc = l
      break
    except Exception as e:
      out["t0_err"] = err(e)
  out["location"] = loc
  if not loc:
    return out
  out.pop("t0_err", None)

  # T1: image_search only (find which modalities are accepted)
  mods = None
  for m in mods_options:
    try:
      r, dt = await call(
          model,
          loc,
          Q_IMG,
          types.GenerateContentConfig(
              response_modalities=m, tools=tool(False, True)
          ),
      )
      out["t1"] = summarize(r) | {"latency": round(dt, 1)}
      mods = m
      break
    except Exception as e:
      out["t1_err" + ("" if m is None else "_" + "+".join(m))] = err(e)
  out["modalities_ok"] = (
      "+".join(mods) if mods else ("TEXT" if not is_image else None)
  )
  if "t1" not in out:
    # Also confirm web-only grounding works on this model (baseline).
    try:
      r, _ = await call(
          model,
          loc,
          Q_BOTH[0],
          types.GenerateContentConfig(tools=tool(True, False)),
      )
      out["web_only"] = summarize(r)
    except Exception as e:
      out["web_only_err"] = err(e)
    return out

  # T2/T3: both search types, plain vs step prompt
  qs = [Q_BOTH[i % len(Q_BOTH)] for i in range(reps)]
  out["t2"] = await asyncio.gather(
      *(_run_search_types(model, loc, mods, None, q) for q in qs)
  )
  out["t3"] = await asyncio.gather(
      *(_run_search_types(model, loc, mods, STEP_SYS, q) for q in qs)
  )
  return out


async def main():
  parser = argparse.ArgumentParser(
      description="Probe image-search grounding support matrix across models."
  )
  parser.add_argument(
      "reps",
      nargs="?",
      type=int,
      default=3,
      help="Number of repetitions per query for T2/T3 tests (default: 3)",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "imgsearch_matrix.json")

  res = await asyncio.gather(
      *([probe(m, False, args.reps) for m in TEXT_MODELS]
        + [probe(m, True, args.reps) for m in IMAGE_MODELS])
  )
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1, default=str)
  for r in res:
    print(json.dumps(r, default=str)[:600])


if __name__ == "__main__":
  asyncio.run(main())
