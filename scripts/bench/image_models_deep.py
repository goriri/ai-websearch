"""Deep-dive: image models for grounded text + image search in ONE call.

Compares gemini-nano-banana-2.1 vs gemini-3.1-flash-image on:
  - Both searches executed (web + image grounding)
  - Unwanted generated images (inline_data parts)
  - Grounding supports pointing to image chunks (text -> image attribution)
  - Attribution correctness judged by gemini-3.8-flash
  - Streaming vs non-streaming behavior and latency

Backs:
  - docs/image-search-grounding.md (Section 6.2: Streaming Incompatibility and
    Section 6.3: Alternative Image Models)

Required environment:
  - GOOGLE_CLOUD_PROJECT: Google Cloud project ID with Vertex AI enabled.
  - GOOGLE_CLOUD_LOCATION: Vertex AI region (default: "global").

Approximate cost:
  - 2 models × 6 queries × 2 modes (streamed/non-streamed) = 24 generation calls
  - ~20-30 judge calls (gemini-3.8-flash)
  - Total: ~50 Gemini API calls.

Example:
  export GOOGLE_CLOUD_PROJECT="my-project"
  python scripts/bench/image_models_deep.py gemini-3.1-flash-image --out-dir /tmp
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
from app import images

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

DEFAULT_MODELS = ["gemini-nano-banana-2.1", "gemini-3.1-flash-image"]
QS = [
    "How tall is Merdeka 118 and what does it look like?",
    "Best beginner hikes in Zion National Park",
    "What are the most iconic buildings designed by Zaha Hadid?",
    "Birds you can spot in Central Park in spring",
    "Best street foods to try in Bangkok",
    "Japandi living room design ideas",
]
SYS = {
    "steps": (
        "Step 1: run Google WEB search queries to collect facts and sources. "
        "Step 2: run Google IMAGE search queries to find real photos of the"
        " things you discuss. You must do both steps before answering. Then"
        " answer in Markdown with ### sections. Respond with text only; do not"
        " generate or edit any images."
    ),
}


def clean(s):
  return (
      re.sub("<[^>]+>", "", s or "")
      .replace("&#39;", "'")
      .replace("&amp;", "&")
  )


async def judge(caption, url):
  c = images.Candidate(image_url=url)
  got = await images.download(c)
  if not got:
    return "download_fail"
  r = await client.aio.models.generate_content(
      model="gemini-3.8-flash",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=(
                          f"Text: {caption}\nIs this photo a correct"
                          " illustration of the main subject of the text? One"
                          " word: yes, partial, or no."
                      )
                  ),
                  types.Part.from_bytes(data=got[0], mime_type="image/jpeg"),
              ],
          )
      ],
  )
  w = (r.text or "").strip().lower()
  return (
      "yes"
      if w.startswith("yes")
      else "partial"
      if w.startswith("partial")
      else "no"
  )


async def one(model, q, sysname, stream):
  cfg = types.GenerateContentConfig(
      system_instruction=SYS[sysname],
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
  )
  t0 = time.time()
  ttft = None
  text = ""
  gen = 0
  gm = None
  try:
    if stream:
      async for ch in await client.aio.models.generate_content_stream(
          model=model, contents=q, config=cfg
      ):
        for c in ch.candidates or []:
          if c.grounding_metadata:
            gm = c.grounding_metadata
          for p in (c.content.parts if c.content else None) or []:
            if p.thought:
              continue
            if p.text:
              ttft = ttft or time.time() - t0
              text += p.text
            if p.inline_data:
              gen += 1
    else:
      r = await client.aio.models.generate_content(
          model=model, contents=q, config=cfg
      )
      c = r.candidates[0]
      gm = c.grounding_metadata
      for p in c.content.parts or []:
        if p.thought:
          continue
        if p.text:
          text += p.text
        if p.inline_data:
          gen += 1
      ttft = time.time() - t0
  except Exception as e:
    return {"model": model, "q": q, "stream": stream, "error": str(e)[:160]}
  chunks = (gm.grounding_chunks or []) if gm else []
  img = {i: ch.image for i, ch in enumerate(chunks) if ch.image}
  sups = (gm.grounding_supports or []) if gm else []
  img_sups = [
      (
          s.segment.text if s.segment else "",
          [i for i in (s.grounding_chunk_indices or []) if i in img],
      )
      for s in sups
  ]
  img_sups = [(t, ix) for t, ix in img_sups if ix]
  verdicts = await asyncio.gather(
      *(judge(t, img[ix[0]].image_uri) for t, ix in img_sups[:6])
  )
  return {
      "model": model,
      "q": q[:40],
      "stream": stream,
      "ttft": round(ttft or 0, 1),
      "total": round(time.time() - t0, 1),
      "web_q": len(gm.web_search_queries or []) if gm else 0,
      "img_q": len(gm.image_search_queries or []) if gm else 0,
      "web_chunks": sum(1 for ch in chunks if ch.web),
      "img_chunks": len(img),
      "supports": len(sups),
      "img_supports": len(img_sups),
      "generated_images": gen,
      "attrib_judged": verdicts,
      "examples": [
          (t[:70], clean(img[ix[0]].title)[:50]) for t, ix in img_sups[:3]
      ],
  }


async def main():
  parser = argparse.ArgumentParser(
      description="Deep dive on image models for grounded search."
  )
  parser.add_argument(
      "models",
      nargs="*",
      default=[],
      help="Image models to test (default: gemini-nano-banana-2.1 gemini-3.1-flash-image)",
  )
  parser.add_argument(
      "--out-dir",
      default=os.environ.get("BENCH_OUT_DIR", "/tmp"),
      help="Directory for output files (default: /tmp or $BENCH_OUT_DIR)",
  )
  args = parser.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  out_path = os.path.join(args.out_dir, "image_models_deep.json")

  models = args.models or DEFAULT_MODELS
  jobs = [one(m, q, "steps", s) for m in models for q in QS for s in (False, True)]
  res = await asyncio.gather(*jobs)
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1)
  for r in res:
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
  asyncio.run(main())
