"""Runs a search through the pipeline from the command line and prints events.

Usage: python -m scripts.cli_search "your query" [--mode visual] [--images web]
"""

import argparse
import asyncio
import json
import time

from app import pipeline


async def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("query")
  ap.add_argument("--mode", default="search")
  ap.add_argument("--model", default="fast")
  ap.add_argument("--images", default="auto")
  args = ap.parse_args()
  t0 = time.monotonic()
  req = pipeline.SearchRequest(query=args.query, mode=args.mode,
                               model=args.model, images=args.images)

  async def emit(event, data):
    ts = f"[{time.monotonic() - t0:5.1f}s]"
    if event == "text":
      print(data["delta"], end="", flush=True)
    elif event == "image":
      print(f"\n{ts} IMAGE {data['id']} kind={data['kind']} {data['width']}x"
            f"{data['height']} {data['domain']} | {data['title'][:60]} | "
            f"{len(data['src']) // 1024}KB")
    elif event == "final":
      print(f"\n{ts} FINAL model={data['model']} sources={len(data['sources'])}"
            f" queries={data['queries']}")
      print(data["markdown"][:3000])
      for s in data["sources"]:
        print("   ", s["n"], s["domain"], s["url"][:90])
    else:
      print(f"\n{ts} {event.upper()} {json.dumps(data)[:300]}")

  runner = pipeline.run_visual if args.mode == "visual" else pipeline.run_search
  await runner(req, emit)


if __name__ == "__main__":
  asyncio.run(main())
