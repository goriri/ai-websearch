"""FastAPI web server for Prism multimodal search."""

import asyncio
import collections
import json
import logging
import os
import time

import fastapi
from fastapi import responses, staticfiles

from app import cost, pipeline
from app.config import settings

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("prism")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = fastapi.FastAPI(title="Prism Multimodal Search", docs_url=None, redoc_url=None)
app.mount("/static", staticfiles.StaticFiles(directory=STATIC_DIR), name="static")

_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)


def _client_ip(request: fastapi.Request) -> str:
  fwd = request.headers.get("x-forwarded-for")
  if fwd:
    return fwd.split(",")[0].strip()
  return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
  limit = settings.rate_limit_per_min
  if limit <= 0:
    return False
  now = time.monotonic()
  q = _hits[ip]
  while q and now - q[0] > 60:
    q.popleft()
  if len(q) >= limit:
    return True
  q.append(now)
  return False


@app.get("/", include_in_schema=False)
async def index():
  return responses.FileResponse(
      os.path.join(STATIC_DIR, "index.html"),
      headers={"Cache-Control": "no-cache"},
  )


@app.get("/compare", include_in_schema=False)
async def compare():
  return responses.FileResponse(
      os.path.join(STATIC_DIR, "compare.html"),
      headers={"Cache-Control": "no-cache"},
  )


@app.get("/health")
async def healthz():
  return {"ok": True}


@app.get("/api/config")
async def config():
  archs = ["gemini"] + (["thirdparty", "aimode"] if settings.serpapi_api_key else [])
  return {
      "answer_model": settings.answer_model,
      "pro_model": settings.pro_model,
      "image_model": settings.image_model,
      "lite_model": settings.lite_model,
      "image_generation": settings.allow_image_generation,
      "ai_mode": bool(settings.serpapi_api_key),
      "archs": archs,
  }


def _sse(event: str, data: dict) -> bytes:
  return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@app.post("/api/search")
async def search(req: pipeline.SearchRequest, request: fastapi.Request):
  if len(req.query) > settings.max_query_chars:
    raise fastapi.HTTPException(400, "Query is too long.")
  if req.arch in ("thirdparty", "aimode") and not settings.serpapi_api_key:
    raise fastapi.HTTPException(400, "This architecture needs SERPAPI_API_KEY")
  if req.mode == "visual" and not req.arch and not settings.allow_image_generation:
    raise fastapi.HTTPException(
        400, "Visual mode is disabled (image generation is turned off).")
  if req.mode == "aimode" and not req.arch and not settings.serpapi_api_key:
    raise fastapi.HTTPException(400, "AI Mode needs SERPAPI_API_KEY")
  if _rate_limited(_client_ip(request)):
    raise fastapi.HTTPException(429, "Too many searches, please wait a minute.")
  try:
    pipeline.build_contents(req)  # Validates the optional attached image.
  except ValueError as e:
    raise fastapi.HTTPException(400, str(e)) from e

  queue: asyncio.Queue = asyncio.Queue()

  async def emit(event: str, data: dict) -> None:
    await queue.put(_sse(event, data))

  if req.arch == "aimode":
    runner = pipeline.run_ai_mode
  elif req.arch in ("gemini", "thirdparty"):
    runner = pipeline.run_search
  else:
    runners = {"visual": pipeline.run_visual, "aimode": pipeline.run_ai_mode}
    runner = runners.get(req.mode, pipeline.run_search)
  log.info("search mode=%s arch=%s model=%s images=%s q=%r", req.mode, req.arch,
           req.model, req.images, req.query[:200])

  async def run():
    meter = cost.start()
    cost_emitted = False

    async def wrapped_emit(event: str, data: dict) -> None:
      nonlocal cost_emitted
      if event == "done" and not cost_emitted:
        cost_emitted = True
        await emit("cost", meter.summary())
      await emit(event, data)

    try:
      await runner(req, wrapped_emit)
    except Exception as e:  # pylint: disable=broad-except
      log.exception("pipeline crashed")
      await wrapped_emit("error", {"message": str(e)})
      await wrapped_emit("done", {})
    finally:
      await queue.put(None)

  async def stream():
    task = asyncio.create_task(run())
    try:
      yield b": stream-open\n\n"
      while True:
        try:
          item = await asyncio.wait_for(queue.get(), timeout=15)
        except asyncio.TimeoutError:
          yield b": keep-alive\n\n"
          continue
        if item is None:
          break
        yield item
    finally:
      if not task.done():
        task.cancel()

  return responses.StreamingResponse(
      stream(),
      media_type="text/event-stream",
      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
  )
