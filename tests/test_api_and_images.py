"""Tests for the HTTP API and the image pipeline, with Gemini/network mocked."""

import asyncio
import base64
import io
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from app import images, main, pipeline


import contextlib


@contextlib.contextmanager
def _setting(name, value):
  """Temporarily overrides a field of the frozen Settings dataclass."""
  from app.config import settings
  old = getattr(settings, name)
  object.__setattr__(settings, name, value)
  try:
    yield
  finally:
    object.__setattr__(settings, name, old)


def _png(w=640, h=480, color=(200, 30, 30)) -> bytes:
  buf = io.BytesIO()
  Image.new("RGB", (w, h), color).save(buf, format="PNG")
  return buf.getvalue()


def _parse_sse(body: str):
  events = []
  for block in body.split("\n\n"):
    ev, data = None, None
    for line in block.splitlines():
      if line.startswith("event:"):
        ev = line[6:].strip()
      elif line.startswith("data:"):
        data = json.loads(line[5:].strip())
    if ev:
      events.append((ev, data))
  return events


class ApiTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_index_and_static(self):
    r = self.client.get("/")
    self.assertEqual(r.status_code, 200)
    self.assertIn("Prism", r.text)
    self.assertEqual(self.client.get("/static/app.js").status_code, 200)
    self.assertEqual(self.client.get("/static/vendor/marked.umd.js").status_code, 200)
    self.assertEqual(self.client.get("/health").json(), {"ok": True})

  def test_search_streams_events(self):
    async def fake_run(req, emit):
      await emit("text", {"delta": "Hello"})
      await emit("slot", {"id": "s1", "query": "q", "caption": "c"})
      await emit("done", {"elapsed_ms": 1})

    with mock.patch.object(pipeline, "run_search", fake_run):
      r = self.client.post("/api/search", json={"query": "hi"})
    self.assertEqual(r.status_code, 200)
    self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
    evs = _parse_sse(r.text)
    self.assertEqual([e for e, _ in evs], ["text", "slot", "cost", "done"])  # cost precedes done
    self.assertEqual(evs[0][1], {"delta": "Hello"})

  def test_visual_mode_dispatch(self):
    calls = []

    async def fake_visual(req, emit):
      calls.append(req.mode)
      await emit("done", {})

    with _setting("allow_image_generation", True), \
         mock.patch.object(pipeline, "run_visual", fake_visual):
      self.client.post("/api/search", json={"query": "hi", "mode": "visual"})
    self.assertEqual(calls, ["visual"])

  def test_visual_mode_rejected_when_generation_disabled(self):
    with _setting("allow_image_generation", False):
      r = self.client.post("/api/search", json={"query": "hi", "mode": "visual"})
      self.assertEqual(r.status_code, 400)
      self.assertFalse(self.client.get("/api/config").json()["image_generation"])

  def test_pipeline_crash_becomes_error_event(self):
    async def boom(req, emit):
      raise RuntimeError("kaboom")

    with mock.patch.object(pipeline, "run_search", boom):
      r = self.client.post("/api/search", json={"query": "hi"})
    evs = _parse_sse(r.text)
    self.assertEqual(evs[0], ("error", {"message": "kaboom"}))

  def test_validation(self):
    self.assertEqual(self.client.post("/api/search", json={"query": ""}).status_code, 422)
    self.assertEqual(
        self.client.post("/api/search", json={"query": "x", "mode": "bogus"}).status_code, 422)
    self.assertEqual(
        self.client.post("/api/search", json={"query": "x" * 5000}).status_code, 400)
    r = self.client.post("/api/search", json={"query": "x", "image": "data:image/png;base64,AAAA"})
    self.assertEqual(r.status_code, 400)
    r = self.client.post("/api/search", json={"query": "x", "image": "http://evil/x.png"})
    self.assertEqual(r.status_code, 400)

  def test_rate_limit(self):
    async def fake_run(req, emit):
      await emit("done", {})

    orig = main.settings.rate_limit_per_min
    object.__setattr__(main.settings, "rate_limit_per_min", 2)  # Frozen dataclass.
    try:
      with mock.patch.object(pipeline, "run_search", fake_run):
        codes = [self.client.post("/api/search", json={"query": "x"}).status_code
                 for _ in range(3)]
    finally:
      object.__setattr__(main.settings, "rate_limit_per_min", orig)
    self.assertEqual(codes, [200, 200, 429])


class BuildContentsTest(unittest.TestCase):

  def test_history_and_image(self):
    data_url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    req = pipeline.SearchRequest(
        query="and this?",
        image=data_url,
        history=[
            pipeline.Turn(role="user", text="first"),
            pipeline.Turn(role="model", text="answer @@SLOT:s1@@ more@@CITE:1@@"),
        ],
    )
    contents = pipeline.build_contents(req)
    self.assertEqual([c.role for c in contents], ["user", "model", "user"])
    self.assertEqual(contents[1].parts[0].text, "answer  more")
    self.assertEqual(contents[2].parts[0].inline_data.mime_type, "image/jpeg")
    self.assertEqual(contents[2].parts[1].text, "and this?")


class NormalizeImageTest(unittest.TestCase):

  def test_resizes_and_converts(self):
    jpeg, w, h = images.normalize_image(_png(3000, 2000), 1024)
    self.assertEqual((w, h), (1024, 683))
    self.assertEqual(jpeg[:2], b"\xff\xd8")

  def test_rejects_small_and_extreme(self):
    with self.assertRaises(ValueError):
      images.normalize_image(_png(100, 100), 1024)
    with self.assertRaises(ValueError):
      images.normalize_image(_png(2000, 300), 1024)
    with self.assertRaises(ValueError):
      images.normalize_image(b"not an image", 1024)

  def test_rgba_flattened(self):
    buf = io.BytesIO()
    Image.new("RGBA", (400, 300), (0, 0, 0, 0)).save(buf, format="PNG")
    jpeg, _, _ = images.normalize_image(buf.getvalue(), 1024)
    px = Image.open(io.BytesIO(jpeg)).getpixel((10, 10))
    self.assertGreater(sum(px), 700)  # Transparent -> white background.


class ResolveSlotTest(unittest.IsolatedAsyncioTestCase):

  def _cand(self, url, kind="web"):
    return images.Candidate(image_url=url, source_url=url + "/page", domain="ex.com", kind=kind)

  async def test_web_then_dedupe(self):
    web = [self._cand("http://a"), self._cand("http://b")]

    async def fake_download(c):
      return (b"\xff\xd8jpeg", 800, 600)

    with mock.patch.object(images, "google_image_search", mock.AsyncMock(return_value=web)), \
         mock.patch.object(images, "download", fake_download), \
         mock.patch.object(images, "normalize_image", lambda b, m: (b, 100, 100)), \
         mock.patch.object(images, "rerank", mock.AsyncMock(return_value=0)):
      used = set()
      first = await images.resolve_slot("q", "c", "ctx", "web", used)
      second = await images.resolve_slot("q", "c", "ctx", "web", used)
    self.assertEqual(first.source_url, "http://a/page")
    self.assertEqual(second.source_url, "http://b/page")  # No duplicate image.
    self.assertTrue(first.data_url.startswith("data:image/jpeg;base64,"))

  async def test_fallback_to_wikimedia_then_generation(self):
    gen = images.ResolvedImage(data_url="data:x", width=1, height=1, kind="generated")
    with _setting("allow_image_generation", True), mock.patch.object(images, "google_image_search", mock.AsyncMock(return_value=[])), \
         mock.patch.object(images, "wikimedia_search", mock.AsyncMock(return_value=[])), \
         mock.patch.object(images, "generate_image", mock.AsyncMock(return_value=gen)) as g:
      self.assertIs(await images.resolve_slot("q", "c", "ctx", "auto", set()), gen)
      self.assertIsNone(await images.resolve_slot("q", "c", "ctx", "web", set()))
      self.assertEqual(g.await_count, 1)  # "web" never generates.

  async def test_rerank_rejection_falls_through(self):
    web = [self._cand("http://a")]
    wiki = [self._cand("http://w", kind="wikimedia")]

    async def fake_download(c):
      return (b"jpeg", 800, 600)

    with mock.patch.object(images, "google_image_search", mock.AsyncMock(return_value=web)), \
         mock.patch.object(images, "wikimedia_search", mock.AsyncMock(return_value=wiki)), \
         mock.patch.object(images, "download", fake_download), \
         mock.patch.object(images, "normalize_image", lambda b, m: (b, 100, 100)), \
         mock.patch.object(images, "rerank", mock.AsyncMock(side_effect=[None, 0])):
      img = await images.resolve_slot("q", "c", "ctx", "web", set())
    self.assertEqual(img.kind, "wikimedia")

  async def test_ai_preference_generates(self):
    gen = images.ResolvedImage(data_url="data:x", width=1, height=1, kind="generated")
    with _setting("allow_image_generation", True), mock.patch.object(images, "generate_image", mock.AsyncMock(return_value=gen)), \
         mock.patch.object(images, "google_image_search", mock.AsyncMock()) as s:
      self.assertIs(await images.resolve_slot("q", "c", "ctx", "ai", set()), gen)
      s.assert_not_awaited()


class NoGenerationTest(unittest.IsolatedAsyncioTestCase):
  """With ALLOW_IMAGE_GENERATION=false (the default) nothing is ever generated."""

  async def test_default_is_disabled(self):
    from app.config import Settings
    self.assertFalse(Settings().allow_image_generation)

  async def test_preferences_never_generate(self):
    fake_client = mock.MagicMock()
    with _setting("allow_image_generation", False), \
         mock.patch.object(images, "client", return_value=fake_client), \
         mock.patch.object(images, "google_image_search", mock.AsyncMock(return_value=[])), \
         mock.patch.object(images, "wikimedia_search", mock.AsyncMock(return_value=[])):
      for pref in ("auto", "ai", "web"):
        self.assertIsNone(await images.resolve_slot("q", "c", "ctx", pref, set()))
      self.assertIsNone(await images.generate_image("c", "ctx"))
    fake_client.aio.models.generate_content.assert_not_called()

  async def test_image_search_call_is_capped(self):
    resp = mock.MagicMock()
    resp.candidates = []
    gc = mock.AsyncMock(return_value=resp)
    fake_client = mock.MagicMock()
    fake_client.aio.models.generate_content = gc
    with mock.patch.object(images, "client", return_value=fake_client):
      await images.google_image_search("q", "ctx")
    cfg = gc.await_args.kwargs["config"]
    self.assertEqual(cfg.max_output_tokens, images.settings.image_search_max_tokens)
    self.assertLess(cfg.max_output_tokens, 1000)  # An image needs >1,000 tokens.


if __name__ == "__main__":
  unittest.main()
