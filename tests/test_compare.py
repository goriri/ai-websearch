"""Tests for 3-architecture comparison portal and per-request arch overrides."""

import asyncio
import contextlib
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from google.genai import types

from app import images, main, pipeline, providers
from app.config import settings


@contextlib.contextmanager
def _setting(**kw):
  old = {k: getattr(settings, k) for k in kw}
  for k, v in kw.items():
    object.__setattr__(settings, k, v)
  try:
    yield
  finally:
    for k, v in old.items():
      object.__setattr__(settings, k, v)


class _Part:
  def __init__(self, text):
    self.text = text
    self.thought = False


class _Chunk:
  def __init__(self, text):
    cand = mock.MagicMock()
    cand.grounding_metadata = None
    cand.content.parts = [_Part(text)]
    self.candidates = [cand]


class SearchRequestArchTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_search_request_accepts_valid_archs(self):
    for val in ("gemini", "thirdparty", "aimode", None):
      req = pipeline.SearchRequest(query="q", arch=val)
      self.assertEqual(req.arch, val)

  def test_search_request_rejects_invalid_arch_422(self):
    res = self.client.post("/api/search", json={"query": "q", "arch": "invalid_arch"})
    self.assertEqual(res.status_code, 422)


class CompareRouteAndConfigTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_compare_html_route_200(self):
    res = self.client.get("/compare")
    self.assertEqual(res.status_code, 200)
    self.assertIn("Compare architectures", res.text)
    self.assertEqual(res.headers.get("cache-control"), "no-cache")

  def test_api_config_archs(self):
    with _setting(serpapi_api_key=""):
      res = self.client.get("/api/config")
      self.assertEqual(res.status_code, 200)
      self.assertEqual(res.json()["archs"], ["gemini"])

    with _setting(serpapi_api_key="mock_key"):
      res = self.client.get("/api/config")
      self.assertEqual(res.status_code, 200)
      self.assertEqual(res.json()["archs"], ["gemini", "thirdparty", "aimode"])


class ArchDispatchTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_thirdparty_and_aimode_without_key_return_400(self):
    with _setting(serpapi_api_key=""):
      r1 = self.client.post("/api/search", json={"query": "q", "arch": "thirdparty"})
      self.assertEqual(r1.status_code, 400)
      self.assertIn("This architecture needs SERPAPI_API_KEY", r1.text)

      r2 = self.client.post("/api/search", json={"query": "q", "arch": "aimode"})
      self.assertEqual(r2.status_code, 400)
      self.assertIn("This architecture needs SERPAPI_API_KEY", r2.text)

  def test_aimode_with_key_dispatches_to_run_ai_mode(self):
    mock_run = mock.AsyncMock()
    with _setting(serpapi_api_key="key"), mock.patch.object(pipeline, "run_ai_mode", mock_run):
      res = self.client.post("/api/search", json={"query": "q", "arch": "aimode"})
      self.assertEqual(res.status_code, 200)
      mock_run.assert_called_once()
      req_arg = mock_run.call_args[0][0]
      self.assertEqual(req_arg.arch, "aimode")


class RunSearchArchOverrideTest(unittest.IsolatedAsyncioTestCase):

  async def test_arch_thirdparty_calls_serpapi_and_passes_provider_to_resolve_slot(self):
    tp_results = [{"title": "Title", "url": "https://a.com/page", "domain": "a.com", "snippet": "snippet"}]

    async def fake_stream(**kw):
      async def gen():
        yield _Chunk("Answer text with a slot:\n\n[[IMAGE: mountain query | scenic mountain]]\n\nMore text.")
      return gen()

    fake_client = mock.MagicMock()
    fake_client.aio.models.generate_content_stream = fake_stream

    mock_serpapi_web = mock.AsyncMock(return_value=tp_results)
    mock_resolve_slot = mock.AsyncMock(return_value=None)
    events = []

    mock_emit = mock.AsyncMock(side_effect=lambda ev, d: events.append((ev, d)))

    req = pipeline.SearchRequest(query="find mountain", arch="thirdparty")
    image_tasks = []

    with _setting(serpapi_api_key="k", web_provider="gemini", image_provider="gemini"), \
         mock.patch.dict(providers.WEB_PROVIDERS, {"serpapi": mock_serpapi_web}), \
         mock.patch.object(pipeline, "client", return_value=fake_client), \
         mock.patch.object(images, "resolve_slot", mock_resolve_slot), \
         mock.patch.object(pipeline, "related_questions", mock.AsyncMock(return_value=[])):
      await pipeline._run_search(req, mock_emit, image_tasks)
      if image_tasks:
        await asyncio.gather(*image_tasks, return_exceptions=True)

    mock_serpapi_web.assert_awaited_once_with("find mountain")
    mock_resolve_slot.assert_awaited()
    self.assertEqual(mock_resolve_slot.call_args.kwargs.get("provider"), "serpapi")

  async def test_arch_gemini_with_global_serpapi_uses_gemini_grounding(self):
    seen_config = {}

    async def fake_stream(**kw):
      seen_config.update(kw)
      async def gen():
        yield _Chunk("Gemini grounded answer.")
      return gen()

    fake_client = mock.MagicMock()
    fake_client.aio.models.generate_content_stream = fake_stream

    mock_serpapi_web = mock.AsyncMock()
    events = []
    mock_emit = mock.AsyncMock(side_effect=lambda ev, d: events.append((ev, d)))

    req = pipeline.SearchRequest(query="gemini test", arch="gemini")
    image_tasks = []

    with _setting(serpapi_api_key="k", web_provider="serpapi", image_provider="serpapi"), \
         mock.patch.dict(providers.WEB_PROVIDERS, {"serpapi": mock_serpapi_web}), \
         mock.patch.object(pipeline, "client", return_value=fake_client), \
         mock.patch.object(pipeline, "related_questions", mock.AsyncMock(return_value=[])):
      await pipeline._run_search(req, mock_emit, image_tasks)

    # serpapi_web should NOT have been called because arch="gemini" overrides web_provider
    mock_serpapi_web.assert_not_awaited()
    tools = seen_config["config"].tools
    self.assertIsNotNone(tools)
    self.assertTrue(any(t.google_search is not None for t in tools))


class SearchImagesProviderOverrideTest(unittest.IsolatedAsyncioTestCase):

  async def test_search_images_provider_param_overrides_settings(self):
    cands = [images.Candidate(image_url="https://example.com/img.jpg")]
    mock_openverse = mock.AsyncMock(return_value=cands)
    mock_gemini_images = mock.AsyncMock(return_value=[])

    with _setting(image_provider="gemini"), \
         mock.patch.dict(providers.IMAGE_PROVIDERS, {"openverse": mock_openverse}), \
         mock.patch.object(images, "google_image_search", mock_gemini_images):
      out = await images.search_images("cat", "context", provider="openverse")

    self.assertEqual(out, cands)
    mock_openverse.assert_awaited_once_with("cat")
    mock_gemini_images.assert_not_awaited()


if __name__ == "__main__":
  unittest.main()
