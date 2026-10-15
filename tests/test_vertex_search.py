"""Tests for Vertex AI Search image provider and architecture D (vertexsearch)."""

import asyncio
import contextlib
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from google.genai import types
import httpx

from app import cost, genai_client, images, main, pipeline, providers
from app.config import settings


@contextlib.contextmanager
def _setting(**kw):
  old = {k: object.__getattribute__(settings, k) for k in kw}
  for k, v in kw.items():
    object.__setattr__(settings, k, v)
  try:
    yield
  finally:
    for k, v in old.items():
      object.__setattr__(settings, k, v)


SAMPLE_RESPONSE = {
    "results": [{
        "document": {
            "derivedStructData": {
                "link": "https://example.com/photo.jpg",
                "title": "Iconic Building - ArchDaily",
                "htmlTitle": "<b>Iconic Building</b>",
                "displayLink": "www.archdaily.com",
                "mime": "image/jpeg",
                "fileFormat": "image/jpeg",
                "image": {
                    "contextLink": "https://archdaily.com/article/1",
                    "width": 2000,
                    "height": 1333,
                    "byteSize": 369436,
                    "thumbnailLink": "https://encrypted-tbn0.gstatic.com/thumb",
                    "thumbnailWidth": 150,
                    "thumbnailHeight": 100,
                },
                "snippets": ["A stunning modern building."],
            }
        }
    }],
    "totalSize": 1,
}


def _make_item(link, width=None, context=None, display="www.example.com"):
  img = {}
  if width is not None:
    img["width"] = width
  if context is not None:
    img["contextLink"] = context
  return {
      "document": {
          "derivedStructData": {
              "link": link,
              "title": f"Title for {link}",
              "displayLink": display,
              "image": img,
          }
      }
  }


ROUND_ROBIN_RESP1 = {
    "results": [
        _make_item("https://a.com/1.jpg", width=500),
        _make_item("https://a.com/dup.jpg", width=600),
        _make_item("https://a.com/too_small.jpg", width=250),
        _make_item("ftp://a.com/nonhttp.jpg", width=500),
        _make_item("https://a.com/3.jpg", width=500),
    ]
}

ROUND_ROBIN_RESP2 = {
    "results": [
        _make_item("https://b.com/2.jpg", width=700),
        _make_item("https://a.com/dup.jpg", width=600),
        _make_item("https://b.com/no_width.jpg", width=None),
    ]
}


async def _fake_round_robin_post(url, **kw):
  mock_r = mock.MagicMock()
  mock_r.status_code = 200
  mock_r.raise_for_status = mock.MagicMock()
  if "eng-1" in url:
    mock_r.json.return_value = ROUND_ROBIN_RESP1
  else:
    mock_r.json.return_value = ROUND_ROBIN_RESP2
  return mock_r


FAILING_RESP_GOOD = {
    "results": [{
        "document": {
            "derivedStructData": {
                "link": "https://good.com/pic.jpg",
                "title": "Good Photo",
                "displayLink": "good.com",
                "image": {"width": 800},
            }
        }
    }]
}


async def _fake_failing_post(url, **kw):
  if "eng-bad" in url:
    r = httpx.Response(500, request=httpx.Request("POST", url))
    raise httpx.HTTPStatusError("Server error", request=r.request, response=r)
  mock_r = mock.MagicMock()
  mock_r.status_code = 200
  mock_r.raise_for_status = mock.MagicMock()
  mock_r.json.return_value = FAILING_RESP_GOOD
  return mock_r


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


async def _fake_chunk_stream():
  yield _Chunk("Here is a building:\n\n[[IMAGE: modern tower | glass tower]]\n\nDone.")


class VertexSearchProviderTest(unittest.IsolatedAsyncioTestCase):

  async def test_parsing_sample_response(self):
    mock_creds = mock.MagicMock()
    mock_creds.valid = True
    mock_creds.token = "fake-token"

    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = mock.MagicMock()
    mock_resp.json.return_value = SAMPLE_RESPONSE

    fake_http = mock.MagicMock()
    fake_http.post = mock.AsyncMock(return_value=mock_resp)

    meter = cost.start()
    with _setting(vertex_search_engines=("prism-images-1",), vertex_search_location="global"), \
         mock.patch.object(providers, "_get_vertex_credentials", mock.AsyncMock(return_value=mock_creds)), \
         mock.patch.object(genai_client, "project_id", return_value="my-project"), \
         mock.patch.object(providers, "api_http", return_value=fake_http):
      cands = await providers.vertex_search_images("modern building")

    self.assertEqual(len(cands), 1)
    cand = cands[0]
    self.assertEqual(cand.image_url, "https://example.com/photo.jpg")
    self.assertEqual(cand.source_url, "https://archdaily.com/article/1")
    self.assertEqual(cand.title, "Iconic Building - ArchDaily")
    self.assertEqual(cand.domain, "archdaily.com")
    self.assertEqual(cand.kind, "web")

    fake_http.post.assert_awaited_once()
    call_args, call_kwargs = fake_http.post.call_args
    url = call_args[0]
    self.assertIn("projects/my-project/locations/global", url)
    self.assertIn("engines/prism-images-1/servingConfigs/default_search:search", url)
    headers = call_kwargs["headers"]
    self.assertEqual(headers["Authorization"], "Bearer fake-token")
    self.assertEqual(headers["X-Goog-User-Project"], "my-project")
    self.assertEqual(call_kwargs["json"], {"query": "modern building", "pageSize": 10, "params": {"search_type": 1}})
    self.assertEqual(call_kwargs["timeout"], 15.0)

    # Cost meter records one vertex search query.
    self.assertEqual(meter.vertex_search_queries, 1)

  async def test_round_robin_merge_dedupe_and_width_filter(self):
    mock_creds = mock.MagicMock(valid=True, token="token")

    fake_http = mock.MagicMock()
    fake_http.post = mock.AsyncMock(side_effect=_fake_round_robin_post)

    with _setting(vertex_search_engines=("eng-1", "eng-2")), \
         mock.patch.object(providers, "_get_vertex_credentials", mock.AsyncMock(return_value=mock_creds)), \
         mock.patch.object(genai_client, "project_id", return_value="proj"), \
         mock.patch.object(providers, "api_http", return_value=fake_http):
      cands = await providers.vertex_search_images("query", limit=8)

    urls = [c.image_url for c in cands]
    # Expected order: eng-1 #1, eng-2 #1, eng-1 #2, eng-2 #2, eng-1 #3
    self.assertEqual(urls, [
        "https://a.com/1.jpg",
        "https://b.com/2.jpg",
        "https://a.com/dup.jpg",
        "https://b.com/no_width.jpg",
        "https://a.com/3.jpg",
    ])
    self.assertNotIn("https://a.com/too_small.jpg", urls)
    self.assertNotIn("ftp://a.com/nonhttp.jpg", urls)

  async def test_one_engine_failing_does_not_fail_others(self):
    mock_creds = mock.MagicMock(valid=True, token="token")

    fake_http = mock.MagicMock()
    fake_http.post = mock.AsyncMock(side_effect=_fake_failing_post)

    with _setting(vertex_search_engines=("eng-bad", "eng-good")), \
         mock.patch.object(providers, "_get_vertex_credentials", mock.AsyncMock(return_value=mock_creds)), \
         mock.patch.object(genai_client, "project_id", return_value="proj"), \
         mock.patch.object(providers, "api_http", return_value=fake_http):
      cands = await providers.vertex_search_images("query")

    self.assertEqual(len(cands), 1)
    self.assertEqual(cands[0].image_url, "https://good.com/pic.jpg")

  async def test_no_engines_configured_raises_provider_error_and_search_images_returns_empty(self):
    with _setting(vertex_search_engines=()):
      with self.assertRaises(providers.ProviderError):
        await providers.vertex_search_images("query")

      res = await images.search_images("query", "context", provider="vertex_search")
      self.assertEqual(res, [])


class VertexSearchCostTest(unittest.TestCase):

  def test_vertex_search_cost_per_query_default_and_override(self):
    self.assertEqual(cost.vertex_search_per_query(), 0.004)
    with mock.patch.dict(os.environ, {"VERTEX_SEARCH_COST_PER_QUERY": "0.008"}):
      self.assertEqual(cost.vertex_search_per_query(), 0.008)

  def test_cost_meter_item_price_and_summary_note(self):
    meter = cost.Meter()
    meter.add_vertex_search(2)
    s = meter.summary()
    item = [it for it in s["items"] if it["label"] == "Vertex AI Search"][0]
    self.assertEqual(item["usd"], 0.008)
    self.assertEqual(item["detail"], "2 queries")
    self.assertEqual(s["total_usd"], 0.008)
    self.assertIn("Vertex AI Search $0.004/query (Enterprise)", s["note"])

  def test_cost_meter_single_query_singular_detail(self):
    meter = cost.Meter()
    meter.add_vertex_search()
    s = meter.summary()
    item = [it for it in s["items"] if it["label"] == "Vertex AI Search"][0]
    self.assertEqual(item["usd"], 0.004)
    self.assertEqual(item["detail"], "1 query")

  def test_note_does_not_contain_vertex_when_not_used(self):
    meter = cost.Meter()
    s = meter.summary()
    self.assertNotIn("Vertex AI Search", s["note"])

  def test_record_vertex_search_module_helper(self):
    m = cost.start()
    cost.record_vertex_search()
    cost.record_vertex_search(2)
    self.assertEqual(m.vertex_search_queries, 3)


class _FakeStreamHandler:

  def __init__(self, seen_config):
    self.seen_config = seen_config

  async def stream(self, **kw):
    self.seen_config.update(kw)
    return _fake_chunk_stream()


class VertexSearchPipelineAndApiTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  async def test_arch_vertexsearch_dispatch_resolve_slot_and_gemini_tools(self):
    seen_config = {}
    fake_client = mock.MagicMock()
    fake_client.aio.models.generate_content_stream = _FakeStreamHandler(seen_config).stream

    mock_resolve_slot = mock.AsyncMock(return_value=None)
    events = []
    mock_emit = mock.AsyncMock(side_effect=lambda ev, d: events.append((ev, d)))

    req = pipeline.SearchRequest(query="modern tower", arch="vertexsearch")
    image_tasks = []

    with _setting(vertex_search_engines=("prism-images-1",), web_provider="serpapi", image_provider="gemini"), \
         mock.patch.object(pipeline, "client", return_value=fake_client), \
         mock.patch.object(images, "resolve_slot", mock_resolve_slot), \
         mock.patch.object(pipeline, "related_questions", mock.AsyncMock(return_value=[])):
      await pipeline._run_search(req, mock_emit, image_tasks)
      if image_tasks:
        await asyncio.gather(*image_tasks, return_exceptions=True)

    # resolve_slot must receive provider="vertex_search"
    mock_resolve_slot.assert_awaited()
    self.assertEqual(mock_resolve_slot.call_args.kwargs.get("provider"), "vertex_search")

    # Gemini tools must be present (Grounding with Google Search)
    tools = seen_config["config"].tools
    self.assertIsNotNone(tools)
    self.assertTrue(any(t.google_search is not None for t in tools))

  def test_main_400_without_engines(self):
    with _setting(vertex_search_engines=()):
      res = self.client.post("/api/search", json={"query": "test query", "arch": "vertexsearch"})
      self.assertEqual(res.status_code, 400)
      self.assertIn("This architecture needs VERTEX_SEARCH_ENGINES", res.text)

  def test_api_config_archs_with_engines(self):
    with _setting(serpapi_api_key="", vertex_search_engines=("prism-images-1",)):
      res = self.client.get("/api/config")
      self.assertEqual(res.status_code, 200)
      self.assertEqual(res.json()["archs"], ["gemini", "vertexsearch"])

    with _setting(serpapi_api_key="key", vertex_search_engines=("prism-images-1", "prism-images-2")):
      res = self.client.get("/api/config")
      self.assertEqual(res.status_code, 200)
      self.assertEqual(res.json()["archs"], ["gemini", "thirdparty", "aimode", "vertexsearch"])


if __name__ == "__main__":
  unittest.main()
