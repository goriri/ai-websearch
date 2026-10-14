"""Tests for third-party providers and the third-party web-search pipeline mode
(network and Gemini mocked)."""

import contextlib
import unittest
from unittest import mock

from app import images, pipeline, providers
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


class _Resp:
  def __init__(self, data):
    self._data = data

  def raise_for_status(self):
    pass

  def json(self):
    return self._data


def _fake_http(data):
  m = mock.MagicMock()
  m.get = mock.AsyncMock(return_value=_Resp(data))
  return m


class ProviderParsingTest(unittest.IsolatedAsyncioTestCase):

  async def test_serpapi_images(self):
    data = {"images_results": [
        {"original": "https://a.com/1.jpg", "link": "https://a.com/p", "source": "a.com",
         "title": "<b>One</b>", "original_width": 1200},
        {"original": "https://b.com/s.jpg", "link": "https://b.com/p", "original_width": 120},  # too small
        {"original": "https://c.com/x.jpg", "link": "https://c.com/p", "is_product": True},  # product
        {"thumbnail": "https://t.com/t.jpg", "link": "https://www.d.com/p"},
    ]}
    http = _fake_http(data)
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      out = await providers.serpapi_images("q")
    self.assertEqual([c.image_url for c in out], ["https://a.com/1.jpg", "https://t.com/t.jpg"])
    self.assertEqual(out[0].title, "One")
    self.assertEqual(out[1].domain, "d.com")
    self.assertEqual(http.get.await_args.kwargs["params"]["engine"], "google_images")

  async def test_brave_images(self):
    data = {"results": [{"url": "https://p.com/a", "source": "p.com", "title": "T",
                         "properties": {"url": "https://p.com/a.jpg"}}]}
    http = _fake_http(data)
    with _setting(brave_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      out = await providers.brave_images("q")
    self.assertEqual(out[0].image_url, "https://p.com/a.jpg")
    self.assertEqual(http.get.await_args.kwargs["headers"]["X-Subscription-Token"], "k")

  async def test_openverse_images(self):
    data = {"results": [{"url": "https://live.staticflickr.com/x.jpg", "title": "Owl",
                         "foreign_landing_url": "https://www.flickr.com/photos/x",
                         "license": "by", "license_version": "2.0", "creator": "Ann"}]}
    with mock.patch.object(providers, "api_http", return_value=_fake_http(data)):
      out = await providers.openverse_images("owl")
    self.assertEqual(out[0].domain, "flickr.com")
    self.assertEqual(out[0].license, "CC BY 2.0 · Ann")

  async def test_missing_key_raises(self):
    with _setting(serpapi_api_key=""):
      with self.assertRaises(providers.ProviderError):
        await providers.serpapi_images("q")

  async def test_serpapi_web(self):
    data = {"answer_box": {"answer": "828 m", "title": "Height", "link": "https://x.com/h"},
            "organic_results": [{"title": "A", "link": "https://www.a.com/1", "snippet": "sa"},
                                {"title": "B", "link": "https://b.org/2", "snippet": "sb"}]}
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=_fake_http(data)):
      out = await providers.serpapi_web("q")
    self.assertEqual([r["domain"] for r in out], ["x.com", "a.com", "b.org"])
    self.assertEqual(out[0]["snippet"], "828 m")


class DispatchTest(unittest.IsolatedAsyncioTestCase):

  async def test_search_images_dispatch(self):
    cands = [images.Candidate(image_url="u")]
    with _setting(image_provider="openverse"), \
         mock.patch.dict(providers.IMAGE_PROVIDERS, {"openverse": mock.AsyncMock(return_value=cands)}), \
         mock.patch.object(images, "google_image_search", mock.AsyncMock()) as g:
      self.assertEqual(await images.search_images("q", "ctx"), cands)
      g.assert_not_awaited()

  async def test_provider_failure_returns_empty(self):
    with _setting(image_provider="serpapi", serpapi_api_key=""):
      self.assertEqual(await images.search_images("q", "ctx"), [])


class ThirdPartyCitationTest(unittest.TestCase):

  def test_brackets_to_tokens(self):
    md = "Tall tower [1]. Also [2, 3] and a [link](http://x) and [9]."
    out = pipeline.citations_from_brackets(md, 3)
    self.assertEqual(out, "Tall tower @@CITE:1@@. Also @@CITE:2,3@@ and a [link](http://x) and [9].")

  def test_context_block(self):
    ctx = pipeline.third_party_context([{"title": "T", "domain": "d.com", "snippet": "s", "url": "u"}])
    self.assertIn("[1] T (d.com)", ctx)


class ThirdPartyPipelineTest(unittest.IsolatedAsyncioTestCase):

  async def test_serpapi_web_mode_uses_context_not_tool(self):
    results = [{"title": "Src", "url": "https://s.com/a", "domain": "s.com", "snippet": "fact"}]
    seen = {}

    class _Part:
      def __init__(self, text):
        self.text, self.thought = text, False

    class _Chunk:
      def __init__(self, text):
        cand = mock.MagicMock()
        cand.grounding_metadata = None
        cand.content.parts = [_Part(text)]
        self.candidates = [cand]

    async def fake_stream(**kw):
      seen.update(kw)

      async def gen():
        yield _Chunk("The fact is true [1].")
      return gen()

    fake_client = mock.MagicMock()
    fake_client.aio.models.generate_content_stream = fake_stream
    events = []

    async def emit(ev, d):
      events.append((ev, d))

    with _setting(web_provider="serpapi"), \
         mock.patch.dict(providers.WEB_PROVIDERS, {"serpapi": mock.AsyncMock(return_value=results)}), \
         mock.patch.object(pipeline, "client", return_value=fake_client), \
         mock.patch.object(pipeline, "related_questions", mock.AsyncMock(return_value=[])):
      await pipeline.run_search(pipeline.SearchRequest(query="q"), emit)
    self.assertIsNone(seen["config"].tools)  # No Gemini search tool.
    self.assertIn("[1] Src (s.com)", seen["contents"][-1].parts[-1].text)
    final = dict(events)["final"]
    self.assertEqual(final["sources"][0]["url"], "https://s.com/a")
    self.assertIn("@@CITE:1@@", final["markdown"])
    self.assertEqual(final["search_entry_point"], "")


if __name__ == "__main__":
  unittest.main()
