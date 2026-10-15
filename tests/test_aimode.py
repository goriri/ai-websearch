"""Tests for the Google AI Mode (SerpApi) architecture: rendering, pipeline,
API dispatch and provider parsing. Network and Gemini are mocked."""

import contextlib
import copy
import io
import re
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from app import aimode, images, main, pipeline, providers
from app.config import settings
from app.textproc import slot_token


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


def _fake_http(*payloads):
  m = mock.MagicMock()
  m.get = mock.AsyncMock(side_effect=[_Resp(p) for p in payloads])
  return m


IMG1 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:1"
IMG2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:2"
IMG3 = "https://encrypted-tbn1.gstatic.com/images?q=tbn:3"
ALPHA = "https://encrypted-tbn2.gstatic.com/images?q=tbn:alpha"
BETA = "https://encrypted-tbn2.gstatic.com/images?q=tbn:beta"

FIXTURE = {
    "text_blocks": [
        {"type": "paragraph", "snippet": "Intro paragraph.", "reference_indexes": [0]},
        {"type": "heading", "snippet": "Top picks"},
        {"type": "paragraph", "snippet": "Para A.", "reference_indexes": [2, 1, 9, 1]},
        {"type": "list", "list": [
            {"title": "Alpha", "snippet": "first", "reference_indexes": [0],
             "thumbnail": ALPHA},
            {"title": "Beta", "snippet": "second", "shopping_result": {"thumbnail": BETA},
             "list": [{"snippet": "nested item", "reference_indexes": [2]}]},
            # Same URL as the first inline image: must be deduplicated.
            {"title": "Gamma", "snippet": "third", "local_result": {"thumbnail": IMG1}},
        ]},
        {"type": "heading", "snippet": "Details"},
        {"type": "table", "table": [["Name", "Note"], ["A|B", "x"], ["C"]]},
        {"type": "paragraph", "snippet": "Para B."},
        {"type": "code_block", "language": "python", "code": "print(1)\n"},
        {"type": "expandable", "title": "More", "text_blocks": [
            {"type": "paragraph", "snippet": "Hidden detail.", "reference_indexes": [2]},
        ]},
        {"type": "chart_block", "snippet": "CHART-SHOULD-NOT-RENDER"},
        {"type": "heading", "snippet": "Extra"},
        {"type": "paragraph", "snippet": "Para C."},
        {"type": "mystery", "snippet": "Unknown block text."},
    ],
    "references": [
        {"index": 0, "title": "<b>Ref</b> Zero", "link": "https://www.zero.com/a",
         "snippet": "s0", "source": "Zero"},
        {"index": 1, "title": "Ref One", "link": "https://one.org/b", "snippet": "s1",
         "source": "One"},
        {"index": 2, "title": "", "link": "https://two.net/c", "snippet": "s2",
         "source": "Two Source"},
    ],
    "inline_images": [IMG1, IMG2, IMG3],
    "related_questions": [
        {"question": "What else?", "serpapi_link": "https://serpapi.com/x"},
        {"question": "And then?", "serpapi_link": "https://serpapi.com/y"},
    ],
}


def _slot_positions(md):
  return [m.group(1) for m in re.finditer(r"@@SLOT:(\w+)@@", md)]


class RenderTest(unittest.TestCase):

  def setUp(self):
    self.r = aimode.render(copy.deepcopy(FIXTURE))
    self.md = self.r.markdown

  def test_headings_and_paragraphs(self):
    self.assertIn("### Top picks", self.md)
    self.assertIn("### Details", self.md)
    self.assertIn("Intro paragraph.@@CITE:1@@", self.md)
    self.assertIn("Unknown block text.", self.md)
    self.assertNotIn("CHART-SHOULD-NOT-RENDER", self.md)

  def test_citations_are_one_based_sorted_and_deduped(self):
    # reference index 9 does not exist and is dropped.
    self.assertIn("Para A.@@CITE:2,3@@", self.md)
    self.assertIn("Hidden detail.@@CITE:3@@", self.md)

  def test_sources(self):
    self.assertEqual(self.r.sources, [
        {"n": 1, "title": "Ref Zero", "url": "https://www.zero.com/a", "domain": "zero.com"},
        {"n": 2, "title": "Ref One", "url": "https://one.org/b", "domain": "one.org"},
        {"n": 3, "title": "Two Source", "url": "https://two.net/c", "domain": "two.net"},
    ])

  def test_reference_index_mapping(self):
    resp = {"text_blocks": [{"type": "paragraph", "snippet": "P", "reference_indexes": [7]}],
            "references": [{"index": 7, "title": "T", "link": "https://x.com"}]}
    self.assertIn("P@@CITE:1@@", aimode.render(resp).markdown)

  def test_list(self):
    self.assertIn("- **Alpha** first@@CITE:1@@", self.md)
    self.assertIn("- **Beta** second\n  - nested item@@CITE:3@@", self.md)
    self.assertIn("- **Gamma** third", self.md)

  def test_gfm_table(self):
    self.assertIn("| Name | Note |\n| --- | --- |\n| A\\|B | x |\n| C |  |", self.md)

  def test_code_and_expandable(self):
    self.assertIn("```python\nprint(1)\n```", self.md)
    self.assertIn("**More**\n\nHidden detail.", self.md)

  def test_list_thumbnail_slots_follow_the_list(self):
    slots = {s["id"]: s for s in self.r.slots}
    list_end = self.md.index("- **Gamma** third")
    after_list = self.md[list_end:self.md.index("### Details")]
    ids = _slot_positions(after_list)
    self.assertEqual([slots[i]["image_url"] for i in ids], [ALPHA, BETA, IMG1])
    self.assertEqual([slots[i]["caption"] for i in ids], ["Alpha", "Beta", "Gamma"])

  def test_inline_images_after_intro_and_first_paragraph_under_headings(self):
    slots = {s["id"]: s for s in self.r.slots}
    # IMG1 was already used by a list item, so the intro gets IMG2 (hero image)
    # and Para A (first paragraph under the first heading) gets IMG3.
    intro = "Intro paragraph.@@CITE:1@@"
    self.assertTrue(self.md.startswith(intro + slot_token("a1")))
    self.assertEqual(slots["a1"], {"id": "a1", "image_url": IMG2, "caption": ""})
    a = self.md.index("Para A.@@CITE:2,3@@") + len("Para A.@@CITE:2,3@@")
    sid = _slot_positions(self.md[a:])[0]
    self.assertTrue(self.md[a:].startswith(slot_token(sid)))
    self.assertEqual(slots[sid]["image_url"], IMG3)
    # Inline images are exhausted: nothing after Para B or Para C.
    b = self.md.index("Para B.") + len("Para B.")
    self.assertFalse(self.md[b:].startswith(slot_token("a5")))
    c = self.md.index("Para C.") + len("Para C.")
    self.assertEqual(_slot_positions(self.md[c:]), [])

  def test_dedupe_and_ids(self):
    urls = [s["image_url"] for s in self.r.slots]
    self.assertEqual(len(urls), len(set(urls)))
    self.assertEqual(len(urls), 5)
    self.assertEqual([s["id"] for s in self.r.slots], ["a1", "a2", "a3", "a4", "a5"])
    self.assertEqual(_slot_positions(self.md), ["a1", "a2", "a3", "a4", "a5"])

  def test_max_images(self):
    r = aimode.render(copy.deepcopy(FIXTURE), max_images=2)
    self.assertEqual([s["image_url"] for s in r.slots], [ALPHA, BETA])
    self.assertEqual(_slot_positions(r.markdown), ["a1", "a2"])
    self.assertEqual(aimode.render(copy.deepcopy(FIXTURE), max_images=0).slots, [])

  def test_no_headings_uses_first_paragraph_only(self):
    resp = {"text_blocks": [{"type": "paragraph", "snippet": "One."},
                            {"type": "paragraph", "snippet": "Two."}],
            "inline_images": [IMG1, IMG2, IMG1]}
    r = aimode.render(resp)
    self.assertEqual(r.markdown, "One." + slot_token("a1") + "Two.")
    self.assertEqual(r.slots, [{"id": "a1", "image_url": IMG1, "caption": ""}])

  def test_related(self):
    self.assertEqual(self.r.related, ["What else?", "And then?"])

  def test_reconstructed_markdown_fallback(self):
    resp = {"text_blocks": [],
            "reconstructed_markdown": "## Title\n\nFirst para.\n\nSecond para.",
            "inline_images": [IMG1, IMG2, IMG1],
            "references": FIXTURE["references"]}
    r = aimode.render(resp)
    self.assertEqual(
        r.markdown,
        "## Title\n\nFirst para." + slot_token("a1") + slot_token("a2") + "Second para.")
    self.assertEqual([s["image_url"] for s in r.slots], [IMG1, IMG2])
    self.assertEqual(len(r.sources), 3)

  def test_empty_response(self):
    r = aimode.render({})
    self.assertEqual((r.markdown, r.sources, r.slots, r.related), ("", [], [], []))


class RunAiModeTest(unittest.IsolatedAsyncioTestCase):

  async def _run(self, resp=None, error=None):
    events = []

    async def emit(ev, d):
      events.append((ev, d))

    async def fake_download(cand, min_px=None):
      self.download_calls.append((cand, min_px))
      if cand.image_url == BETA:
        return None
      return (b"\xff\xd8jpeg", 120, 100)

    self.download_calls = []
    fake_client = mock.MagicMock()
    serp = mock.AsyncMock(return_value=resp, side_effect=error)
    with mock.patch.object(providers, "serpapi_ai_mode", serp), \
         mock.patch.object(images, "download", fake_download), \
         mock.patch.object(pipeline, "client", fake_client), \
         mock.patch.object(images, "client", fake_client), \
         mock.patch.object(pipeline, "related_questions", mock.AsyncMock()) as rq:
      await pipeline.run_ai_mode(pipeline.SearchRequest(query="best q", mode="aimode"), emit)
    self.assertFalse(fake_client.called)
    self.assertEqual(fake_client.mock_calls, [])
    rq.assert_not_called()
    serp.assert_awaited_once_with("best q")
    return events

  async def test_event_order_and_payloads(self):
    events = await self._run(copy.deepcopy(FIXTURE))
    names = [e for e, _ in events]
    self.assertEqual(names[0], "status")
    fi = names.index("final")
    stream = names[1:fi]
    self.assertTrue(set(stream) <= {"text", "slot"})
    self.assertEqual(stream.count("slot"), 5)
    self.assertEqual(stream[0], "text")
    self.assertEqual(sorted(names[fi + 1:-2]), ["image"] * 4 + ["image_failed"])
    self.assertEqual(names[-2:], ["related", "done"])

    final = events[fi][1]
    # The client's assembly of text deltas + slot tokens equals the final md.
    assembled = "".join(d["delta"] if e == "text" else slot_token(d["id"])
                        for e, d in events[1:fi])
    self.assertEqual(assembled, final["markdown"])
    self.assertEqual(final["model"], "serpapi:google_ai_mode")
    self.assertEqual(final["queries"], ["best q"])
    self.assertEqual(final["search_entry_point"], "")
    self.assertEqual(final["sources"][0]["domain"], "zero.com")
    self.assertEqual(dict(events)["related"], {"questions": ["What else?", "And then?"]})
    self.assertIn("elapsed_ms", events[-1][1])

    slot_evs = [d for e, d in events if e == "slot"]
    self.assertEqual(slot_evs[0], {"id": "a1", "query": "", "caption": ""})
    imgs = {d["id"]: d for e, d in events if e == "image"}
    a1 = imgs["a1"]
    self.assertEqual(a1["kind"], "web")
    self.assertEqual(a1["domain"], "encrypted-tbn0.gstatic.com")
    self.assertEqual(a1["source_url"], IMG2)
    self.assertEqual(a1["title"], "")
    self.assertTrue(a1["src"].startswith("data:image/jpeg;base64,"))
    self.assertEqual((a1["width"], a1["height"]), (120, 100))
    failed = [d["id"] for e, d in events if e == "image_failed"]
    self.assertEqual(failed, ["a4"])  # BETA (a1 is now the intro image)
    self.assertEqual(len(self.download_calls), 5)
    self.assertTrue(all(m == pipeline.AI_MODE_MIN_PX == 100 for _, m in self.download_calls))
    self.assertTrue(all(c.kind == "web" for c, _ in self.download_calls))

  async def test_provider_error(self):
    events = await self._run(error=providers.ProviderError("quota exceeded"))
    self.assertEqual([e for e, _ in events], ["status", "error", "done"])
    self.assertIn("quota exceeded", events[1][1]["message"])

  async def test_empty_answer(self):
    events = await self._run({"text_blocks": []})
    self.assertEqual([e for e, _ in events], ["status", "error", "done"])


class ApiAiModeTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_rejected_without_key(self):
    with _setting(serpapi_api_key=""):
      r = self.client.post("/api/search", json={"query": "hi", "mode": "aimode"})
      self.assertEqual(r.status_code, 400)
      self.assertIn("SERPAPI_API_KEY", r.json()["detail"])
      self.assertFalse(self.client.get("/api/config").json()["ai_mode"])

  def test_dispatch_with_key(self):
    calls = []

    async def fake(req, emit):
      calls.append(req.mode)
      await emit("done", {})

    with _setting(serpapi_api_key="k"), mock.patch.object(pipeline, "run_ai_mode", fake):
      r = self.client.post("/api/search", json={"query": "hi", "mode": "aimode"})
      self.assertEqual(r.status_code, 200)
      self.assertTrue(self.client.get("/api/config").json()["ai_mode"])
    self.assertEqual(calls, ["aimode"])


class ProviderAiModeTest(unittest.IsolatedAsyncioTestCase):

  async def test_ai_mode_params_and_timeout(self):
    http = _fake_http({"text_blocks": [{"type": "paragraph", "snippet": "x"}]})
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      out = await providers.serpapi_ai_mode("q")
    self.assertEqual(out["text_blocks"][0]["snippet"], "x")
    kw = http.get.await_args.kwargs
    self.assertEqual(kw["params"]["engine"], "google_ai_mode")
    self.assertEqual(kw["params"]["q"], "q")
    self.assertEqual(kw["params"]["api_key"], "k")
    self.assertEqual(kw["timeout"].read, 30.0)

  async def test_ai_mode_error(self):
    http = _fake_http({"error": "Google hasn't returned any results for this query."})
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      with self.assertRaisesRegex(providers.ProviderError, "any results"):
        await providers.serpapi_ai_mode("q")

  async def test_missing_key(self):
    with _setting(serpapi_api_key=""):
      with self.assertRaises(providers.ProviderError):
        await providers.serpapi_ai_mode("q")
      with self.assertRaises(providers.ProviderError):
        await providers.serpapi_ai_overview("q")

  async def test_ai_overview_page_token_follow_up(self):
    http = _fake_http(
        {"ai_overview": {"page_token": "TOKEN"}},
        {"ai_overview": {"text_blocks": [{"type": "paragraph", "snippet": "AO"}]}},
    )
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      out = await providers.serpapi_ai_overview("q")
    self.assertEqual(out["text_blocks"][0]["snippet"], "AO")
    first, second = http.get.await_args_list
    self.assertEqual(first.kwargs["params"]["engine"], "google")
    self.assertEqual(second.kwargs["params"]["engine"], "google_ai_overview")
    self.assertEqual(second.kwargs["params"]["page_token"], "TOKEN")

  async def test_ai_overview_inline_and_absent(self):
    inline = {"ai_overview": {"text_blocks": [{"type": "paragraph", "snippet": "AO"}]}}
    http = _fake_http(inline, {"organic_results": []})
    with _setting(serpapi_api_key="k"), mock.patch.object(providers, "api_http", return_value=http):
      self.assertEqual(await providers.serpapi_ai_overview("q"), inline["ai_overview"])
      self.assertEqual(await providers.serpapi_ai_overview("q"), {})
    self.assertEqual(http.get.await_count, 2)


class MinPxTest(unittest.TestCase):

  def test_min_px_allows_small_thumbnails(self):
    buf = io.BytesIO()
    Image.new("RGB", (120, 110), (10, 20, 30)).save(buf, format="PNG")
    with self.assertRaises(ValueError):
      images.normalize_image(buf.getvalue(), 1024)  # Default minimum unchanged.
    _, w, h = images.normalize_image(buf.getvalue(), 1024, min_px=100)
    self.assertEqual((w, h), (120, 110))
    with self.assertRaises(ValueError):
      images.normalize_image(buf.getvalue(), 1024, min_px=200)


if __name__ == "__main__":
  unittest.main()
