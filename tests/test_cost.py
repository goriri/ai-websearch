"""Tests for cost metering, pricing calculation, and instrumentation."""

import asyncio
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app import cost, genai_client, main, pipeline, providers
from app.config import settings


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
  events = []
  for block in raw.split("\n\n"):
    block = block.strip()
    if not block or block.startswith(":"):
      continue
    ev, data = "message", ""
    for line in block.split("\n"):
      if line.startswith("event:"):
        ev = line.split(":", 1)[1].strip()
      elif line.startswith("data:"):
        data = line.split(":", 1)[1].strip()
    if data:
      import json
      try:
        events.append((ev, json.loads(data)))
      except Exception:
        events.append((ev, {"raw": data}))
  return events


class FakeCandidate:

  def __init__(self, web_q=None, img_q=None):
    gm = mock.MagicMock()
    gm.web_search_queries = web_q or []
    gm.image_search_queries = img_q or []
    self.grounding_metadata = gm


class FakeResponse:

  def __init__(self):
    self.usage_metadata = {
        "prompt_token_count": 200,
        "candidates_token_count": 300,
    }
    self.candidates = [FakeCandidate(web_q=["q1", "q2"])]


class FakeChunk:

  def __init__(self, text, usage=None, queries=None):
    self.text = text
    self.usage_metadata = usage
    self.candidates = [FakeCandidate(web_q=queries)] if queries else []


async def _fake_stream_gen():
  yield FakeChunk("chunk1", queries=["q3"])
  yield FakeChunk(
      "chunk2",
      usage={"prompt_token_count": 400, "candidates_token_count": 600},
  )


class FakeModels:

  async def generate_content(self, *a, **k):
    return FakeResponse()

  async def generate_content_stream(self, *a, **k):
    return _fake_stream_gen()


class FakeAio:

  def __init__(self, m):
    self.models = m


class FakeClient:

  def __init__(self):
    self.aio = FakeAio(FakeModels())


class FakeResp:

  def raise_for_status(self):
    pass

  def json(self):
    return {}


async def _worker_a():
  m = cost.start()
  cost.record_serpapi("google")
  await asyncio.sleep(0.01)
  cost.record_serpapi("google")
  return m.summary()


async def _worker_b():
  m = cost.start()
  await asyncio.sleep(0.005)
  cost.record_serpapi("google_images")
  return m.summary()


async def _fake_api_runner(req, emit):
  await emit("text", {"delta": "result"})
  await emit("done", {})


class CostTokenMathTest(unittest.TestCase):

  def test_token_math_and_pricing(self):
    meter = cost.Meter()
    usage = {
        "prompt_token_count": 100,
        "tool_use_prompt_token_count": 50,
        "candidates_token_count": 500,
        "thoughts_token_count": 200,
        "candidates_tokens_details": [
            {"modality": "IMAGE", "token_count": 100},
            {"modality": "TEXT", "token_count": 400},
        ],
    }
    meter.add_genai("gemini-3.1-flash-image", usage)
    s = meter.summary()
    self.assertEqual(len(s["items"]), 1)
    item = s["items"][0]
    self.assertEqual(item["label"], "gemini-3.1-flash-image")
    self.assertIn("150 in / 600 out tokens", item["detail"])
    self.assertIn("100 image tokens", item["detail"])
    self.assertEqual(item["usd"], 0.007875)
    self.assertEqual(s["total_usd"], 0.007875)
    self.assertNotIn("estimated", item)

  def test_unknown_model_marked_estimated(self):
    meter = cost.Meter()
    usage = {"prompt_token_count": 1000, "candidates_token_count": 500}
    meter.add_genai("gemini-custom-future", usage)
    s = meter.summary()
    item = s["items"][0]
    self.assertEqual(item["label"], "gemini-custom-future")
    self.assertTrue(item.get("estimated"))
    self.assertEqual(item["usd"], 0.002625)

  def test_gemini2_grounding_per_prompt(self):
    meter = cost.Meter()
    meter.add_genai("gemini-2.5-flash", None, grounding_queries=10, grounded=True)
    s = meter.summary()
    grounding = [it for it in s["items"] if it["label"] == "Google Search grounding"][0]
    self.assertEqual(grounding["usd"], cost.GROUNDING_PER_PROMPT_GEMINI2)
    self.assertIn("grounded prompt", grounding["detail"])

  def test_gemini3_grounding_per_query(self):
    meter = cost.Meter()
    meter.add_genai("gemini-3.8-flash", None, grounding_queries=7, grounded=True)
    s = meter.summary()
    grounding = [it for it in s["items"] if it["label"] == "Google Search grounding"][0]
    self.assertEqual(grounding["usd"], round(7 * cost.GROUNDING_PER_QUERY, 6))
    self.assertEqual(grounding["detail"], "7 search queries")

  def test_serpapi_env_override_and_summary_grouping(self):
    with mock.patch.dict(os.environ, {"SERPAPI_COST_PER_SEARCH": "0.02"}):
      meter = cost.Meter()
      meter.add_serpapi("google")
      meter.add_serpapi("google_images")
      meter.add_serpapi("google_images")
      s = meter.summary()
      serp = [it for it in s["items"] if it["label"] == "SerpApi"][0]
      self.assertEqual(serp["usd"], 0.06)
      self.assertEqual(serp["detail"], "3 searches (google ×1, google_images ×2)")
      self.assertIn("$0.02/search", s["note"])


class ConcurrentMeterTest(unittest.IsolatedAsyncioTestCase):

  async def test_meters_isolated_across_concurrent_tasks(self):
    sum_a, sum_b = await asyncio.gather(_worker_a(), _worker_b())
    serp_a = [it for it in sum_a["items"] if it["label"] == "SerpApi"][0]
    serp_b = [it for it in sum_b["items"] if it["label"] == "SerpApi"][0]
    self.assertIn("google ×2", serp_a["detail"])
    self.assertIn("google_images ×1", serp_b["detail"])


class InstrumentationTest(unittest.IsolatedAsyncioTestCase):

  async def test_instrument_generate_content_and_stream(self):
    client = FakeClient()
    cost.instrument(client)
    self.assertIs(genai_client._instrument, cost.instrument)

    meter = cost.start()
    await client.aio.models.generate_content(model="gemini-3.8-flash")
    stream = await client.aio.models.generate_content_stream(model="gemini-3.8-flash")
    async for _ in stream:
      pass

    s = meter.summary()
    gemini_item = [it for it in s["items"] if it["label"] == "gemini-3.8-flash"][0]
    self.assertIn("2 calls · 600 in / 900 out tokens", gemini_item["detail"])
    grounding_item = [it for it in s["items"] if it["label"] == "Google Search grounding"][0]
    self.assertEqual(grounding_item["detail"], "3 search queries")


class ProvidersCostTest(unittest.IsolatedAsyncioTestCase):

  async def test_providers_record_serpapi(self):
    fake_http = mock.MagicMock()
    fake_http.get = mock.AsyncMock(return_value=FakeResp())

    meter = cost.start()
    old_key = settings.serpapi_api_key
    try:
      object.__setattr__(settings, "serpapi_api_key", "key")
      with mock.patch.object(providers, "api_http", return_value=fake_http):
        await providers.serpapi_images("test")
        await providers.serpapi_web("test")
        await providers._serpapi_json({"engine": "google_ai_mode", "q": "test"})
    finally:
      object.__setattr__(settings, "serpapi_api_key", old_key)

    s = meter.summary()
    serp = [it for it in s["items"] if it["label"] == "SerpApi"][0]
    self.assertEqual(serp["usd"], 3 * 0.015)
    self.assertEqual(serp["detail"], "3 searches (google ×1, google_ai_mode ×1, google_images ×1)")


class ApiCostStreamTest(unittest.TestCase):

  def setUp(self):
    main._hits.clear()
    self.client = TestClient(main.app)

  def test_api_stream_ends_with_cost_then_done(self):
    with mock.patch.object(pipeline, "run_search", _fake_api_runner):
      r = self.client.post("/api/search", json={"query": "test query"})
    self.assertEqual(r.status_code, 200)
    evs = _parse_sse(r.text)
    event_names = [e for e, _ in evs]
    self.assertEqual(event_names[-2:], ["cost", "done"])
    cost_data = evs[-2][1]
    self.assertIn("total_usd", cost_data)
    self.assertIn("items", cost_data)


if __name__ == "__main__":
  unittest.main()
