"""Third-party search providers (alternative to Gemini's built-in Google Search
grounding).

Image search providers return `images.Candidate` lists and feed the same
download → normalize → visual re-rank → dedupe path as Gemini image search.
The web search provider returns plain result dicts that `pipeline.py` passes to
Gemini as context, so the answer model doesn't need Gemini's grounding tool.

Providers:
  images: gemini (default) | serpapi (Google Images via SerpApi) | brave |
          openverse (keyless, CC-licensed images only)
  web:    gemini (default, Grounding with Google Search) | serpapi (Google web
          results via SerpApi)
  answer: serpapi_ai_mode / serpapi_ai_overview return a complete Google AI
          Mode / AI Overview answer (used by the "aimode" architecture)
"""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

import httpx

from app import cost
from app.config import settings
from app.textproc import clean_title

log = logging.getLogger(__name__)

_api_http: Optional[httpx.AsyncClient] = None


def api_http() -> httpx.AsyncClient:
  global _api_http
  if _api_http is None:
    # SerpApi's uncached Google Images latency varies widely (~1-17 s).
    _api_http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
  return _api_http


def _domain(url: str) -> str:
  return (urlparse(url).hostname or "").removeprefix("www.")


def _cache_params() -> dict:
  """SERPAPI_NO_CACHE=1 forces fresh results (benchmarks); cached hits are free."""
  return {"no_cache": "true"} if os.environ.get("SERPAPI_NO_CACHE") else {}


class ProviderError(RuntimeError):
  pass


def _require(key: str, name: str) -> str:
  if not key:
    raise ProviderError(f"{name} API key is not configured")
  return key


# --------------------------------------------------------------------------
# Image search
# --------------------------------------------------------------------------


async def serpapi_images(query: str, limit: int = 8):
  """Google Images results via SerpApi (engine=google_images)."""
  from app.images import Candidate  # Local import avoids a cycle.

  key = _require(settings.serpapi_api_key, "SerpApi")
  r = await api_http().get("https://serpapi.com/search.json", params={
      "engine": "google_images", "q": query, "api_key": key,
      "hl": "en", "safe": "active", "ijn": 0, **_cache_params(),
  })
  r.raise_for_status()
  cost.record_serpapi("google_images")
  out = []
  for it in r.json().get("images_results", [])[: limit * 2]:
    url = it.get("original") or it.get("thumbnail")
    if not url or it.get("is_product"):
      continue
    if (it.get("original_width") or 1000) < 300:
      continue
    page = it.get("link", "")
    out.append(Candidate(image_url=url, source_url=page, title=clean_title(it.get("title")),
                         domain=it.get("source") or _domain(page), kind="web"))
    if len(out) >= limit:
      break
  return out


async def brave_images(query: str, limit: int = 8):
  """Brave Search image results."""
  from app.images import Candidate

  key = _require(settings.brave_api_key, "Brave Search")
  r = await api_http().get(
      "https://api.search.brave.com/res/v1/images/search",
      params={"q": query, "count": limit * 2, "safesearch": "strict"},
      headers={"X-Subscription-Token": key, "Accept": "application/json"},
  )
  r.raise_for_status()
  out = []
  for it in r.json().get("results", []):
    props = it.get("properties") or {}
    url = props.get("url") or (it.get("thumbnail") or {}).get("src")
    if not url:
      continue
    page = it.get("url", "")
    out.append(Candidate(image_url=url, source_url=page, title=clean_title(it.get("title")),
                         domain=it.get("source") or _domain(page), kind="web"))
    if len(out) >= limit:
      break
  return out


async def openverse_images(query: str, limit: int = 8):
  """Openverse (WordPress) CC-licensed image search. Keyless, rate-limited.

  Openverse matches all keywords, so long, specific slot queries often return
  nothing; retry with the first few words.
  """
  words = query.split()
  for q in dict.fromkeys([query, " ".join(words[:4]), " ".join(words[:2])]):
    out = await _openverse_once(q, limit)
    if out:
      return out
  return []


async def _openverse_once(query: str, limit: int):
  from app.images import Candidate

  r = await api_http().get(
      "https://api.openverse.org/v1/images/",
      params={"q": query, "page_size": limit, "mature": "false"},
      headers={"User-Agent": settings.wikimedia_user_agent},
  )
  r.raise_for_status()
  out = []
  for it in r.json().get("results", []):
    url = it.get("url")
    if not url:
      continue
    lic = " ".join(x for x in ((it.get("license") or "").upper(), it.get("license_version") or "") if x)
    creator = it.get("creator") or ""
    out.append(Candidate(
        image_url=url, source_url=it.get("foreign_landing_url") or url,
        title=clean_title(it.get("title")), domain=_domain(it.get("foreign_landing_url") or url),
        kind="web", license=" · ".join(x for x in (f"CC {lic}" if lic else "", creator[:60]) if x)))
  return out


IMAGE_PROVIDERS = {
    "serpapi": serpapi_images,
    "brave": brave_images,
    "openverse": openverse_images,
}


# --------------------------------------------------------------------------
# Web search (answer context)
# --------------------------------------------------------------------------


async def serpapi_web(query: str, limit: int = 8) -> list[dict]:
  """Google web results via SerpApi (engine=google)."""
  key = _require(settings.serpapi_api_key, "SerpApi")
  r = await api_http().get("https://serpapi.com/search.json", params={
      "engine": "google", "q": query, "api_key": key, "hl": "en", "num": limit,
      **_cache_params(),
  })
  r.raise_for_status()
  cost.record_serpapi("google")
  data = r.json()
  results = []
  ab = data.get("answer_box") or {}
  if ab.get("snippet") or ab.get("answer"):
    results.append({"title": clean_title(ab.get("title") or "Answer"),
                    "url": ab.get("link", ""), "domain": _domain(ab.get("link", "")),
                    "snippet": ab.get("answer") or ab.get("snippet")})
  for it in data.get("organic_results", []):
    if not it.get("link"):
      continue
    results.append({"title": clean_title(it.get("title")), "url": it["link"],
                    "domain": _domain(it["link"]), "snippet": it.get("snippet", "")})
    if len(results) >= limit:
      break
  return results


WEB_PROVIDERS = {"serpapi": serpapi_web}


# --------------------------------------------------------------------------
# Google AI Mode / AI Overview (complete answers, used by mode="aimode")
# --------------------------------------------------------------------------

_SERPAPI_URL = "https://serpapi.com/search.json"
# AI Mode answers are generated on demand and can take much longer than a
# plain results page.
_AI_TIMEOUT = httpx.Timeout(30.0, connect=4.0)


async def _serpapi_json(params: dict) -> dict:
  """Calls SerpApi and returns the JSON body.

  Raises:
    ProviderError: if the key is missing, the body carries an `error`, or the
      body is not a JSON object.
    httpx.HTTPError: on transport / HTTP status errors without an error body.
  """
  key = _require(settings.serpapi_api_key, "SerpApi")
  r = await api_http().get(
      _SERPAPI_URL, params=params | {"api_key": key} | _cache_params(),
      timeout=_AI_TIMEOUT,
  )
  try:
    data = r.json()
  except ValueError:
    data = None
  # SerpApi reports failures (including non-2xx ones) as {"error": "..."}.
  if isinstance(data, dict) and data.get("error"):
    raise ProviderError(str(data["error"]))
  r.raise_for_status()
  if not isinstance(data, dict):
    raise ProviderError("SerpApi returned an invalid response")
  cost.record_serpapi(params.get("engine", "google"))
  return data


async def serpapi_ai_mode(query: str) -> dict:
  """Google AI Mode answer via SerpApi (engine=google_ai_mode).

  Returns the raw response: text_blocks, references, inline_images,
  related_questions, reconstructed_markdown, ...
  """
  return await _serpapi_json(
      {"engine": "google_ai_mode", "q": query, "hl": "en"}
  )


async def serpapi_ai_overview(query: str) -> dict:
  """Google AI Overview via SerpApi (engine=google -> `ai_overview`).

  If the results page only carries a `page_token`, the overview is fetched
  with engine=google_ai_overview (the token expires after ~60 s, so this is
  done immediately). Returns {} when Google showed no AI Overview.
  """
  data = await _serpapi_json({"engine": "google", "q": query, "hl": "en"})
  overview = data.get("ai_overview") or {}
  token = overview.get("page_token")
  if token and not overview.get("text_blocks"):
    data = await _serpapi_json(
        {"engine": "google_ai_overview", "page_token": token}
    )
    overview = data.get("ai_overview") or {}
  return overview
