"""Image retrieval, verification and generation for image slots.

Resolution chain for a slot (query + caption), depending on the user's image
preference:

  web / auto:  Google image search (via Gemini grounding) -> Wikimedia Commons
               -> (auto only) Gemini image generation
  ai:          Gemini image generation

Candidates found on the web are downloaded, validated and normalized with
Pillow, then (optionally) visually re-ranked by a small Gemini model that looks
at the actual pixels and picks the image that best matches the caption.
"""

import asyncio
import base64
import dataclasses
import io
import json
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx
from google.genai import types
from PIL import Image

from app.config import settings
from app.genai_client import client
from app.textproc import clean_title

log = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)
_MAX_DOWNLOAD = 12 * 1024 * 1024
_MIN_W, _MIN_H = 240, 160

Image.MAX_IMAGE_PIXELS = 60_000_000  # Guard against decompression bombs.


@dataclasses.dataclass
class Candidate:
  image_url: str
  source_url: str = ""
  title: str = ""
  domain: str = ""
  kind: str = "web"  # web | wikimedia | generated
  license: str = ""


@dataclasses.dataclass
class ResolvedImage:
  data_url: str
  width: int
  height: int
  kind: str
  source_url: str = ""
  title: str = ""
  domain: str = ""
  license: str = ""

  def to_event(self) -> dict:
    d = dataclasses.asdict(self)
    d["src"] = d.pop("data_url")
    return d


_http: Optional[httpx.AsyncClient] = None


def http() -> httpx.AsyncClient:
  global _http
  if _http is None:
    _http = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(8.0, connect=4.0),
        headers={"User-Agent": _BROWSER_UA, "Accept": "image/*,*/*;q=0.5"},
        limits=httpx.Limits(max_connections=40),
    )
  return _http


def _is_wikimedia(url: str) -> bool:
  host = urlparse(url).hostname or ""
  return host.endswith("wikimedia.org") or host.endswith("wikipedia.org")


# --------------------------------------------------------------------------
# Image normalization
# --------------------------------------------------------------------------


def normalize_image(raw: bytes, max_px: int) -> tuple[bytes, int, int]:
  """Decodes any common format, downsizes and re-encodes as JPEG.

  Returns (jpeg_bytes, width, height). Raises ValueError if unusable.
  """
  try:
    img = Image.open(io.BytesIO(raw))
    img.load()
  except Exception as e:  # pylint: disable=broad-except
    raise ValueError(f"undecodable image: {e}") from e
  if getattr(img, "is_animated", False):
    img.seek(0)
  w, h = img.size
  if w < _MIN_W or h < _MIN_H:
    raise ValueError(f"too small: {w}x{h}")
  if w / h > 4 or h / w > 4:
    raise ValueError(f"extreme aspect ratio: {w}x{h}")
  if img.mode in ("RGBA", "LA", "P"):
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    img = bg
  elif img.mode != "RGB":
    img = img.convert("RGB")
  img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
  out = io.BytesIO()
  img.save(out, format="JPEG", quality=82, optimize=True, progressive=True)
  return out.getvalue(), img.size[0], img.size[1]


def to_data_url(jpeg: bytes) -> str:
  return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


async def download(cand: Candidate) -> Optional[tuple[bytes, int, int]]:
  headers = {}
  if _is_wikimedia(cand.image_url):
    headers["User-Agent"] = settings.wikimedia_user_agent
  try:
    async with http().stream("GET", cand.image_url, headers=headers) as resp:
      if resp.status_code != 200:
        log.info("image %s -> HTTP %s", cand.image_url[:80], resp.status_code)
        return None
      ctype = resp.headers.get("content-type", "")
      if ctype and not ctype.startswith("image/") and "octet-stream" not in ctype:
        return None
      if "svg" in ctype:
        return None
      buf = bytearray()
      async for part in resp.aiter_bytes():
        buf.extend(part)
        if len(buf) > _MAX_DOWNLOAD:
          return None
      # Record the final (post-redirect) URL for display.
      cand.image_url = str(resp.url)
    return await asyncio.to_thread(normalize_image, bytes(buf), settings.image_max_px)
  except (httpx.HTTPError, ValueError) as e:
    log.info("image download failed %s: %s", cand.image_url[:80], e)
    return None


# --------------------------------------------------------------------------
# Candidate sources
# --------------------------------------------------------------------------


async def google_image_search(query: str, context: str) -> list[Candidate]:
  """Finds real web images using Gemini's Google image-search grounding.

  `image_search` grounding is only available on the Gemini image models. We
  ask the model a "what does it look like" question, which reliably triggers
  image search, and harvest the image grounding chunks (image URL + page URL).
  """
  prompt = (
      f"What does this look like: {query}?"
      + (f" (Context: {context[:300]})" if context else "")
      + " Look it up with Google image search, then reply with one short"
      " sentence. Text only."
  )
  for attempt in range(2):
    try:
      resp = await client().aio.models.generate_content(
          model=settings.image_model,
          contents=prompt,
          config=types.GenerateContentConfig(
              # Image models reject TEXT-only; IMAGE output is never used and
              # is made impossible by the small output-token cap.
              response_modalities=["TEXT", "IMAGE"],
              max_output_tokens=settings.image_search_max_tokens,
              tools=[
                  types.Tool(
                      google_search=types.GoogleSearch(
                          search_types=types.SearchTypes(
                              image_search=types.ImageSearch()
                          )
                      )
                  )
              ],
          ),
      )
    except Exception as e:  # pylint: disable=broad-except
      log.warning("image search failed for %r: %s", query, e)
      return []
    cands = []
    for cand in resp.candidates or []:
      gm = cand.grounding_metadata
      for ch in (gm.grounding_chunks or []) if gm else []:
        if ch.image and ch.image.image_uri:
          cands.append(
              Candidate(
                  image_url=ch.image.image_uri,
                  source_url=ch.image.source_uri or "",
                  title=clean_title(ch.image.title),
                  domain=ch.image.domain or "",
                  kind="web",
              )
          )
    if cands:
      return cands
    # The model occasionally answers from memory without searching; nudge it.
    prompt = (f"Search Google Images for photos of {query}, then reply with"
              " one short sentence. Text only.")
  return []


async def wikimedia_search(query: str, limit: int = 4) -> list[Candidate]:
  if not settings.enable_wikimedia:
    return []
  params = {
      "action": "query",
      "format": "json",
      "generator": "search",
      "gsrsearch": f"{query} filetype:bitmap",
      "gsrnamespace": 6,
      "gsrlimit": limit,
      "prop": "imageinfo",
      "iiprop": "url|extmetadata",
      "iiurlwidth": settings.image_max_px,
  }
  try:
    r = await http().get(
        "https://commons.wikimedia.org/w/api.php",
        params=params,
        headers={"User-Agent": settings.wikimedia_user_agent},
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
  except (httpx.HTTPError, ValueError) as e:
    log.info("wikimedia search failed for %r: %s", query, e)
    return []
  out = []
  for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
    info = (page.get("imageinfo") or [{}])[0]
    url = info.get("thumburl") or info.get("url")
    if not url:
      continue
    meta = info.get("extmetadata", {})
    lic = clean_title(meta.get("LicenseShortName", {}).get("value", ""))
    artist = clean_title(meta.get("Artist", {}).get("value", ""))
    out.append(
        Candidate(
            image_url=url,
            source_url=info.get("descriptionurl", ""),
            title=clean_title(page.get("title", "").removeprefix("File:")),
            domain="commons.wikimedia.org",
            kind="wikimedia",
            license=" · ".join(x for x in (lic, artist[:60]) if x),
        )
    )
  return out


async def generate_image(caption: str, context: str) -> Optional[ResolvedImage]:
  """Generates an illustrative image with Gemini (native image output)."""
  if not settings.allow_image_generation:
    return None  # Cost control: generation disabled.
  prompt = (
      "Create a single high-quality, realistic, informative illustration for a"
      f" search result about: {context[:300]}\nThe image should show: {caption}"
      "\nNo text overlays, no watermarks."
  )
  try:
    resp = await client().aio.models.generate_content(
        model=settings.image_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", image_size="1K"),
        ),
    )
  except Exception as e:  # pylint: disable=broad-except
    log.warning("image generation failed for %r: %s", caption, e)
    return None
  for cand in resp.candidates or []:
    for part in (cand.content.parts if cand.content else None) or []:
      if part.inline_data and part.inline_data.data and not part.thought:
        try:
          jpeg, w, h = await asyncio.to_thread(
              normalize_image, part.inline_data.data, settings.image_max_px
          )
        except ValueError:
          continue
        return ResolvedImage(
            data_url=to_data_url(jpeg), width=w, height=h, kind="generated",
            title="Generated with Gemini", domain="AI-generated",
        )
  return None


# --------------------------------------------------------------------------
# Visual re-ranking
# --------------------------------------------------------------------------


async def rerank(
    caption: str, query: str, images: list[tuple[Candidate, bytes]]
) -> Optional[int]:
  """Asks a small multimodal Gemini model which image best fits the caption.

  Returns the index into `images`, or None if none is a good match.
  """
  if len(images) == 0:
    return None
  parts: list[types.Part] = [
      types.Part.from_text(
          text=(
              "You are verifying images for a search results page.\n"
              f"Image slot search query: {query}\nIntended caption: {caption}\n"
              f"Below are {len(images)} candidate images, numbered from 0. Pick"
              " the single candidate that most accurately and attractively"
              " depicts the caption. Reject candidates that are mostly text,"
              " logos, screenshots of web pages, collages, watermarked stock"
              " previews, or that show a different subject. If none is"
              " acceptable return -1."
          )
      )
  ]
  for i, (cand, jpeg) in enumerate(images):
    parts.append(types.Part.from_text(text=f"Candidate {i} (from {cand.domain}):"))
    parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
  try:
    resp = await client().aio.models.generate_content(
        model=settings.lite_model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "best": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["best"],
            },
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        ),
    )
    best = int(json.loads(resp.text)["best"])
  except Exception as e:  # pylint: disable=broad-except
    log.info("rerank failed (%s); using first candidate", e)
    return 0
  if 0 <= best < len(images):
    return best
  return None


async def _first_valid(
    cands: list[Candidate], caption: str, query: str, exclude: set[str],
    take: int = 4,
) -> Optional[ResolvedImage]:
  """Downloads candidates in parallel, re-ranks and returns the winner.

  `exclude` holds final image URLs already used by other slots of the same
  answer, so that the same picture is never shown twice.
  """
  seen = set()
  uniq = []
  for c in cands:
    if c.image_url not in seen:
      seen.add(c.image_url)
      uniq.append(c)
  uniq = uniq[: take + 2]  # A couple of spares in case downloads fail.
  results = await asyncio.gather(*(download(c) for c in uniq))
  good = [
      (c, r) for c, r in zip(uniq, results)
      if r is not None and c.image_url not in exclude
  ][:take]
  if not good:
    return None
  idx = 0
  if settings.rerank_images:
    # Use smaller thumbnails for the ranking call to keep it fast.
    thumbs = []
    for c, (jpeg, _, _) in good:
      try:
        small, _, _ = await asyncio.to_thread(normalize_image, jpeg, 384)
      except ValueError:
        small = jpeg
      thumbs.append((c, small))
    idx = await rerank(caption, query, thumbs)
    if idx is None:
      return None
  # Another slot may have claimed this image while we were awaiting.
  order = [idx] + [i for i in range(len(good)) if i != idx]
  for i in order:
    if good[i][0].image_url not in exclude:
      idx = i
      break
  else:
    return None
  cand, (jpeg, w, h) = good[idx]
  exclude.add(cand.image_url)
  return ResolvedImage(
      data_url=to_data_url(jpeg), width=w, height=h, kind=cand.kind,
      source_url=cand.source_url or cand.image_url, title=cand.title,
      domain=cand.domain or (urlparse(cand.source_url).hostname or ""),
      license=cand.license,
  )


async def resolve_slot(
    query: str, caption: str, context: str, preference: str = "auto",
    exclude_urls: Optional[set[str]] = None,
) -> Optional[ResolvedImage]:
  """Finds the best image for a slot according to the user's preference.

  preference: "auto" (real web images, AI-generated fallback), "web" (real
  images only) or "ai" (always generate). When image generation is disabled
  (ALLOW_IMAGE_GENERATION=false), every preference behaves like "web".
  """
  if not settings.allow_image_generation:
    preference = "web"
  if preference == "ai":
    return await generate_image(caption, context)

  exclude = exclude_urls if exclude_urls is not None else set()
  img = None
  web = await google_image_search(query, context)
  if web:
    img = await _first_valid(web, caption, query, exclude)
  if img is None:
    wiki = await wikimedia_search(query)
    if wiki:
      img = await _first_valid(wiki, caption, query, exclude)
  if img is None and preference == "auto":
    img = await generate_image(caption, context)
  return img
