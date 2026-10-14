"""Runtime configuration, read from environment variables."""

import dataclasses
import os


def _bool(name: str, default: bool) -> bool:
  val = os.environ.get(name)
  if val is None:
    return default
  return val.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass(frozen=True)
class Settings:
  """All tunables for the app. Every field can be overridden via env vars."""

  # Empty -> resolved from Application Default Credentials at startup.
  project: str = os.environ.get(
      "GOOGLE_CLOUD_PROJECT", os.environ.get("PROJECT_ID", "")
  )
  location: str = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

  # Text answer models ("Fast" and "Pro" in the UI) + a fallback.
  answer_model: str = os.environ.get("ANSWER_MODEL", "gemini-3.8-flash")
  pro_model: str = os.environ.get("PRO_MODEL", "gemini-3.1-pro-preview")
  fallback_model: str = os.environ.get("FALLBACK_MODEL", "gemini-2.5-flash")

  # Native interleaved text+image model; also used for Google image search
  # grounding (image_search is only supported on the image models).
  image_model: str = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")

  # Small/fast multimodal model used for image re-ranking & related questions.
  lite_model: str = os.environ.get("LITE_MODEL", "gemini-3.1-flash-lite")

  rerank_images: bool = _bool("RERANK_IMAGES", True)
  enable_wikimedia: bool = _bool("ENABLE_WIKIMEDIA", True)
  max_image_slots: int = int(os.environ.get("MAX_IMAGE_SLOTS", "6"))
  image_concurrency: int = int(os.environ.get("IMAGE_CONCURRENCY", "6"))
  image_max_px: int = int(os.environ.get("IMAGE_MAX_PX", "1024"))
  max_query_chars: int = int(os.environ.get("MAX_QUERY_CHARS", "2000"))
  max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(7 * 1024 * 1024)))
  # Simple in-process rate limit: requests per minute per client IP (0 = off).
  rate_limit_per_min: int = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))

  # Descriptive UA required by Wikimedia's robot policy.
  wikimedia_user_agent: str = os.environ.get(
      "WIKIMEDIA_USER_AGENT",
      "PrismMultimodalSearch/1.0 (multimodal search demo on Google Cloud) httpx",
  )


settings = Settings()
