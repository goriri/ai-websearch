"""Shared, lazily-created Gemini client (Vertex AI / Gemini Enterprise Agent Platform)."""

import functools
import logging

import google.auth
from google import genai
from google.genai import types

from app import cost
from app.config import settings

log = logging.getLogger(__name__)

_instrument = cost.instrument


@functools.cache
def project_id() -> str:
  if settings.project:
    return settings.project
  _, proj = google.auth.default()
  if not proj:
    raise RuntimeError(
        "No GCP project configured. Set GOOGLE_CLOUD_PROJECT."
    )
  return proj


@functools.cache
def client() -> genai.Client:
  proj = project_id()
  log.info("Using Vertex AI project=%s location=%s", proj, settings.location)
  c = genai.Client(
      vertexai=True,
      project=proj,
      location=settings.location,
      http_options=types.HttpOptions(
          retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1.0),
      ),
  )
  return _instrument(c)
