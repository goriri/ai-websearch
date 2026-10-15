"""Renders a SerpApi Google AI Mode response into Prism's answer format.

The "aimode" architecture makes ONE SerpApi call (engine=google_ai_mode) that
already contains the answer as structured `text_blocks`, the `references`
they cite, a flat list of `inline_images` and `related_questions`. This module
is pure (no network): it turns that response into the same markdown the other
modes produce, i.e. GitHub-flavored Markdown with ``@@CITE:1,2@@`` citation
tokens (1-based source numbers) and ``textproc.slot_token(id)`` image slots.

Image placement:
  1. Thumbnails of list items (``thumbnail`` or the thumbnail of an attached
     ``shopping_result`` / ``local_result``) become slots right after the
     block that contains the list, captioned with the item title.
  2. ``inline_images`` (unpositioned) are then distributed in order: one slot
     after the intro paragraph and after the first paragraph of each heading
     section.
  Image URLs are deduplicated and the total is capped at ``max_images``.
"""

import dataclasses
import re
from typing import Any, Optional
from urllib.parse import urlparse

from app.textproc import cite_token, clean_title, slot_token


@dataclasses.dataclass
class Rendered:
  """A rendered AI Mode answer.

  Attributes:
    markdown: Answer markdown with citation and slot tokens.
    sources: [{n, title, url, domain}] in `references` order (n is 1-based).
    slots: [{id, image_url, caption}] in document order.
    related: Related questions suggested by Google.
  """

  markdown: str = ""
  sources: list[dict] = dataclasses.field(default_factory=list)
  slots: list[dict] = dataclasses.field(default_factory=list)
  related: list[str] = dataclasses.field(default_factory=list)


def _domain(url: str) -> str:
  return (urlparse(url).hostname or "").removeprefix("www.")


def _text(value: Any) -> str:
  return str(value).strip() if value else ""


def _one_line(value: Any) -> str:
  return re.sub(r"\s+", " ", _text(value))


def _image_url(value: Any) -> str:
  """Returns an http(s) image URL from a string or an image-like dict."""
  if isinstance(value, dict):
    value = value.get("image") or value.get("thumbnail") or value.get("original")
  url = _text(value)
  return url if url.startswith(("http://", "https://")) else ""


def _item_thumbnail(item: dict) -> str:
  for value in (
      item.get("thumbnail"),
      (item.get("shopping_result") or {}).get("thumbnail"),
      (item.get("local_result") or {}).get("thumbnail"),
  ):
    url = _image_url(value)
    if url:
      return url
  return ""


def _indent(text: str, depth: int) -> str:
  pad = "  " * depth
  return "\n".join(pad + line if line else line for line in text.split("\n"))


def _table_cell(value: Any) -> str:
  cell = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
  return cell.replace("|", r"\|")


def _table(rows: Any) -> str:
  """Renders a 2-D array (first row = header) as a GFM table."""
  rows = [r for r in (rows or []) if isinstance(r, list) and r]
  if not rows:
    return ""
  width = max(len(r) for r in rows)

  def line(row: list) -> str:
    cells = [_table_cell(c) for c in row] + [""] * (width - len(row))
    return "| " + " | ".join(cells) + " |"

  sep = "| " + " | ".join(["---"] * width) + " |"
  return "\n".join([line(rows[0]), sep] + [line(r) for r in rows[1:]])


def _code(language: Any, code: Any) -> str:
  code = str(code or "").rstrip("\n")
  if not code.strip():
    return ""
  fence = "```"
  while fence in code:
    fence += "`"
  lang = re.sub(r"[^\w+#.-]", "", str(language or ""))
  return f"{fence}{lang}\n{code}\n{fence}"


def _sources(resp: dict) -> tuple[list[dict], dict[int, int]]:
  """Returns (sources, map of reference `index` -> 1-based source number)."""
  sources: list[dict] = []
  ref_map: dict[int, int] = {}
  for pos, ref in enumerate(resp.get("references") or []):
    if not isinstance(ref, dict):
      continue
    url = _text(ref.get("link"))
    if not url:
      continue
    n = len(sources) + 1
    domain = _domain(url) or clean_title(ref.get("source"))
    sources.append({
        "n": n,
        "title": clean_title(ref.get("title")) or clean_title(ref.get("source"))
                 or domain,
        "url": url,
        "domain": domain,
    })
    index = ref.get("index", pos)
    if isinstance(index, int):
      ref_map[index] = n
  return sources, ref_map


class _BlockRenderer:
  """Renders text blocks to markdown, collecting list-item thumbnails."""

  def __init__(self, ref_map: dict[int, int]):
    self._ref_map = ref_map
    self._thumbs: list[tuple[str, str]] = []

  def take_thumbs(self) -> list[tuple[str, str]]:
    """Returns and clears the (url, caption) thumbnails seen since last call."""
    out, self._thumbs = self._thumbs, []
    return out

  def cite(self, obj: dict) -> str:
    nums = sorted({
        self._ref_map[i] for i in (obj.get("reference_indexes") or [])
        if isinstance(i, int) and i in self._ref_map
    })
    return cite_token(nums) if nums else ""

  def blocks(self, blocks: Any) -> list[str]:
    out = []
    for b in blocks or []:
      if isinstance(b, dict):
        md = self.block(b)
        if md:
          out.append(md)
    return out

  def block(self, b: dict) -> str:
    """Renders one block (without image slots); returns "" to skip it."""
    kind = b.get("type")
    snippet = _text(b.get("snippet"))
    if kind == "heading":
      return f"### {_one_line(snippet)}" if snippet else ""
    if kind == "paragraph":
      return snippet + self.cite(b) if snippet else ""
    if kind == "list":
      return self.list_items(b.get("list"), 0)
    if kind == "table":
      return _table(b.get("table"))
    if kind == "code_block":
      return _code(b.get("language"), b.get("code"))
    if kind in ("expandable", "comparison"):
      return self.container(b)
    if kind == "chart_block":
      return ""
    return snippet + self.cite(b) if snippet else ""

  def container(self, b: dict) -> str:
    """Renders expandable / comparison blocks (title + nested blocks)."""
    parts = []
    title = _one_line(b.get("title"))
    if title:
      parts.append(f"**{title}**")
    nested = b.get("text_blocks")
    snippet = _text(b.get("snippet"))
    if snippet and not nested:
      parts.append(snippet + self.cite(b))
    labels = b.get("product_labels")
    rows = b.get("comparison")
    if isinstance(labels, list) and isinstance(rows, list):
      table = [["Feature"] + [str(x) for x in labels]]
      for row in rows:
        if isinstance(row, dict):
          table.append([row.get("feature", "")] + list(row.get("values") or []))
      if len(table) > 1:
        parts.append(_table(table))
    parts.extend(self.blocks(nested))
    return "\n\n".join(parts)

  def list_items(self, items: Any, depth: int) -> str:
    lines = []
    pad = "  " * depth
    for item in items or []:
      if isinstance(item, str):
        if item.strip():
          lines.append(f"{pad}- {_one_line(item)}")
        continue
      if not isinstance(item, dict):
        continue
      title = _one_line(item.get("title"))
      snippet = _one_line(item.get("snippet"))
      head = " ".join(x for x in (f"**{title}**" if title else "", snippet) if x)
      thumb = _item_thumbnail(item)
      if thumb:
        self._thumbs.append((thumb, title))
      child = depth + 1
      if head:
        lines.append(f"{pad}- {head}{self.cite(item)}")
      else:
        child = depth
      sub = self.list_items(item.get("list"), child)
      if sub:
        lines.append(sub)
      code = item.get("code_block")
      if isinstance(code, dict):
        fenced = _code(code.get("language"), code.get("code"))
        if fenced:
          lines.append(_indent(fenced, child))
      body = "\n\n".join(self.blocks(item.get("text_blocks")))
      if body:
        lines.append(_indent(body, child))
    return "\n".join(lines)


@dataclasses.dataclass
class _Entry:
  markdown: str
  thumbs: list[tuple[str, str]]
  anchor: bool = False  # Receives one inline image.
  images: list[tuple[str, str]] = dataclasses.field(default_factory=list)


def _inline_images(resp: dict) -> list[str]:
  urls = (_image_url(x) for x in (resp.get("inline_images") or []))
  return list(dict.fromkeys(u for u in urls if u))


def _related(resp: dict) -> list[str]:
  out = []
  for q in resp.get("related_questions") or []:
    text = _one_line(q.get("question") if isinstance(q, dict) else q)
    if text and text not in out:
      out.append(text)
  return out


def _entries_from_blocks(blocks: list[dict], ref_map: dict[int, int]):
  renderer = _BlockRenderer(ref_map)
  # The intro paragraph and the first paragraph of every heading section each
  # receive one inline image (the first inline image is usually the hero).
  waiting = True  # Waiting for the section's first paragraph.
  entries: list[_Entry] = []
  for b in blocks:
    md = renderer.block(b)
    thumbs = renderer.take_thumbs()
    if not md and not thumbs:
      continue
    entries.append(_Entry(markdown=md, thumbs=thumbs))
    kind = b.get("type")
    if kind == "heading":
      waiting = True
    elif kind == "paragraph" and waiting:
      entries[-1].anchor = True
      waiting = False
  return entries


def _entries_from_markdown(markdown: str) -> list[_Entry]:
  """Splits fallback markdown so inline images go after the 1st paragraph."""
  pieces = re.split(r"(?<=\n\n)", markdown.strip())
  for i, piece in enumerate(pieces):
    stripped = piece.strip()
    if stripped and not stripped.startswith(("#", "|", "```", "- ", "* ")):
      head = "".join(pieces[: i + 1]).rstrip("\n")
      tail = "".join(pieces[i + 1:]).strip("\n")
      entries = [_Entry(markdown=head, thumbs=[], anchor=True)]
      if tail:
        entries.append(_Entry(markdown=tail, thumbs=[]))
      return entries
  return [_Entry(markdown=markdown.strip(), thumbs=[], anchor=True)]


def render(resp: dict, max_images: int = 6) -> Rendered:
  """Renders a google_ai_mode response (see module docstring).

  Args:
    resp: SerpApi JSON response.
    max_images: Maximum number of image slots.

  Returns:
    The rendered answer.
  """
  sources, ref_map = _sources(resp)
  blocks = [b for b in (resp.get("text_blocks") or []) if isinstance(b, dict)]
  fallback: Optional[str] = None
  if blocks:
    entries = _entries_from_blocks(blocks, ref_map)
  else:
    fallback = _text(resp.get("reconstructed_markdown"))
    entries = _entries_from_markdown(fallback) if fallback else []

  used: set[str] = set()
  budget = max(0, max_images)
  # 1) List-item thumbnails, right after their list.
  for e in entries:
    for url, caption in e.thumbs:
      if budget and url not in used:
        used.add(url)
        e.images.append((url, caption))
        budget -= 1
  # 2) Inline images: one per anchor paragraph (all of them after the first
  #    paragraph for reconstructed_markdown).
  queue = [u for u in _inline_images(resp) if u not in used]
  for e in entries:
    if not e.anchor:
      continue
    take = len(queue) if fallback is not None else 1
    while take and queue and budget:
      url = queue.pop(0)
      used.add(url)
      e.images.append((url, ""))
      budget -= 1
      take -= 1

  out: list[str] = []
  slots: list[dict] = []
  for e in entries:
    if e.markdown:
      if out and not out[-1].endswith("\n\n"):
        out.append("\n\n")
      out.append(e.markdown)
    for url, caption in e.images:
      slot_id = f"a{len(slots) + 1}"
      slots.append({"id": slot_id, "image_url": url, "caption": caption})
      out.append(slot_token(slot_id))
  return Rendered(markdown="".join(out), sources=sources, slots=slots,
                  related=_related(resp))
