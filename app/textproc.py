"""Pure text-processing helpers: streaming image-slot parsing and citations.

These have no network dependencies so they are easy to unit test.
"""

import dataclasses
import html
import re
from typing import Iterable, Iterator, Sequence, Union

SLOT_OPEN = "[[IMAGE:"
SLOT_CLOSE = "]]"
# If an opening "[[" is not closed within this many characters, treat it as
# ordinary text (the model wrote something else).
_MAX_MARKER_LEN = 400


@dataclasses.dataclass
class TextEvent:
  text: str


@dataclasses.dataclass
class SlotEvent:
  query: str
  caption: str


ParserEvent = Union[TextEvent, SlotEvent]


def parse_slot_body(body: str) -> SlotEvent:
  """Parses 'query | caption' (caption optional) from inside a marker."""
  if "|" in body:
    query, caption = body.split("|", 1)
  else:
    query, caption = body, ""
  query = query.strip().strip("\"'")
  caption = caption.strip().strip("\"'")
  return SlotEvent(query=query, caption=caption or query)


class SlotMarkerParser:
  """Incrementally splits streamed text into text and image-slot events.

  The model is instructed to write markers like
  ``[[IMAGE: Eiffel Tower at night | The tower lit up after sunset]]``.
  Because text arrives in arbitrary chunks, a marker may be split across
  chunks; this parser holds back just enough text to never emit a partial
  marker, while streaming everything else immediately.
  """

  def __init__(self):
    self._buf = ""

  def feed(self, chunk: str) -> Iterator[ParserEvent]:
    self._buf += chunk
    yield from self._drain(final=False)

  def flush(self) -> Iterator[ParserEvent]:
    yield from self._drain(final=True)

  def _drain(self, final: bool) -> Iterator[ParserEvent]:
    while self._buf:
      start = self._buf.find("[[")
      if start == -1:
        # Hold back a trailing "[" which may begin a marker in the next chunk.
        if not final and self._buf.endswith("["):
          emit, self._buf = self._buf[:-1], "["
        else:
          emit, self._buf = self._buf, ""
        if emit:
          yield TextEvent(emit)
        return
      if start > 0:
        yield TextEvent(self._buf[:start])
        self._buf = self._buf[start:]
      # Buffer now starts with "[[".
      head = self._buf[: len(SLOT_OPEN)]
      if not SLOT_OPEN.startswith(head.upper()) and not head.upper().startswith(
          SLOT_OPEN
      ):
        # "[[" but not an IMAGE marker: emit "[[" as text and keep scanning.
        yield TextEvent("[[")
        self._buf = self._buf[2:]
        continue
      if len(self._buf) < len(SLOT_OPEN):
        if final:
          yield TextEvent(self._buf)
          self._buf = ""
        return  # Need more data to decide.
      end = self._buf.find(SLOT_CLOSE, len(SLOT_OPEN))
      if end == -1:
        if final or len(self._buf) > _MAX_MARKER_LEN:
          yield TextEvent(self._buf[:2])
          self._buf = self._buf[2:]
          continue
        return  # Wait for the closing brackets.
      body = self._buf[len(SLOT_OPEN) : end]
      self._buf = self._buf[end + len(SLOT_CLOSE) :]
      slot = parse_slot_body(body)
      if slot.query:
        yield slot


def strip_slot_markers(text: str) -> str:
  return re.sub(r"\[\[IMAGE:[^\]]*\]\]", "", text, flags=re.IGNORECASE)


def slot_token(slot_id: str) -> str:
  """Placeholder token embedded in markdown where an image slot belongs."""
  return f"\n\n@@SLOT:{slot_id}@@\n\n"


def cite_token(numbers: Sequence[int]) -> str:
  return "@@CITE:" + ",".join(str(n) for n in numbers) + "@@"


@dataclasses.dataclass
class Support:
  text: str
  chunk_indices: list[int]


def insert_citations(markdown: str, supports: Iterable[Support]) -> str:
  """Inserts @@CITE:n,m@@ tokens after each grounded text segment.

  Grounding supports carry the exact segment text, which we locate in the
  assembled markdown (whose offsets differ from the raw model output because
  image markers were replaced by slot tokens). Segments are searched in order
  so repeated phrases attach to the right occurrence.
  """
  insertions: dict[int, list[int]] = {}
  cursor = 0
  for sup in supports:
    seg = (sup.text or "").strip()
    if not seg or not sup.chunk_indices:
      continue
    pos = markdown.find(seg, cursor)
    if pos == -1:
      pos = markdown.find(seg)
    if pos == -1:
      continue
    end = pos + len(seg)
    # Don't split a word / markdown emphasis: extend to just after closing
    # punctuation or emphasis markers that immediately follow.
    while end < len(markdown) and markdown[end] in "*_.,;:!?)\u201d\"'`":
      end += 1
    cursor = end
    # Citation chips look wrong inside headings; skip those segments.
    line_start = markdown.rfind("\n", 0, pos) + 1
    if markdown[line_start:pos].lstrip().startswith("#"):
      continue
    nums = insertions.setdefault(end, [])
    for idx in sup.chunk_indices:
      n = idx + 1
      if n not in nums:
        nums.append(n)
  if not insertions:
    return markdown
  out = []
  last = 0
  for pos in sorted(insertions):
    out.append(markdown[last:pos])
    out.append(cite_token(sorted(insertions[pos])))
    last = pos
  out.append(markdown[last:])
  return "".join(out)


_TAG_RE = re.compile(r"<[^>]+>")


def clean_title(title: str | None) -> str:
  """Strips HTML tags/entities that search grounding puts in titles."""
  if not title:
    return ""
  return html.unescape(_TAG_RE.sub("", title)).strip()
