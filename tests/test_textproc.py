"""Unit tests for app.textproc (no network)."""

import unittest

from app.textproc import (
    SlotEvent,
    SlotMarkerParser,
    Support,
    TextEvent,
    clean_title,
    insert_citations,
    parse_slot_body,
    strip_slot_markers,
)


def run_parser(chunks):
  p = SlotMarkerParser()
  events = []
  for c in chunks:
    events.extend(p.feed(c))
  events.extend(p.flush())
  # Merge adjacent text events for easier assertions.
  merged = []
  for e in events:
    if isinstance(e, TextEvent) and merged and isinstance(merged[-1], TextEvent):
      merged[-1] = TextEvent(merged[-1].text + e.text)
    else:
      merged.append(e)
  return merged


class SlotMarkerParserTest(unittest.TestCase):

  def test_plain_text_passthrough(self):
    self.assertEqual(run_parser(["Hello ", "world"]), [TextEvent("Hello world")])

  def test_single_marker(self):
    ev = run_parser(["Intro.\n\n[[IMAGE: Eiffel Tower | The tower at night]]\n\nMore."])
    self.assertEqual(ev, [
        TextEvent("Intro.\n\n"),
        SlotEvent("Eiffel Tower", "The tower at night"),
        TextEvent("\n\nMore."),
    ])

  def test_marker_split_across_every_character(self):
    text = "A [[IMAGE: red panda | cute]] B"
    ev = run_parser(list(text))
    self.assertEqual(ev, [TextEvent("A "), SlotEvent("red panda", "cute"), TextEvent(" B")])

  def test_split_at_bracket_boundary(self):
    ev = run_parser(["See [", "[IMAGE: x | y]", "] end"])
    self.assertEqual(ev, [TextEvent("See "), SlotEvent("x", "y"), TextEvent(" end")])

  def test_non_marker_double_brackets_are_text(self):
    ev = run_parser(["array[[0]] and [[link]] ok"])
    self.assertEqual(ev, [TextEvent("array[[0]] and [[link]] ok")])

  def test_unclosed_marker_flushed_as_text(self):
    ev = run_parser(["tail [[IMAGE: never closed"])
    self.assertEqual(ev, [TextEvent("tail [[IMAGE: never closed")])

  def test_single_bracket_text(self):
    ev = run_parser(["a [b] c ["])
    self.assertEqual(ev, [TextEvent("a [b] c [")])

  def test_lowercase_marker(self):
    ev = run_parser(["[[image: cat]]"])
    self.assertEqual(ev, [SlotEvent("cat", "cat")])

  def test_text_is_streamed_before_marker_closes(self):
    p = SlotMarkerParser()
    out = list(p.feed("Paragraph one. [[IMAGE: q"))
    self.assertEqual(out, [TextEvent("Paragraph one. ")])
    out = list(p.feed(" | c]] after"))
    self.assertEqual(out, [SlotEvent("q", "c"), TextEvent(" after")])

  def test_empty_query_dropped(self):
    ev = run_parser(["x [[IMAGE:  | only caption]] y"])
    self.assertEqual(ev, [TextEvent("x  y")])


class HelpersTest(unittest.TestCase):

  def test_parse_slot_body(self):
    self.assertEqual(parse_slot_body(' "Q" | "C" '), SlotEvent("Q", "C"))
    self.assertEqual(parse_slot_body("only query"), SlotEvent("only query", "only query"))

  def test_strip_slot_markers(self):
    self.assertEqual(strip_slot_markers("a [[IMAGE: x | y]] b"), "a  b")

  def test_clean_title(self):
    self.assertEqual(clean_title("<b>Angel&#39;s</b> Landing"), "Angel's Landing")
    self.assertEqual(clean_title(None), "")


class CitationTest(unittest.TestCase):

  def test_inserts_after_segments(self):
    md = "Alpha is first. Beta is second."
    out = insert_citations(md, [
        Support("Alpha is first", [0]),
        Support("Beta is second", [1, 0]),
    ])
    self.assertEqual(out, "Alpha is first.@@CITE:1@@ Beta is second.@@CITE:1,2@@")

  def test_missing_segment_ignored(self):
    self.assertEqual(insert_citations("abc", [Support("zzz", [0])]), "abc")

  def test_merges_same_position(self):
    out = insert_citations("Fact.", [Support("Fact", [0]), Support("Fact", [2])])
    self.assertEqual(out, "Fact.@@CITE:1,3@@")

  def test_ordered_search_handles_repeats(self):
    md = "X happened. Later, X happened."
    out = insert_citations(md, [Support("X happened", [0]), Support("X happened", [1])])
    self.assertEqual(out, "X happened.@@CITE:1@@ Later, X happened.@@CITE:2@@")

  def test_does_not_break_bold(self):
    out = insert_citations("**Bold claim** here", [Support("Bold claim", [0])])
    self.assertEqual(out, "**Bold claim**@@CITE:1@@ here")

  def test_skips_headings(self):
    md = "### Heading text\n\nBody text."
    out = insert_citations(md, [Support("Heading text", [0]), Support("Body text", [1])])
    self.assertEqual(out, "### Heading text\n\nBody text.@@CITE:2@@")


if __name__ == "__main__":
  unittest.main()
