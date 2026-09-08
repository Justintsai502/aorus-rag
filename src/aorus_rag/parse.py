"""Turn the cached HTML into structured records, using only the stdlib.

Two extractors live here:

``SpecTableParser``
    The spec sheet is a clean key/value table. In the DOM it is
    ``ul.spec-item-list > li.spec-title + li.spec-desc``. Crucially the page
    ALSO contains a desktop comparison widget holding the specs of three
    sibling models (BZH / BYH / BXH) in ``div.spec-item-list`` blocks that
    carry values but no titles. Keying on the ``li`` variant scopes us to the
    AM6H column and avoids silently mixing in another laptop's specs -- the
    single most dangerous failure mode for this dataset, because the answers
    would look plausible and be wrong.

``FeatureSectionParser``
    The marketing page carries prose (cooling, GiMATE, I/O layout, ...) that
    the spec table does not. It is a secondary corpus for "how/why" questions.

No BeautifulSoup, lxml or readability: ``html.parser`` from the standard
library is enough and keeps the dependency surface at numpy + httpx.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import ClassVar

FOOTNOTE_STYLE = re.compile(r"font-size:\s*80%", re.IGNORECASE)
_WS = re.compile(r"[ \t　]+")


def _clean(text: str) -> str:
    """Collapse whitespace but keep meaningful characters intact."""
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


# --------------------------------------------------------------------------
# Spec table
# --------------------------------------------------------------------------


@dataclass
class SpecItem:
    """One row of the spec table."""

    index: int
    key: str
    lines: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)

    @property
    def value(self) -> str:
        return "\n".join(self.lines)


class SpecTableParser(HTMLParser):
    """Extract ``li.spec-title`` / ``li.spec-desc`` pairs.

    ``<br>`` becomes a line break, ``<p style="font-size:80%">`` is treated as
    a footnote rather than a spec value, and link targets are appended so a
    URL mentioned in the page survives into the corpus.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[SpecItem] = []
        self._in_list = False
        self._mode: str | None = None  # "title" | "desc" | None
        self._buf: list[str] = []
        self._current: SpecItem | None = None
        self._in_footnote = False
        self._href: str | None = None
        self._skip_depth = 0

    # -- helpers ---------------------------------------------------------

    def _flush(self) -> None:
        text = _clean("".join(self._buf))
        self._buf.clear()
        if not text or self._current is None:
            return
        if self._mode == "title":
            self._current.key = text
        elif self._in_footnote:
            self._current.footnotes.append(text)
        else:
            self._current.lines.append(text)

    # -- HTMLParser hooks ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        classes = attr.get("class", "").split()

        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "ul" and "spec-item-list" in classes:
            self._in_list = True
            self._current = SpecItem(index=len(self.items), key="")
            return

        if not self._in_list:
            return

        if tag == "li":
            self._flush()
            if "spec-title" in classes:
                self._mode = "title"
            elif "spec-desc" in classes:
                self._mode = "desc"
            else:
                self._mode = None
        elif tag == "br":
            self._flush()
        elif tag == "p":
            self._flush()
            self._in_footnote = bool(FOOTNOTE_STYLE.search(attr.get("style", "")))
        elif tag == "a":
            self._href = attr.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or not self._in_list:
            return

        if tag == "a" and self._href:
            joined = "".join(self._buf)
            if self._href not in joined:
                self._buf.append(f" ({self._href})")
            self._href = None
        elif tag == "p":
            self._flush()
            self._in_footnote = False
        elif tag == "li":
            self._flush()
            self._mode = None
        elif tag == "ul":
            self._flush()
            if self._current is not None and self._current.key:
                self.items.append(self._current)
            self._current = None
            self._in_list = False
            self._mode = None
            self._in_footnote = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._in_list or self._mode is None:
            return
        self._buf.append(data)


def parse_spec_table(html: str) -> list[SpecItem]:
    parser = SpecTableParser()
    parser.feed(html)
    parser.close()
    return parser.items


# --------------------------------------------------------------------------
# Feature page
# --------------------------------------------------------------------------


@dataclass
class FeatureBlock:
    """A prose section from the marketing page."""

    block_id: str
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.headings and not self.paragraphs


class FeatureSectionParser(HTMLParser):
    """Collect headings and paragraphs inside ``.key-features-section``.

    Scoping to that container is what keeps the site-wide navigation menu
    (hundreds of product links) out of the corpus.
    """

    _TEXT_TAGS: ClassVar[set[str]] = {"h1", "h2", "h3", "h4", "h5", "h6", "p"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[FeatureBlock] = []
        self._depth = 0  # depth inside the features container, 0 = outside
        self._section_stack: list[FeatureBlock] = []
        self._text_tag: str | None = None
        self._buf: list[str] = []
        self._skip_depth = 0

    def _current_block(self) -> FeatureBlock | None:
        return self._section_stack[-1] if self._section_stack else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        classes = attr.get("class", "")

        if tag in ("script", "style", "svg", "noscript"):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if self._depth == 0:
            if "key-features" in classes:
                self._depth = 1
                self._section_stack.append(FeatureBlock(block_id="root"))
            return

        # Inside the features container: track nesting so we know when to stop.
        if tag == "section" or (tag == "div" and "section-" in classes):
            self._depth += 1
            label = attr.get("id") or next(
                (c for c in classes.split() if c.startswith("section-")), f"block{len(self.blocks)}"
            )
            self._section_stack.append(FeatureBlock(block_id=label))
        elif tag in self._TEXT_TAGS:
            self._text_tag = tag
            self._buf.clear()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "svg", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or self._depth == 0:
            return

        if tag in self._TEXT_TAGS and self._text_tag == tag:
            text = _clean("".join(self._buf))
            block = self._current_block()
            if text and block is not None:
                (block.headings if tag.startswith("h") else block.paragraphs).append(text)
            self._buf.clear()
            self._text_tag = None
        elif tag == "section" or tag == "div":
            if len(self._section_stack) > 1 and self._depth > 1:
                block = self._section_stack.pop()
                self._depth -= 1
                if not block.is_empty():
                    self.blocks.append(block)
            elif self._depth == 1 and tag == "section":
                block = self._section_stack.pop() if self._section_stack else None
                if block is not None and not block.is_empty():
                    self.blocks.append(block)
                self._depth = 0

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._depth == 0 or self._text_tag is None:
            return
        self._buf.append(data)


def parse_feature_page(html: str) -> list[FeatureBlock]:
    parser = FeatureSectionParser()
    parser.feed(html)
    parser.close()
    # Drop navigation-ish leftovers: real feature copy has a heading or a
    # sentence, not a bare two-word link label.
    return [
        b
        for b in parser.blocks
        if any(len(p) > 20 for p in b.paragraphs) or any(len(h) > 4 for h in b.headings)
    ]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

EXPECTED_SPEC_ROWS = 17

# Values that must never appear: they belong to the sibling models shown in
# the desktop comparison widget.
SIBLING_MARKERS = ("AORUS MASTER 16 BZH", "AORUS MASTER 16 BYH", "AORUS MASTER 16 BXH")


def validate_spec_items(items: list[SpecItem], label: str) -> None:
    """Fail loudly rather than shipping a silently wrong corpus."""
    if len(items) != EXPECTED_SPEC_ROWS:
        raise ValueError(
            f"{label}: expected {EXPECTED_SPEC_ROWS} spec rows, got {len(items)}. "
            "The page layout probably changed -- re-check parse.SpecTableParser."
        )
    for item in items:
        if not item.key:
            raise ValueError(f"{label}: row {item.index} has an empty key")
        if not item.lines:
            raise ValueError(f"{label}: row {item.index} ({item.key}) has no value")
        blob = item.value
        for marker in SIBLING_MARKERS:
            if marker in blob:
                raise ValueError(f"{label}: row {item.key} leaked sibling-model content ({marker})")
