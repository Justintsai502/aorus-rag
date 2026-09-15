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
from html import unescape as html_unescape
from html.parser import HTMLParser
from typing import ClassVar

# Footnotes are marked on the page by <p style="font-size:80%">.
FOOTNOTE_STYLE = re.compile(r"font-size:\s*80%", re.IGNORECASE)
# Space, tab and the fullwidth ideographic space all count as whitespace.
_WS = re.compile(r"[ \t　]+")


def _clean(text: str) -> str:
    """Collapse whitespace but keep meaningful characters intact."""
    # \xa0 is &nbsp;. Convert it to a plain space, then collapse runs of whitespace.
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


# --------------------------------------------------------------------------
# Spec table
# --------------------------------------------------------------------------


@dataclass
class SpecItem:
    """One row of the spec table."""

    # One row = one spec field, e.g. Display.
    #   index      row position (0-based)
    #   key        field name: "顯示器" on the zh page, "Display" on the en page
    #   lines      the value, one entry per <br>-separated line
    #   footnotes  caveats for this row, kept apart from the value
    index: int
    key: str
    lines: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)

    @property
    def value(self) -> str:
        # Joined form for whole-value matching.
        return "\n".join(self.lines)


class SpecTableParser(HTMLParser):
    """Extract ``li.spec-title`` / ``li.spec-desc`` pairs.

    ``<br>`` becomes a line break, ``<p style="font-size:80%">`` is treated as
    a footnote rather than a spec value, and link targets are appended so a
    URL mentioned in the page survives into the corpus.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # HTMLParser is event-driven: handle_starttag, handle_data and handle_endtag fire
        # as the document is read, and it does not track position for you. State:
        #   _in_list      inside <ul class="spec-item-list">
        #   _mode         reading a title (field name) or a desc (value)
        #   _buf          text fragments not yet flushed into a line
        #   _current      the row being assembled
        #   _in_footnote  current text belongs to a footnote
        #   _href         target of the current <a>
        #   _skip_depth   > 0 while inside <script> / <style>
        self.items: list[SpecItem] = []
        self._in_list = False
        self._mode: str | None = None  # "title" | "desc" | None
        self._buf: list[str] = []
        self._current: SpecItem | None = None
        self._in_footnote = False
        self._href: str | None = None
        self._skip_depth = 0

    # -- helpers ---------------------------------------------------------

    # Turn the buffered fragments into one line and route it to key, footnotes or
    # lines depending on state. Called at every line boundary (<li>, <br>, <p>).
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
        # attrs arrives as [(name, value), ...]; value is None for bare attributes, hence
        # `v or ""`. The class attribute is a space-separated list, so split it.
        classes = attr.get("class", "").split()

        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        # Only <ul class="spec-item-list"> is accepted -- these are the AM6H's 17 rows.
        # The same page also has 51 <div class="spec-item-list"> blocks (17 rows x 3
        # SKUs) holding BZH / BYH / BXH values with no field names.
        if tag == "ul" and "spec-item-list" in classes:
            self._in_list = True
            # Each <ul> is one row: start an empty item and fill it in as tags arrive.
            self._current = SpecItem(index=len(self.items), key="")
            return

        # Outside the spec table none of the tags below are relevant.
        if not self._in_list:
            return

        # <li class="spec-title"> holds the field name, <li class="spec-desc"> the value.
        # Flush the previous text before switching.
        if tag == "li":
            self._flush()
            if "spec-title" in classes:
                self._mode = "title"
            elif "spec-desc" in classes:
                self._mode = "desc"
            else:
                self._mode = None
        # <br> is a line break inside a value; each line later becomes a spec_line chunk.
        elif tag == "br":
            self._flush()
        elif tag == "p":
            self._flush()
            # A <p> styled font-size:80% starts a footnote; any other <p> is ordinary value text.
            self._in_footnote = bool(FOOTNOTE_STYLE.search(attr.get("style", "")))
        # Remember the link target; handle_endtag appends it to the text on </a>.
        elif tag == "a":
            self._href = attr.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or not self._in_list:
            return

        # Append the link target if the anchor text does not already contain it, so URLs
        # survive parsing.
        if tag == "a" and self._href:
            joined = "".join(self._buf)
            if self._href not in joined:
                self._buf.append(f" ({self._href})")
            self._href = None
        elif tag == "p":
            self._flush()
            self._in_footnote = False
        # </li> closes a title or desc cell: flush its text and leave title/desc mode.
        elif tag == "li":
            self._flush()
            self._mode = None
        # </ul> closes a row. Keep it only if it has a key, then reset for the next row.
        elif tag == "ul":
            self._flush()
            if self._current is not None and self._current.key:
                self.items.append(self._current)
            self._current = None
            self._in_list = False
            self._mode = None
            self._in_footnote = False

    def handle_data(self, data: str) -> None:
        # Only collect text inside the spec table and inside a title / desc <li>.
        if self._skip_depth or not self._in_list or self._mode is None:
            return
        self._buf.append(data)


def parse_spec_table(html: str) -> list[SpecItem]:
    # feed() consumes the whole document; close() flushes whatever is still buffered.
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

    # block_id    the section's id or section-* class; "root" for text placed
    #             directly in the features container
    # headings    h1-h6 text in document order
    # paragraphs  <p> text in document order
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

    # Only headings (h1-h6) and paragraphs are collected.
    _TEXT_TAGS: ClassVar[set[str]] = {"h1", "h2", "h3", "h4", "h5", "h6", "p"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[FeatureBlock] = []
        self._depth = 0  # depth inside the features container, 0 = outside
        # Sections nest; text is attributed to the block on top of the stack.
        self._section_stack: list[FeatureBlock] = []
        self._text_tag: str | None = None
        self._buf: list[str] = []
        self._skip_depth = 0

    # The innermost open block, or None outside the features container.
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

        # Ignore everything until the key-features container starts.
        if self._depth == 0:
            if "key-features" in classes:
                self._depth = 1
                # Text directly inside the container (not in a nested section) belongs to "root".
                self._section_stack.append(FeatureBlock(block_id="root"))
            return

        # Inside the features container: track nesting so we know when to stop.
        # New section: go one level deeper and open a block named after its id or its
        # section-* class.
        if tag == "section" or (tag == "div" and "section-" in classes):
            self._depth += 1
            label = attr.get("id") or next(
                (c for c in classes.split() if c.startswith("section-")), f"block{len(self.blocks)}"
            )
            self._section_stack.append(FeatureBlock(block_id=label))
        # Start capturing a heading or paragraph; the buffer is reset for the new element.
        elif tag in self._TEXT_TAGS:
            self._text_tag = tag
            self._buf.clear()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "svg", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or self._depth == 0:
            return

        # A heading or paragraph closed. Headings and paragraphs are stored separately
        # because chunk_features uses a block's first heading as the anchor for its prose.
        if tag in self._TEXT_TAGS and self._text_tag == tag:
            text = _clean("".join(self._buf))
            block = self._current_block()
            if text and block is not None:
                (block.headings if tag.startswith("h") else block.paragraphs).append(text)
            self._buf.clear()
            self._text_tag = None
        # Block ends: pop it and keep it if non-empty. A </section> at depth 1 closes the
        # whole features container.
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

# A different row count means the page layout changed.
EXPECTED_SPEC_ROWS = 17

# Values that must never appear: they belong to the sibling models shown in
# the desktop comparison widget.
SIBLING_MARKERS = ("AORUS MASTER 16 BZH", "AORUS MASTER 16 BYH", "AORUS MASTER 16 BXH")


# Three checks: exactly 17 rows; every row has a key and a value; no value
# mentions a sibling model.
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


# --------------------------------------------------------------------------
# SKU variants
#
# AM6H is the model page; BZH / BYH / BXH are its sellable SKUs (their own URLs
# 302 to this page). The desktop comparison widget carries one column per SKU in
# `div.spec-item-list[data-spec-row=N]` blocks that hold values but no labels --
# the SKU names live separately in `div.model-base-info-subtitle`, in the same
# order as the columns.
#
# `parse_spec_table` deliberately excludes those blocks, because merging
# unlabelled values would put three contradictory answers behind one question.
# This parser does the opposite and the only safe thing: it pairs each value
# back with the SKU it belongs to, so the information can be added to the corpus
# *bound to its model code* rather than floating free.
# --------------------------------------------------------------------------

# The comparison widget's markup is regular enough for regexes:
#   _SUBTITLE  SKU names (BZH, BYH, BXH) in column order
#   _SKU_ROW   each <div data-spec-row="N"> is one SKU's value for spec row N
#   _SKU_CODE  the short code from the full name, e.g. BZH
_SUBTITLE = re.compile(r'<div class="model-base-info-subtitle">(.*?)</div>', re.DOTALL)
_SKU_ROW = re.compile(r'<div class="spec-item-list" data-spec-row="(\d+)">(.*?)</div>', re.DOTALL)
_SKU_CODE = re.compile(r"AORUS MASTER 16 ([A-Z0-9]+)")


@dataclass
class SkuVariant:
    """One purchasable configuration and the spec rows where it differs."""

    code: str  # "BZH"
    full_name: str  # "AORUS MASTER 16 BZH"
    values: dict[int, str] = field(default_factory=dict)  # row index -> value


# One cell of HTML -> plain text: <br> to newline, strip tags, unescape entities,
# then join the lines with "; ".
def _sku_cell_text(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [_clean(line) for line in html_unescape(text).split("\n")]
    return "; ".join(line for line in lines if line)


def parse_sku_variants(html: str) -> tuple[list[SkuVariant], list[int]]:
    """Return the SKUs and the indices of the spec rows that differ between them.

    Rows identical across every SKU are omitted: they are already covered by the
    main spec table, and repeating them per SKU would add near-duplicate chunks
    for no gain.
    """
    # No subtitle, or no SKU codes in it, means the comparison widget is absent:
    # return nothing rather than guess.
    subtitle = _SUBTITLE.search(html)
    if not subtitle:
        return ([], [])
    codes = _SKU_CODE.findall(html_unescape(subtitle.group(1)))
    if not codes:
        return ([], [])

    # columns[row] = [value for SKU 1, value for SKU 2, value for SKU 3]
    # findall returns matches in document order, so list order is column order,
    # which matches the SKU order in the subtitle.
    columns: dict[int, list[str]] = {}
    for idx, body in _SKU_ROW.findall(html):
        columns.setdefault(int(idx), []).append(_sku_cell_text(body))

    # Every row must hold exactly one value per SKU; otherwise values cannot be
    # paired with model codes safely.
    n = len(codes)
    if not columns or any(len(v) != n for v in columns.values()):
        # Layout changed; better to add nothing than to mispair a value.
        return ([], [])

    # Keep only rows whose values differ between SKUs (on this page: the GPU row).
    differing = sorted(i for i, vals in columns.items() if len(set(vals)) > 1)
    # One SkuVariant per SKU; ``values`` holds only the differing rows: {row: value}.
    variants = [
        SkuVariant(
            code=code,
            full_name=f"AORUS MASTER 16 {code}",
            values={i: columns[i][col] for i in differing},
        )
        for col, code in enumerate(codes)
    ]
    return (variants, differing)
