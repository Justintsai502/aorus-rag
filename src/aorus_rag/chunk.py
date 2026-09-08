"""Key-anchored chunking.

The single most important decision in this project. A generic splitter
(``chunk_size=1000, overlap=200``, the usual default) would merge the whole
17-row spec sheet into three blobs, each holding a dozen unrelated numbers.
Asked "how big is the battery?", a 3B model handed such a blob has to choose
between 99Wh, 330W, 64GB, 24GB, 5.4GHz and 5600MHz -- and small models choose
wrong.

So chunks are built around the table's own structure, at three granularities:

``spec_row``   the whole row -- answers "what are the display specs?"
``spec_line``  one line of a row -- answers "how many Type-C ports?"
``fact``       one derived measurable -- answers "what's the refresh rate?"

plus ``feature`` chunks from the marketing prose for "how/why" questions.

Every chunk is prefixed with its bilingual key ("顯示器 / Display: ..."). Without
that anchor the embedding of a line like "1 x HDMI 2.1" carries almost no
signal about what question it answers, and a Chinese query has nothing
Chinese to match against -- the page's values are English in both locales.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SOURCES
from .normalize import Fact
from .parse import FeatureBlock, SpecItem

# Sentence-ish boundaries for both scripts.
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|\n+")
# "Left Side:" / "Right Side:" are structural markers inside the I/O row, not
# spec values. They are dropped from L1 chunks; the side information survives
# in the derived io.side.* facts instead.
_STRUCTURAL_LINE = re.compile(r"^(left|right)\s+side\s*:$", re.IGNORECASE)
MAX_FEATURE_CHARS = 240
FEATURE_OVERLAP_CHARS = 40


@dataclass
class Chunk:
    """A retrievable unit."""

    chunk_id: str
    doc_id: str
    kind: str
    text: str  # what gets embedded / indexed
    key_zh: str = ""
    key_en: str = ""
    lang: str = "bi"
    source: str = ""
    meta: dict = field(default_factory=dict)

    def context_line(self) -> str:
        """How the chunk is rendered into the LLM prompt."""
        return self.text


def _anchor(key_zh: str, key_en: str, body: str) -> str:
    if key_zh and key_en and key_zh != key_en:
        return f"{key_zh} / {key_en}: {body}"
    return f"{key_zh or key_en}: {body}"


def _split_long(text: str, limit: int = MAX_FEATURE_CHARS) -> list[str]:
    """Sentence-aware windowing with a small overlap for continuity."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            windows.append(current)
            current = (current[-FEATURE_OVERLAP_CHARS:] + " " + sentence).strip()
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows


def chunk_spec_rows(zh_items: list[SpecItem], en_items: list[SpecItem]) -> list[Chunk]:
    """L0 (whole row) + L1 (per line) + footnotes."""
    chunks: list[Chunk] = []
    for zh, en in zip(zh_items, en_items):
        doc_id = f"spec.{en.key.lower().replace(' ', '_').replace('/', '')}"
        doc_id = re.sub(r"[^a-z0-9_.]", "", doc_id)

        # L0 -- the full row.
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#row",
                doc_id=doc_id,
                kind="spec_row",
                text=_anchor(zh.key, en.key, "; ".join(zh.lines)),
                key_zh=zh.key,
                key_en=en.key,
                source=SOURCES["spec_zh"],
                meta={"line_count": len(zh.lines)},
            )
        )

        # L1 -- one chunk per line, still carrying the key anchor.
        if len(zh.lines) > 1:
            for i, line in enumerate(zh.lines):
                if _STRUCTURAL_LINE.match(line):
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#L{i}",
                        doc_id=doc_id,
                        kind="spec_line",
                        text=_anchor(zh.key, en.key, line),
                        key_zh=zh.key,
                        key_en=en.key,
                        source=SOURCES["spec_zh"],
                        meta={"line_index": i},
                    )
                )

        # Footnotes are caveats, not values: kept, but clearly marked so the
        # model does not quote a disclaimer as if it were a spec.
        for i, note in enumerate(zh.footnotes):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#note{i}",
                    doc_id=doc_id,
                    kind="footnote",
                    text=_anchor(zh.key, en.key, f"[註 / note] {note}"),
                    key_zh=zh.key,
                    key_en=en.key,
                    source=SOURCES["spec_zh"],
                    meta={"footnote_index": i},
                )
            )
    return chunks


def chunk_facts(facts: list[Fact]) -> list[Chunk]:
    """L2 -- one measurable per chunk, the highest-precision level."""
    return [
        Chunk(
            chunk_id=f"fact.{f.fact_id}",
            doc_id=f"fact.{f.fact_id.split('.')[0]}",
            kind="fact",
            text=_anchor(f.label_zh, f.label_en, f.value),
            key_zh=f.label_zh,
            key_en=f.label_en,
            source=SOURCES["spec_zh"],
            meta={"fact_id": f.fact_id, "spec_key": f.source_key_en, **f.extra},
        )
        for f in facts
    ]


def chunk_features(blocks: list[FeatureBlock], lang: str) -> list[Chunk]:
    """Marketing prose -- the "how/why" corpus the spec table cannot answer."""
    source = SOURCES["feature_zh" if lang == "zh" else "feature_en"]
    chunks: list[Chunk] = []
    product = "AORUS MASTER 16 AM6H"
    for block in blocks:
        # The marketing page nests loosely; a heading harvested from a large
        # catch-all block is more likely to mislabel a paragraph than to help.
        heading = (
            block.headings[0]
            if block.headings and block.block_id != "root" and len(block.paragraphs) <= 8
            else f"{product} 產品特色 / product feature"
        )
        for pi, para in enumerate(block.paragraphs):
            for wi, window in enumerate(_split_long(para)):
                if len(window) < 12:  # nav labels, button text
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"feature.{lang}.{block.block_id}.{pi}.{wi}",
                        doc_id=f"feature.{lang}.{block.block_id}",
                        kind="feature",
                        text=f"{heading}: {window}" if heading else window,
                        key_zh=heading if lang == "zh" else "",
                        key_en=heading if lang == "en" else "",
                        lang=lang,
                        source=source,
                        meta={"block": block.block_id, "heading": heading},
                    )
                )
    return chunks


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact text duplicates, keeping the first (most precise) occurrence."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        norm = " ".join(c.text.split()).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(c)
    return out


def write_corpus(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_corpus(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `uv run aorus-rag build` to create the corpus."
        )
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
