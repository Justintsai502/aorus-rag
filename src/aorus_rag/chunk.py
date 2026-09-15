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
from .parse import FeatureBlock, SkuVariant, SpecItem

# Sentence-ish boundaries for both scripts.
# Split after zh / en sentence-ending punctuation, or at newlines.
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|\n+")
# "Left Side:" / "Right Side:" are structural markers inside the I/O row, not
# spec values. They are dropped from L1 chunks; the side information survives
# in the derived io.side.* facts instead.
_STRUCTURAL_LINE = re.compile(r"^(left|right)\s+side\s*:$", re.IGNORECASE)
# Windowing applies to marketing prose only; spec rows are chunked by structure.
# Consecutive windows overlap by 40 characters so a sentence cut at a boundary
# keeps some context.
MAX_FEATURE_CHARS = 240
FEATURE_OVERLAP_CHARS = 40


@dataclass
class Chunk:
    """A retrievable unit."""

    # Fields:
    #   chunk_id  unique id, e.g. spec.display#L3
    #   doc_id    source row; shared by every chunk cut from that row, and used by
    #             the retriever to cap chunks per row
    #   kind      spec_row / spec_line / fact / footnote / feature / sku
    #   lang      "bi" for bilingual chunks; feature chunks are "zh" or "en"
    #   meta      debugging info, not used for retrieval
    chunk_id: str
    doc_id: str
    kind: str
    text: str  # what gets embedded / indexed
    key_zh: str = ""
    key_en: str = ""
    lang: str = "bi"
    source: str = ""
    meta: dict = field(default_factory=dict)

    # Single hook for prompt rendering. The text already carries its bilingual key
    # anchor, so it is used as-is.
    def context_line(self) -> str:
        """How the chunk is rendered into the LLM prompt."""
        return self.text


# Build "zh key / en key: body"; write the key once if they match or one is missing.
def _anchor(key_zh: str, key_en: str, body: str) -> str:
    if key_zh and key_en and key_zh != key_en:
        return f"{key_zh} / {key_en}: {body}"
    return f"{key_zh or key_en}: {body}"


def _split_long(text: str, limit: int = MAX_FEATURE_CHARS) -> list[str]:
    """Sentence-aware windowing with a small overlap for continuity."""
    # Normalise whitespace; short text is returned as a single window.
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text]
    # Split into sentences, then pack them greedily into windows of at most `limit` chars.
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    windows: list[str] = []
    current = ""
    # Windows never cut a sentence: a single sentence longer than the limit becomes an
    # oversized window of its own.
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            windows.append(current)
            # Window full: emit it and start the next one with its last 40 characters.
            current = (current[-FEATURE_OVERLAP_CHARS:] + " " + sentence).strip()
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows


def chunk_spec_rows(zh_items: list[SpecItem], en_items: list[SpecItem]) -> list[Chunk]:
    """L0 (whole row) + L1 (per line) + footnotes."""
    chunks: list[Chunk] = []
    # zh and en rows are paired by position. Chunk text uses the zh row's lines (the
    # values are identical in both locales) together with both keys.
    for zh, en in zip(zh_items, en_items):
        # Derived from the English key, e.g. "I/O Port" -> spec.io_port
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
        # A single-line row would produce an L1 chunk identical to its L0 chunk.
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
            # First segment of the fact id, e.g. display.refresh_rate -> fact.display, so
            # related facts share the per-doc cap.
            doc_id=f"fact.{f.fact_id.split('.')[0]}",
            kind="fact",
            # e.g. "螢幕更新率 / Display refresh rate: 240Hz" -- one key, exactly one value.
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
    # Fallback anchor for prose whose own heading is missing or unreliable.
    product = "AORUS MASTER 16 AM6H"
    for block in blocks:
        # The marketing page nests loosely; a heading harvested from a large
        # catch-all block is more likely to mislabel a paragraph than to help.
        heading = (
            block.headings[0]
            if block.headings and block.block_id != "root" and len(block.paragraphs) <= 8
            else f"{product} 產品特色 / product feature"
        )
        # Each paragraph is windowed and prefixed with the block heading as its anchor.
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
                        # Feature chunks are monolingual: the heading fills only its own language's key.
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
        # Compare ignoring whitespace and case. build_corpus adds facts first, so the
        # occurrence kept is the most precise one.
        norm = " ".join(c.text.split()).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(c)
    return out


def write_corpus(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSONL, one chunk per line. ensure_ascii=False keeps CJK readable in the file
    # and in git diffs.
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_corpus(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `uv run aorus-rag build` to create the corpus."
        )
    with path.open(encoding="utf-8") as fh:
        # Each JSON object maps directly onto Chunk's fields.
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def chunk_sku_variants(
    variants: list[SkuVariant],
    differing_rows: list[int],
    zh_items: list[SpecItem],
    en_items: list[SpecItem],
) -> list[Chunk]:
    """One chunk per SKU per differing row, plus an overview.

    Every value is prefixed with the model code it belongs to. That prefix is
    the whole point: the same three values merged without labels would be three
    contradictory answers to "what GPU does this have?", which is exactly the
    failure `parse_spec_table` exists to prevent. Bound to a model code they
    become three answers to three different questions.
    """
    if not variants or not differing_rows:
        return []

    chunks: list[Chunk] = []
    # Field names of the differing rows in both languages, the SKU code list, and the
    # number of rows identical across all SKUs -- all used in the overview text.
    keys_zh = [zh_items[i].key for i in differing_rows]
    keys_en = [en_items[i].key for i in differing_rows]
    codes = "、".join(v.code for v in variants)
    same_count = len(zh_items) - len(differing_rows)

    chunks.append(
        Chunk(
            # Overview chunk: answers "how many models are there, and how do they differ?"
            chunk_id="sku.overview",
            doc_id="sku",
            kind="sku",
            text=(
                f"機型版本 / Model variants: AORUS MASTER 16 AM6H 共有 {len(variants)} 個型號"
                f"（{codes}），規格差異僅在「{'、'.join(keys_zh)}」"
                f"（{', '.join(keys_en)}），其餘 {same_count} 項規格三個型號完全相同。"
            ),
            key_zh="機型版本",
            key_en="Model variants",
            source=SOURCES["spec_zh"],
            meta={"skus": [v.code for v in variants], "differing_keys": keys_en},
        )
    )

    # Then one chunk per SKU per differing row, e.g.
    # "AORUS MASTER 16 BYH 的顯示晶片 / Video Graphics: ... RTX 5080 ..."
    for v in variants:
        for row in differing_rows:
            zh_key, en_key = zh_items[row].key, en_items[row].key
            chunks.append(
                Chunk(
                    chunk_id=f"sku.{v.code.lower()}.{row}",
                    doc_id=f"sku.{v.code.lower()}",
                    kind="sku",
                    text=f"{v.full_name} 的{zh_key} / {en_key}: {v.values[row]}",
                    # Prefixing the key with the SKU code lets a question such as "BYH 顯示晶片"
                    # trigger the retriever's key boost for exactly this chunk.
                    key_zh=f"{v.code} {zh_key}",
                    key_en=f"{v.code} {en_key}",
                    source=SOURCES["spec_zh"],
                    meta={"sku": v.code, "spec_key": en_key},
                )
            )
    return chunks
