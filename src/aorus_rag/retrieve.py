"""Retrieval: dense + BM25 + RRF, with a key-match boost and doc diversity.

Everything here is a deliberate choice rather than a framework default, and
each one is switchable from the CLI so it can be ablated in the benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .chunk import Chunk
from .embed import Embedder
from .index import BM25Index, IndexBundle, VectorIndex, rrf_fuse, tokenize

MODES = ("dense", "bm25", "hybrid")

# Chunks are three-level; when several chunks of the same spec row survive, we
# prefer the precise ones. Lower sorts first.
KIND_PRIORITY = {"fact": 0, "spec_line": 1, "spec_row": 2, "feature": 3, "footnote": 4}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    dense_rank: int | None = None
    bm25_rank: int | None = None


def _normalise_key(text: str) -> str:
    return re.sub(r"[\s/()（）]+", "", text).lower()


class Retriever:
    """Owns the corpus, both indexes, and the fusion policy."""

    def __init__(
        self,
        chunks: list[Chunk],
        bundle: IndexBundle | None,
        embedder: Embedder | None,
        mode: str = "hybrid",
        key_boost: float = 0.35,
        max_per_doc: int = 2,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.chunks = chunks
        self.mode = mode
        self.key_boost = key_boost
        self.max_per_doc = max_per_doc
        self.embedder = embedder

        self.bm25 = BM25Index([tokenize(c.text) for c in chunks])
        self.vector: VectorIndex | None = None
        if bundle is not None:
            if len(bundle.chunk_ids) != len(chunks):
                raise ValueError("index/corpus mismatch: rebuild with `uv run aorus-rag build`")
            self.vector = VectorIndex(bundle.embeddings)

        self._keys = [(_normalise_key(c.key_zh), _normalise_key(c.key_en)) for c in chunks]

    # ----------------------------------------------------------------

    def _dense(self, query: str, pool: int) -> list[tuple[int, float]]:
        if self.vector is None or self.embedder is None:
            return []
        vec = self.embedder.encode([query], is_query=True)[0]
        return self.vector.search(vec, top_k=pool)

    def _apply_key_boost(
        self, fused: list[tuple[int, float]], query: str
    ) -> list[tuple[int, float]]:
        """Nudge chunks whose key literally appears in the question.

        "螢幕更新率是多少" contains "螢幕更新率"; matching that exactly is a much
        stronger signal than any similarity score, and it costs one substring
        test per candidate.
        """
        if self.key_boost <= 0:
            return fused
        q = _normalise_key(query)
        boosted = []
        for idx, score in fused:
            key_zh, key_en = self._keys[idx]
            hit = (key_zh and len(key_zh) >= 2 and key_zh in q) or (
                key_en and len(key_en) >= 3 and key_en in q
            )
            boosted.append((idx, score * (1 + self.key_boost) if hit else score))
        return sorted(boosted, key=lambda kv: -kv[1])

    def _rescue_top_hits(
        self,
        selected: list[tuple[int, float]],
        rankings: list[list[tuple[int, float]]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Make sure each retriever's own #1 is somewhere in the final set.

        RRF scores consensus, so a chunk that only one retriever finds earns
        points from one ranking and nothing from the other. Usually correct --
        but it loses the case the hybrid exists for. Measured: for "螢幕支援 HDR
        嗎？是哪一個等級？" the row holding "VESA DisplayHDR True Black 500" is
        dense's #1 and absent from BM25's list, so RRF dropped it to rank 8 and
        the answer became a refusal.

        The rescue appends rather than promotes. An earlier version put these
        hits at the front and cost 2.8 points of Recall@1, because BM25's top
        hit is often a weak match that RRF was right to rank below consensus
        results. Displacing the tail keeps RRF's ordering intact and only
        changes whether a hit is present at all.
        """
        present = {i for i, _ in selected}
        missing = [r[0][0] for r in rankings if r and r[0][0] not in present]
        if not missing:
            return selected
        floor = selected[-1][1] if selected else 0.0
        keep = selected[: max(0, top_k - len(missing))]
        return keep + [(i, floor * 0.99) for i in dict.fromkeys(missing)][: top_k - len(keep)]

    def _diversify(self, ranked: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        """Cap chunks per source row so the context is not five views of one row."""
        per_doc: dict[str, int] = {}
        out: list[tuple[int, float]] = []
        for idx, score in ranked:
            doc = self.chunks[idx].doc_id
            if per_doc.get(doc, 0) >= self.max_per_doc:
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            out.append((idx, score))
            if len(out) >= top_k:
                break
        return out

    # ----------------------------------------------------------------

    def search(self, query: str, top_k: int = 4, pool: int = 25) -> list[Hit]:
        dense = self._dense(query, pool) if self.mode in ("dense", "hybrid") else []
        sparse = self.bm25.search(query, top_k=pool) if self.mode in ("bm25", "hybrid") else []

        if self.mode == "dense":
            fused = [(i, s) for i, s in dense]
        elif self.mode == "bm25":
            fused = [(i, s) for i, s in sparse]
        else:
            if not dense:  # no embedder available -> degrade to BM25 rather than fail
                fused = [(i, s) for i, s in sparse]
            else:
                fused = rrf_fuse([dense, sparse], k=60, weights=[1.0, 1.0])

        fused = self._apply_key_boost(fused, query)
        # Stable tie-break towards the more precise chunk kinds.
        fused.sort(key=lambda kv: (-kv[1], KIND_PRIORITY.get(self.chunks[kv[0]].kind, 9)))
        selected = self._diversify(fused, top_k)
        selected = self._rescue_top_hits(selected, [dense, sparse], top_k)

        dense_rank = {i: r for r, (i, _) in enumerate(dense)}
        bm25_rank = {i: r for r, (i, _) in enumerate(sparse)}
        return [
            Hit(
                chunk=self.chunks[idx],
                score=score,
                rank=rank,
                dense_rank=dense_rank.get(idx),
                bm25_rank=bm25_rank.get(idx),
            )
            for rank, (idx, score) in enumerate(selected)
        ]


def embed_corpus(chunks: list[Chunk], embedder: Embedder) -> IndexBundle:
    """Encode every chunk once, offline. Batched -- see index.search_batch."""
    texts = [c.text for c in chunks]
    matrix = embedder.encode(texts, is_query=False)
    matrix = np.asarray(matrix, dtype=np.float32)
    return IndexBundle(
        chunk_ids=[c.chunk_id for c in chunks],
        embeddings=matrix,
        embed_model=embedder.name,
        dim=int(matrix.shape[1]) if matrix.size else 0,
    )
