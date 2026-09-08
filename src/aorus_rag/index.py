"""Hand-written retrieval index: dense (numpy), sparse (BM25), and RRF fusion.

There is no vector database here on purpose. With ~150 chunks a normalised
float32 matrix and one ``@`` is the whole nearest-neighbour search -- an ANN
index would add a dependency, a build step and approximation error to solve a
problem that does not exist at this scale.

The sparse half matters more than it looks. Spec sheets are full of exact
strings ("Thunderbolt 5", "5600MHz", "RTX 5090") where lexical matching beats
semantics, and a small multilingual embedding model is weakest on precisely
those alphanumeric tokens. Dense retrieval covers paraphrase, BM25 covers
exact identifiers, and Reciprocal Rank Fusion combines them without needing
the two score scales to be comparable.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Latin/alphanumeric runs, keeping internal dots and hyphens so that
# "usb3.2", "802.11be" and "type-c" survive as single tokens.
_LATIN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# CJK ranges (plus fullwidth forms) for the bigram path.
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")


def tokenize(text: str) -> list[str]:
    """Mixed zh/en tokenizer.

    Chinese is split into character bigrams rather than words. For a corpus
    this small, bigrams beat dictionary segmentation: no dictionary to ship,
    no out-of-vocabulary problem with product jargon, and partial matches
    ("更新率" vs "螢幕更新率") still overlap.
    """
    lowered = text.lower()
    tokens = _LATIN.findall(lowered)
    for run in _CJK.findall(lowered):
        tokens.extend(run)  # unigrams: recall
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigrams: precision
    return tokens


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25. Rebuilt from the corpus at load time -- it is milliseconds."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.term_freqs: list[Counter] = [Counter(d) for d in docs]

        df: Counter = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        # BM25 IDF with the +1 smoothing that keeps common terms non-negative.
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.n_docs:
            return []
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(q_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


class VectorIndex:
    """Exact cosine search over L2-normalised rows -- one matrix product."""

    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("embedding matrix must be 2-D")
        self.matrix = self._normalise(matrix.astype(np.float32, copy=False))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        # Guard against a zero vector from an empty chunk.
        return m / np.maximum(norms, 1e-12)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        q = self._normalise(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        # Normalised vectors -> the inner product *is* cosine similarity.
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def search_batch(
        self, query_matrix: np.ndarray, top_k: int = 10
    ) -> list[list[tuple[int, float]]]:
        """Batched search -- same amortisation trick as batching an LLM."""
        q = self._normalise(np.asarray(query_matrix, dtype=np.float32))
        scores = q @ self.matrix.T  # (n_queries, n_chunks)
        out = []
        for row in scores:
            order = np.argsort(-row)[:top_k]
            out.append([(int(i), float(row[i])) for i in order])
        return out


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Uses *ranks*, not scores, so a cosine similarity in [0, 1] and an unbounded
    BM25 score can be combined without normalising either. ``k`` damps the
    influence of the top position; 60 is the value from the original paper and
    is what everyone ends up using.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


@dataclass
class IndexBundle:
    """Everything needed to answer a query, minus the models."""

    chunk_ids: list[str]
    embeddings: np.ndarray
    embed_model: str
    dim: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            meta=np.array(
                json.dumps(
                    {
                        "chunk_ids": self.chunk_ids,
                        "embed_model": self.embed_model,
                        "dim": self.dim,
                    },
                    ensure_ascii=False,
                )
            ),
        )

    @classmethod
    def load(cls, path: Path) -> IndexBundle:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `uv run aorus-rag build` to create the index."
            )
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                chunk_ids=meta["chunk_ids"],
                embeddings=data["embeddings"],
                embed_model=meta["embed_model"],
                dim=meta["dim"],
            )
