"""Embedding backends.

The primary backend runs the embedding model through llama.cpp -- the same
inference engine used for generation. That is a deliberate resource decision:
the obvious alternative (``sentence-transformers``) drags in ``torch``, which
costs ~300 MB of RSS on macOS and ~2.5 GB of disk in a CUDA image before a
single weight is loaded. Keeping one engine means the whole project needs
neither torch nor transformers, and the embedding model can sit on the CPU so
that the VRAM ledger only has to account for the generation model.

A dependency-free ``HashingEmbedder`` is included so the pipeline, the tests
and the retrieval evaluation all run with no model download at all. It is a
development aid, not a serious retriever -- it is reported as such in results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import EMBEDDING_MODELS, MODELS_DIR, ModelSpec

# E5-family models are trained with asymmetric prefixes; omitting them costs a
# few points of recall. BGE-M3 needs none.
PREFIXES: dict[str, tuple[str, str]] = {
    "e5-small": ("query: ", "passage: "),
    "bge-m3": ("", ""),
}


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


# --------------------------------------------------------------------------


class HashingEmbedder:
    """Signed feature hashing over the shared tokenizer. No model, no download.

    Deterministic and instant, which makes it the right default for tests and
    for verifying the plumbing before committing to a multi-GB download. It
    captures lexical overlap only -- it cannot match 顯示器 to "screen".
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = "hashing"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        from .index import tokenize

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))


# --------------------------------------------------------------------------


class LlamaCppEmbedder:
    """GGUF embedding model served by llama.cpp.

    ``n_gpu_layers`` defaults to 0: the embedding model stays on the CPU so the
    4 GB VRAM budget is spent entirely on the generation model. Query latency
    cost is a few milliseconds for a 118M-parameter model.
    """

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None = None,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            import llama_cpp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the inference engine with:\n"
                '  CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llama   # macOS\n'
                '  CMAKE_ARGS="-DGGML_CUDA=on"  uv sync --extra llama   # CUDA\n'
                "Or smoke-test the pipeline with no model at all:\n"
                "  uv run aorus-rag build --embed-model hashing"
            ) from exc

        path = model_path or (MODELS_DIR / spec.filename)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found ({spec.approx_gb:.2f} GB).\n"
                f"  Download it:  bash scripts/download_models.sh {spec.name}\n"
                "  Or smoke-test the pipeline with no model at all:\n"
                "                uv run aorus-rag build --embed-model hashing"
            )

        pooling = {
            "mean": getattr(llama_cpp, "LLAMA_POOLING_TYPE_MEAN", 1),
            "cls": getattr(llama_cpp, "LLAMA_POOLING_TYPE_CLS", 2),
        }.get(spec.pooling or "mean", 1)

        self.spec = spec
        self.name = spec.name
        self._llama = llama_cpp.Llama(
            model_path=str(path),
            embedding=True,
            n_ctx=spec.n_ctx,
            n_gpu_layers=n_gpu_layers,
            pooling_type=pooling,
            n_threads=n_threads,
            verbose=verbose,
        )
        probe = self._llama.create_embedding("dimension probe")
        self.dim = len(self._to_vector(probe["data"][0]["embedding"]))

    @staticmethod
    def _to_vector(embedding) -> np.ndarray:
        """llama.cpp returns either a vector or a per-token matrix."""
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim == 2:  # unpooled: average the token vectors ourselves
            arr = arr.mean(axis=0)
        return arr

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        q_prefix, p_prefix = PREFIXES.get(self.spec.name, ("", ""))
        prefix = q_prefix if is_query else p_prefix
        payload = [prefix + t for t in texts]
        response = self._llama.create_embedding(payload)
        return np.stack([self._to_vector(row["embedding"]) for row in response["data"]])


# --------------------------------------------------------------------------


def build_embedder(name: str, n_gpu_layers: int = 0, verbose: bool = False) -> Embedder:
    """Resolve an embedder by name, falling back to hashing when asked."""
    if name in ("hash", "hashing", "none"):
        return HashingEmbedder()
    spec = EMBEDDING_MODELS.get(name)
    if spec is None:
        raise KeyError(
            f"unknown embedding model {name!r}; choose from {sorted(EMBEDDING_MODELS)} or 'hashing'"
        )
    return LlamaCppEmbedder(spec, n_gpu_layers=n_gpu_layers, verbose=verbose)
