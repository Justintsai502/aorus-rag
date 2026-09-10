"""Central configuration: paths, source URLs and model defaults.

Everything that a reader might want to change lives here, so the rest of the
package has no magic constants buried in it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parents[1]

DATA_DIR = Path(os.environ.get("AORUS_RAG_DATA", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = Path(os.environ.get("AORUS_RAG_MODELS", PROJECT_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("AORUS_RAG_RESULTS", PROJECT_ROOT / "results"))

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
INDEX_PATH = DATA_DIR / "index.npz"
QA_PATH = EVAL_DIR / "qa.jsonl"

# --------------------------------------------------------------------------
# Sources
#
# The product page is served through Akamai Bot Manager: a bare `curl` gets a
# 403. A complete set of browser headers gets a 200, so no headless browser is
# needed -- the page is server-side rendered.
# --------------------------------------------------------------------------

PRODUCT = "AORUS-MASTER-16-AM6H"

SOURCES: dict[str, str] = {
    "spec_zh": f"https://www.gigabyte.com/tw/Laptop/{PRODUCT}/sp",
    "spec_en": f"https://www.gigabyte.com/Laptop/{PRODUCT}/sp",
    "feature_zh": f"https://www.gigabyte.com/tw/Laptop/{PRODUCT}",
    "feature_en": f"https://www.gigabyte.com/Laptop/{PRODUCT}",
}

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# --------------------------------------------------------------------------
# Models
#
# VRAM ledger (see README). Defaults target the 4 GB budget with headroom:
#   generation  Qwen3-1.7B Q4_K_M            ~1.11 GB
#   KV cache    n_ctx=4096, type_k/v=q8_0    ~0.23 GB
#   overhead    compute buffers              ~0.30 GB
#   ------------------------------------------------  ~1.64 GB VRAM
#   embedding   bge-m3 Q8_0                  ~0.63 GB  (CPU, outside the budget)
#
# The default was Qwen2.5-3B until measured: at top_k=5 the 1.7B matches it on
# every quality metric while running 1.7x faster on 0.67 GB less. See README 7.4.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A GGUF model we know how to fetch and load."""

    name: str
    repo: str
    filename: str
    approx_gb: float
    # Embedding models only: pooling strategy llama.cpp should use.
    pooling: str | None = None
    n_ctx: int = 4096
    # Qwen3 is a hybrid reasoning model with thinking mode ON by default. Left
    # alone it emits a <think> monologue -- in Simplified Chinese, at that --
    # before every answer, which inflates token counts, delays the first
    # *useful* token, and makes any metric computed over the raw output
    # meaningless. "/no_think" is Qwen3's documented soft switch.
    thinking_switch: str | None = None


# Repos, filenames and sizes below were verified against the HuggingFace API
# on 2026-09-08; sizes are the actual downloaded file sizes.
GENERATION_MODELS: dict[str, ModelSpec] = {
    "qwen2.5-3b": ModelSpec(
        name="qwen2.5-3b",
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        approx_gb=1.93,
    ),
    "qwen3-1.7b": ModelSpec(
        name="qwen3-1.7b",
        repo="unsloth/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        approx_gb=1.11,
        thinking_switch="/no_think",
    ),
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        approx_gb=2.50,
        thinking_switch="/no_think",
    ),
}

EMBEDDING_MODELS: dict[str, ModelSpec] = {
    # NOTE: this conversion predates a llama.cpp requirement and fails to load
    # with "bert model needs to define token type count" -- the GGUF carries
    # tokenizer.ggml.token_type but not bert.token_type_count. Kept here so the
    # smaller option is documented; use bge-m3 unless a newer conversion appears.
    "e5-small": ModelSpec(
        name="e5-small",
        repo="cstr/multilingual-e5-small-GGUF",
        filename="multilingual-e5-small-q8_0.gguf",
        approx_gb=0.13,
        pooling="mean",
        n_ctx=512,
    ),
    "bge-m3": ModelSpec(
        name="bge-m3",
        repo="lm-kit/bge-m3-gguf",
        filename="bge-m3-Q8_0.gguf",
        approx_gb=0.63,
        pooling="cls",
        n_ctx=1024,
    ),
}

DEFAULT_GENERATION_MODEL = os.environ.get("AORUS_RAG_GEN_MODEL", "qwen3-1.7b")
DEFAULT_EMBEDDING_MODEL = os.environ.get("AORUS_RAG_EMBED_MODEL", "bge-m3")


@dataclass
class RuntimeConfig:
    """Knobs that affect the VRAM ledger and the latency numbers."""

    gen_model: str = DEFAULT_GENERATION_MODEL
    embed_model: str = DEFAULT_EMBEDDING_MODEL
    n_ctx: int = 4096
    # Quantised KV cache halves the KV footprint at negligible quality cost.
    kv_type: str = "q8_0"
    # -1 offloads every layer; the embedding model stays on CPU so that the
    # VRAM ledger only has to account for the generation model.
    n_gpu_layers: int = -1
    embed_n_gpu_layers: int = 0
    top_k: int = 5
    max_tokens: int = 384
    temperature: float = 0.2
    seed: int = 1234
    vram_budget_mb: int = 4096
    extra: dict = field(default_factory=dict)
