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

# PKG_ROOT is src/aorus_rag/; two levels up (parents[1]) is the project root.
PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parents[1]

# Every directory can be overridden by an environment variable (e.g. to keep
# models on a separate disk on Kaggle); otherwise it defaults to the project tree.
DATA_DIR = Path(os.environ.get("AORUS_RAG_DATA", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = Path(os.environ.get("AORUS_RAG_MODELS", PROJECT_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("AORUS_RAG_RESULTS", PROJECT_ROOT / "results"))

# The three core data files:
#   corpus.jsonl  the chunks, one JSON object per line
#   index.npz     one vector per chunk (a 240 x 1024 matrix)
#   qa.jsonl      evaluation questions and expected answers
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

# In practice the headers alone are not sufficient: HTTP/2 is also required.
# See fetch.py for the measurements.
PRODUCT = "AORUS-MASTER-16-AM6H"

# Four pages: the spec sheet (zh / en) and the feature page (zh / en). The spec
# sheet is the primary corpus; the feature page is marketing prose that covers
# "how / why" questions.
SOURCES: dict[str, str] = {
    # The /tw/ prefix selects the Taiwan (Traditional Chinese) site; the /sp suffix is
    # the spec tab, and the bare product URL is the feature page.
    "spec_zh": f"https://www.gigabyte.com/tw/Laptop/{PRODUCT}/sp",
    "spec_en": f"https://www.gigabyte.com/Laptop/{PRODUCT}/sp",
    "feature_zh": f"https://www.gigabyte.com/tw/Laptop/{PRODUCT}",
    "feature_en": f"https://www.gigabyte.com/Laptop/{PRODUCT}",
}

# Chrome 128 on macOS. The sec-ch-ua* and Sec-Fetch-* fields are only sent by real
# browsers; without them Akamai classifies the request as a bot.
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


# One ModelSpec describes one GGUF model: where to download it (repo + filename),
# how large it is, and how to load it. frozen=True makes it immutable at runtime.
@dataclass(frozen=True)
class ModelSpec:
    """A GGUF model we know how to fetch and load."""

    # name       identifier used on the CLI (--model / --embed-model)
    # repo       HuggingFace repository; with filename it forms the download URL
    #            https://huggingface.co/{repo}/resolve/main/{filename}
    # approx_gb  file size on disk, quoted in error messages and the VRAM ledger
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
# Generation models (decoder-only LLMs), all Q4_K_M: roughly 4.5-5 bits per
# weight on average, the usual size/quality sweet spot.
#   qwen3-1.7b  current default
#   qwen2.5-3b  original default, kept as the comparison point
#   qwen3-4b    fits in 4 GB; listed as an option, not part of the measured results
# thinking_switch="/no_think" turns off Qwen3's reason-before-answering mode.
GENERATION_MODELS: dict[str, ModelSpec] = {
    # No thinking_switch: Qwen2.5 has no reasoning mode to turn off.
    "qwen2.5-3b": ModelSpec(
        name="qwen2.5-3b",
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        approx_gb=1.93,
    ),
    # Default. Measured at top_k=5 on the 36-question core set: 96.8% keyword
    # accuracy, 100% refusal on negatives, 100% number grounding, 61.7 tok/s decode.
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

# Embedding models (BERT-style encoders) map a text to one fixed-length vector.
# ``pooling`` decides how the per-token outputs become that single vector:
#   cls   take the output at token 0 ([CLS]); with bidirectional attention it has
#         already attended to the whole input
#   mean  average all token outputs
# ``n_ctx`` is the model's maximum input length; chunks are far shorter.
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
    # bge-m3: multilingual, 1024-dim output, CLS pooling. Q8_0 (8-bit) keeps quality
    # high and the file is still only 0.63 GB.
    "bge-m3": ModelSpec(
        name="bge-m3",
        repo="lm-kit/bge-m3-gguf",
        filename="bge-m3-Q8_0.gguf",
        approx_gb=0.63,
        pooling="cls",
        n_ctx=1024,
    ),
}

# Defaults can be switched with environment variables, no code change needed.
DEFAULT_GENERATION_MODEL = os.environ.get("AORUS_RAG_GEN_MODEL", "qwen3-1.7b")
DEFAULT_EMBEDDING_MODEL = os.environ.get("AORUS_RAG_EMBED_MODEL", "bge-m3")


@dataclass
class RuntimeConfig:
    """Knobs that affect the VRAM ledger and the latency numbers."""

    # The model, n_ctx, kv_type and n_gpu_layers together determine VRAM usage.
    # n_ctx is the context length and also the number of KV cache slots allocated up front:
    #   KV cache size ~ layers x KV heads x head dim x n_ctx x bits per value
    gen_model: str = DEFAULT_GENERATION_MODEL
    embed_model: str = DEFAULT_EMBEDDING_MODEL
    n_ctx: int = 4096
    # Quantised KV cache halves the KV footprint at negligible quality cost.
    kv_type: str = "q8_0"
    # -1 offloads every layer; the embedding model stays on CPU so that the
    # VRAM ledger only has to account for the generation model.
    n_gpu_layers: int = -1
    embed_n_gpu_layers: int = 0
    # top_k: chunks placed in the prompt. More chunks -> longer prompt -> longer
    # prefill -> higher TTFT.
    top_k: int = 5
    # temperature=0.2 keeps answers stable -- spec lookup wants consistency, not
    # creativity. A fixed seed makes sampling reproducible.
    max_tokens: int = 384
    temperature: float = 0.2
    seed: int = 1234
    # The 4 GB VRAM limit from the task, recorded for reference.
    vram_budget_mb: int = 4096
    # Free-form slot for experiment-specific settings. default_factory gives every
    # instance its own dict; dataclasses reject a plain {} default because it would be
    # shared between instances.
    extra: dict = field(default_factory=dict)

    @property
    def embed_model_or_index(self) -> str | None:
        """None means "whatever the index was built with" -- the safe default.

        Only an explicit override should ever contradict the index, and that is
        rejected rather than silently producing meaningless cosine scores.
        """
        # Query vectors and document vectors must come from the same model; vectors from
        # different models live in different spaces and their cosine is meaningless.
        return None
