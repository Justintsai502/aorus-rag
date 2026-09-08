"""Streaming generation on llama.cpp, with the timing instrumentation built in.

Two backends:

``LlamaCppLLM``     in-process via llama-cpp-python. Lowest TTFT -- no HTTP
                    round trip, no serialisation -- and the default.
``LlamaServerLLM``  talks OpenAI-style SSE to a running ``llama-server``.
                    Same measurements, deployment-shaped.

Timing definitions used throughout this project (stated explicitly because
"TPS" means at least three different things in the wild):

    TTFT      request submitted -> first non-empty token emitted
    decode_s  first token -> last token
    tps       (n_tokens - 1) / decode_s          decode-only, excludes prefill
    e2e_tps   n_tokens / total_s                 end-to-end, includes prefill

Reporting decode-only TPS separately from end-to-end matters here: prefill is
a GEMM over the whole prompt and decode is a GEMV per token, so mixing them
hides exactly the trade-off that retrieval depth controls.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import GENERATION_MODELS, MODELS_DIR, ModelSpec

# ggml tensor types accepted for the KV cache. Quantising it roughly halves
# the KV footprint, which is the second-largest line in the VRAM ledger.
KV_TYPES = {"f16": 1, "q8_0": 8, "q5_1": 7, "q4_0": 2}


@dataclass
class StreamStats:
    """Everything the benchmark needs from one generation."""

    ttft_s: float = 0.0
    total_s: float = 0.0
    decode_s: float = 0.0
    n_tokens: int = 0
    prompt_tokens: int = 0
    tps: float = 0.0
    e2e_tps: float = 0.0
    backend: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)

    def finalise(self) -> StreamStats:
        self.tps = (
            (self.n_tokens - 1) / self.decode_s if self.decode_s > 0 and self.n_tokens > 1 else 0.0
        )
        self.e2e_tps = self.n_tokens / self.total_s if self.total_s > 0 else 0.0
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class BaseLLM:
    name: str = ""
    backend: str = ""

    def stream(self, messages, max_tokens: int, temperature: float) -> Iterator[str]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 384,
        temperature: float = 0.2,
        on_token=None,
    ) -> tuple[str, StreamStats]:
        """Run a full generation, timing it token by token."""
        stats = StreamStats(backend=self.backend, model=self.name)
        try:
            stats.prompt_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        except Exception:  # noqa: BLE001 - tokenizer may be unavailable (server backend)
            stats.prompt_tokens = 0

        pieces: list[str] = []
        t0 = time.perf_counter()
        t_first: float | None = None
        t_last = t0

        for piece in self.stream(messages, max_tokens=max_tokens, temperature=temperature):
            if not piece:
                continue
            now = time.perf_counter()
            if t_first is None:
                t_first = now
                stats.ttft_s = now - t0
            t_last = now
            stats.n_tokens += 1
            pieces.append(piece)
            if on_token is not None:
                on_token(piece)

        stats.total_s = time.perf_counter() - t0
        stats.decode_s = (t_last - t_first) if t_first is not None else 0.0
        return "".join(pieces), stats.finalise()


class LlamaCppLLM(BaseLLM):
    """In-process llama.cpp."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        kv_type: str = "q8_0",
        seed: int = 1234,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            import llama_cpp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. Build it against your GPU:\n"
                '  CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llama   # macOS\n'
                '  CMAKE_ARGS="-DGGML_CUDA=on"  uv sync --extra llama   # CUDA'
            ) from exc

        path = model_path or (MODELS_DIR / spec.filename)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Fetch it with `bash scripts/download_models.sh`."
            )

        self.spec = spec
        self.name = spec.name
        self.backend = "llama-cpp-python"
        self.n_ctx = n_ctx
        self.kv_type = kv_type

        kwargs = {
            "model_path": str(path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "n_threads": n_threads,
            "verbose": verbose,
        }
        if kv_type != "f16":
            # Quantised KV needs flash attention in llama.cpp.
            kwargs.update(type_k=KV_TYPES[kv_type], type_v=KV_TYPES[kv_type], flash_attn=True)
        try:
            self._llama = llama_cpp.Llama(**kwargs)
        except TypeError:
            # Older binding without type_k/type_v/flash_attn: fall back to f16 KV
            # rather than refusing to run.
            for key in ("type_k", "type_v", "flash_attn"):
                kwargs.pop(key, None)
            self.kv_type = "f16"
            self._llama = llama_cpp.Llama(**kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def stream(self, messages, max_tokens: int, temperature: float) -> Iterator[str]:
        for part in self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            delta = part["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


class LlamaServerLLM(BaseLLM):
    """OpenAI-compatible SSE against a running ``llama-server``."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "local") -> None:
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.backend = "llama-server"

    def count_tokens(self, text: str) -> int:
        import httpx

        resp = httpx.post(f"{self.base_url}/tokenize", json={"content": text}, timeout=30)
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))

    def stream(self, messages, max_tokens: int, temperature: float) -> Iterator[str]:
        import httpx

        payload = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        with httpx.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=payload, timeout=300
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content


def build_llm(
    model: str = "qwen2.5-3b",
    backend: str = "in-process",
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    kv_type: str = "q8_0",
    seed: int = 1234,
    server_url: str = "http://127.0.0.1:8080",
    verbose: bool = False,
) -> BaseLLM:
    if backend == "server":
        return LlamaServerLLM(base_url=server_url, model=model)
    spec = GENERATION_MODELS.get(model)
    if spec is None:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(GENERATION_MODELS)}")
    return LlamaCppLLM(
        spec,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        kv_type=kv_type,
        seed=seed,
        verbose=verbose,
    )
