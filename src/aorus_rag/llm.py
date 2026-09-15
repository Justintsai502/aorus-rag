"""Streaming generation on llama.cpp, with the timing instrumentation built in.

Two backends:

``LlamaCppLLM``     in-process via llama-cpp-python. Lowest TTFT -- no HTTP
                    round trip, no serialisation -- and the default.
``LlamaServerLLM``  talks OpenAI-style SSE to an already-running
                    ``llama-server``. Same measurements, deployment-shaped.
                    Start one with::

                        uv sync --extra server
                        uv run python -m llama_cpp.server \
                            --model models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
                            --n_gpu_layers -1 --n_ctx 4096 --port 8080

                    then ``uv run aorus-rag ask "..." --backend server``.
                    The benchmark numbers in the README come from the
                    in-process backend; the server path is provided for
                    deployment shape and is not part of the measured results.

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
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import GENERATION_MODELS, MODELS_DIR, ModelSpec

# ggml tensor types accepted for the KV cache. Quantising it roughly halves
# the KV footprint, which is the second-largest line in the VRAM ledger.
# Values are ggml type ids (GGML_TYPE_*); llama.cpp takes the id, not the name.
KV_TYPES = {"f16": 1, "q8_0": 8, "q5_1": 7, "q4_0": 2}

# Qwen3 emits <think>...</think> before the answer even with thinking disabled
# (the block is just empty). Everything inside it is internal monologue: it must
# not count towards the answer, and the first token *after* it is the one a user
# actually waits for.
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# An unmatched closing tag shows up when the model re-opens its monologue after
# the switch; anything before it is still monologue.
THINK_ORPHAN = re.compile(r"^.*?</think>\s*", re.DOTALL)
# Qwen3's soft switches occasionally get echoed back into the output.
CONTROL_ECHO = re.compile(r"\s*/no_(?:think|output)\b\s*")
THINK_CLOSE = "</think>"


def strip_thinking(text: str) -> str:
    """Return only what a reader should see.

    Removes the reasoning block, an orphaned ``</think>`` (the model sometimes
    reopens the monologue after the soft switch), and any echoed ``/no_think``
    or ``/no_output`` control token. Measured on 108 answers, the echo appeared
    in 2 of them -- rare, but it lands in the user-visible text, so it is
    cleaned rather than tolerated.
    """
    # 1. remove complete <think>...</think> blocks
    # 2. remove everything up to an orphaned </think>
    # 3. remove echoed /no_think and /no_output switches
    out = THINK_BLOCK.sub("", text)
    if THINK_CLOSE in out:
        out = THINK_ORPHAN.sub("", out)
    out = CONTROL_ECHO.sub(" ", out)
    return out.strip()


@dataclass
class StreamStats:
    """Everything the benchmark needs from one generation."""

    # Times are in seconds, measured from request submission:
    #   ttft_s         first non-empty piece of any kind
    #   total_s        the whole generation
    #   decode_s       first piece -> last piece
    #   n_tokens       streamed pieces (normally one per token)
    #   prompt_tokens  approximate prompt length
    #   tps, e2e_tps   computed by finalise()
    ttft_s: float = 0.0
    total_s: float = 0.0
    decode_s: float = 0.0
    n_tokens: int = 0
    prompt_tokens: int = 0
    tps: float = 0.0
    e2e_tps: float = 0.0
    # First token overall vs first token of the actual answer. They differ only
    # on reasoning models, where the gap is the thinking block.
    ttft_answer_s: float = 0.0
    thinking_tokens: int = 0
    backend: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)

    def finalise(self) -> StreamStats:
        # n_tokens - 1: the first token arrives when prefill finishes, so the decode
        # interval (first token -> last token) contains n - 1 generated tokens.
        self.tps = (
            (self.n_tokens - 1) / self.decode_s if self.decode_s > 0 and self.n_tokens > 1 else 0.0
        )
        # End-to-end rate includes prefill, so it falls as the prompt grows even when
        # decode speed is unchanged.
        self.e2e_tps = self.n_tokens / self.total_s if self.total_s > 0 else 0.0
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# Subclasses implement stream() and count_tokens(); generate() layers timing on top.
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
            # Approximate: message contents tokenized separately, chat-template tokens excluded.
            stats.prompt_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        except Exception:  # noqa: BLE001 - tokenizer may be unavailable (server backend)
            stats.prompt_tokens = 0

        # pieces collects the raw output; seen is the same text as a running string for
        # think-block detection; t_answer marks when visible text first appears.
        pieces: list[str] = []
        # perf_counter is a monotonic high-resolution clock, unaffected by wall-clock changes.
        t0 = time.perf_counter()
        t_first: float | None = None
        t_answer: float | None = None
        t_last = t0
        in_thinking = False
        seen = ""

        for piece in self.stream(messages, max_tokens=max_tokens, temperature=temperature):
            # Empty pieces are ignored so they cannot trigger the TTFT timestamp.
            if not piece:
                continue
            now = time.perf_counter()
            # First non-empty token -> TTFT. This interval is dominated by prefill: running the
            # whole prompt through the model and filling the KV cache.
            if t_first is None:
                t_first = now
                stats.ttft_s = now - t0
            t_last = now
            stats.n_tokens += 1
            pieces.append(piece)

            # Track the reasoning block so ttft_answer_s measures the first
            # token the user actually reads.
            seen += piece
            if t_answer is None:
                # Still inside <think>: count as thinking, not as visible output.
                if "<think>" in seen and THINK_CLOSE not in seen:
                    in_thinking = True
                    stats.thinking_tokens += 1
                # Past </think>, or no think block at all: the first moment visible text exists
                # is the answer's first token.
                elif THINK_CLOSE in seen or "<think>" not in seen:
                    if in_thinking or "<think>" not in seen:
                        stripped = strip_thinking(seen)
                        if stripped:
                            t_answer = now
                            stats.ttft_answer_s = now - t0
            # Hand each piece to the caller as it arrives; the CLI prints it immediately.
            if on_token is not None:
                on_token(piece)

        stats.total_s = time.perf_counter() - t0
        # decode_s spans the first to the last piece; 0 if nothing was generated.
        stats.decode_s = (t_last - t_first) if t_first is not None else 0.0
        # Without a think block both TTFT figures are the same.
        if not stats.ttft_answer_s:
            stats.ttft_answer_s = stats.ttft_s
        # Raw text, think block included; callers strip it for display and scoring.
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
            # Imported lazily so BM25 retrieval and the tests work without llama-cpp-python.
            import llama_cpp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. Build it against your GPU:\n"
                '  CMAKE_ARGS="-DGGML_METAL=on" uv sync --extra llama   # macOS\n'
                '  CMAKE_ARGS="-DGGML_CUDA=on"  uv sync --extra llama   # CUDA'
            ) from exc

        # Default location is models/<filename>, where scripts/download_models.sh saves it.
        path = model_path or (MODELS_DIR / spec.filename)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Fetch it with `bash scripts/download_models.sh`."
            )

        self.spec = spec
        self.name = spec.name
        self.backend = "llama-cpp-python"
        self.thinking_switch = spec.thinking_switch
        self.n_ctx = n_ctx
        self.kv_type = kv_type

        # n_ctx sizes the KV cache up front; n_gpu_layers=-1 offloads every layer to the
        # GPU (Metal on macOS); seed makes sampling reproducible.
        kwargs = {
            "model_path": str(path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "n_threads": n_threads,
            "verbose": verbose,
        }
        # q8_0 stores each K / V value in 8 bits, half the f16 footprint.
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
        # The model's own tokenizer; special=True counts special tokens as single tokens.
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _apply_thinking_switch(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not self.thinking_switch:
            return messages
        # Copy so the caller's messages are not mutated. The switch is appended to the
        # system prompt, or sent as a new system message if there is none.
        out = [dict(m) for m in messages]
        for m in out:
            if m["role"] == "system":
                m["content"] = f"{m['content']} {self.thinking_switch}"
                return out
        out.insert(0, {"role": "system", "content": self.thinking_switch})
        return out

    def stream(self, messages, max_tokens: int, temperature: float) -> Iterator[str]:
        messages = self._apply_thinking_switch(messages)
        # stream=True yields one OpenAI-style delta per generated token instead of
        # returning once generation finishes.
        for part in self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            # Each part is an OpenAI-style chunk; new text is in choices[0].delta.content,
            # which is absent on chunks that carry only the role.
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

        # llama-server's /tokenize endpoint returns {"tokens": [...]}.
        resp = httpx.post(f"{self.base_url}/tokenize", json={"content": text}, timeout=30)
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))

    def stream(self, messages, max_tokens: int, temperature: float) -> Iterator[str]:
        import httpx

        # OpenAI chat-completions request with stream=True, so the server answers with SSE.
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
            # Server-Sent Events: each event is a "data: {json}" line; the stream ends with
            # "data: [DONE]".
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                # Strip the "data:" prefix.
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                # Skip keep-alive or partial lines instead of aborting the whole stream.
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
    # Both backends subclass BaseLLM, so generate()'s timing logic is shared.
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
