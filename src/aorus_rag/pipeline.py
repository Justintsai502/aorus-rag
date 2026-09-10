"""End-to-end orchestration: build the corpus, build the index, answer a query.

Deliberately split into two phases so the two models are never resident at the
same time:

    build   fetch -> parse -> normalise -> chunk -> embed -> index.npz
            (embedding model only; peak footprint ~0.3 GB)
    ask     load index.npz -> retrieve -> generate
            (generation model on GPU, embedding model on CPU)

That split is why the VRAM ledger in the README only has to account for the
generation model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import chunk as chunking
from . import fetch, normalize, parse
from .chunk import Chunk
from .config import CORPUS_PATH, INDEX_PATH, RuntimeConfig
from .embed import Embedder, build_embedder
from .index import IndexBundle
from .llm import BaseLLM, StreamStats
from .prompt import build_no_rag_messages, build_prompt, detect_language, to_messages
from .retrieve import Hit, Retriever, embed_corpus

# --------------------------------------------------------------------------
# Build phase
# --------------------------------------------------------------------------


def build_corpus(offline: bool = True, force_fetch: bool = False) -> list[Chunk]:
    """Parse the cached pages into the chunk corpus and write corpus.jsonl."""
    if not offline or force_fetch:
        fetch.fetch_all(force=force_fetch)

    zh_items = parse.parse_spec_table(fetch.load_cached("spec_zh"))
    en_items = parse.parse_spec_table(fetch.load_cached("spec_en"))
    parse.validate_spec_items(zh_items, "spec_zh")
    parse.validate_spec_items(en_items, "spec_en")

    facts = normalize.extract_facts(zh_items, en_items)
    variants, differing = parse.parse_sku_variants(fetch.load_cached("spec_zh"))

    chunks: list[Chunk] = []
    chunks += chunking.chunk_facts(facts)
    chunks += chunking.chunk_spec_rows(zh_items, en_items)
    chunks += chunking.chunk_sku_variants(variants, differing, zh_items, en_items)
    for lang, key in (("zh", "feature_zh"), ("en", "feature_en")):
        try:
            blocks = parse.parse_feature_page(fetch.load_cached(key))
        except FileNotFoundError:
            continue
        chunks += chunking.chunk_features(blocks, lang)

    chunks = chunking.dedupe(chunks)
    chunking.write_corpus(chunks, CORPUS_PATH)
    return chunks


def build_index(chunks: list[Chunk], embedder: Embedder) -> IndexBundle:
    bundle = embed_corpus(chunks, embedder)
    bundle.save(INDEX_PATH)
    return bundle


# --------------------------------------------------------------------------
# Query phase
# --------------------------------------------------------------------------


@dataclass
class AnswerResult:
    question: str
    lang: str
    answer: str
    hits: list[Hit] = field(default_factory=list)
    stats: StreamStats = field(default_factory=StreamStats)
    retrieval_s: float = 0.0
    context_chars: int = 0

    @property
    def ttft_s(self) -> float:
        """End-to-end first-token latency, retrieval included."""
        return self.retrieval_s + self.stats.ttft_s

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "lang": self.lang,
            "answer": self.answer,
            "retrieval_s": self.retrieval_s,
            "context_chars": self.context_chars,
            "ttft_e2e_s": self.ttft_s,
            "hits": [
                {"chunk_id": h.chunk.chunk_id, "kind": h.chunk.kind, "score": h.score}
                for h in self.hits
            ],
            "stats": self.stats.to_dict(),
        }


class RagPipeline:
    """Retriever + LLM. Keeps no state between questions."""

    def __init__(
        self, retriever: Retriever, llm: BaseLLM, cfg: RuntimeConfig | None = None
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.cfg = cfg or RuntimeConfig()

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k=top_k or self.cfg.top_k)
        return hits, time.perf_counter() - t0

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
        on_token=None,
        use_rag: bool = True,
    ) -> AnswerResult:
        lang = detect_language(question)

        if not use_rag:
            messages = build_no_rag_messages(question, lang)
            text, stats = self.llm.generate(
                messages,
                max_tokens=max_tokens or self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                on_token=on_token,
            )
            return AnswerResult(question=question, lang=lang, answer=text, stats=stats)

        hits, retrieval_s = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, hits, lang=lang)
        text, stats = self.llm.generate(
            to_messages(prompt),
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            on_token=on_token,
        )
        return AnswerResult(
            question=question,
            lang=lang,
            answer=text,
            hits=prompt.used_hits,
            stats=stats,
            retrieval_s=retrieval_s,
            context_chars=prompt.context_chars,
        )


def load_retriever(
    mode: str = "hybrid",
    embed_model: str | None = None,
    on_degrade: Callable[[str], None] | None = None,
) -> Retriever:
    """Load the best retrieval the environment can actually run.

    The corpus and vector index ship with the repo, so nothing is built here.
    What may be missing is the *embedder*: dense retrieval encodes the query at
    request time, which needs llama-cpp-python and a downloaded GGUF. Rather
    than refusing to start, fall back to BM25 -- no model, 0.14 ms, Recall@5 =
    1.000 on the eval set, but blind to paraphrases. ``on_degrade`` is called
    with an explanation so a caller can surface it.
    """
    chunks = chunking.read_corpus(CORPUS_PATH)
    if mode == "bm25":
        # Pure arithmetic over the corpus: no vectors, no model, no llama.cpp.
        return Retriever(chunks, None, None, mode=mode)

    bundle = IndexBundle.load(INDEX_PATH)
    name = embed_model or bundle.embed_model
    if name != bundle.embed_model:
        raise ValueError(
            f"index was built with {bundle.embed_model!r} but {name!r} was requested; "
            f"rebuild with `uv run aorus-rag build --embed-model {name}`"
        )
    try:
        embedder = build_embedder(name, n_gpu_layers=0)
    except (RuntimeError, FileNotFoundError) as exc:
        if on_degrade is not None:
            on_degrade(str(exc).splitlines()[0])
        return Retriever(chunks, None, None, mode="bm25")
    return Retriever(chunks, bundle, embedder, mode=mode)


def load_pipeline(
    cfg: RuntimeConfig,
    llm: BaseLLM,
    mode: str = "hybrid",
    on_degrade: Callable[[str], None] | None = None,
) -> RagPipeline:
    """Corpus + index + models, wired together. Single entry point."""
    return RagPipeline(load_retriever(mode, cfg.embed_model_or_index, on_degrade), llm, cfg)
