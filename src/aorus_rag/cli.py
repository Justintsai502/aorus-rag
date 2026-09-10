"""Command line interface.

uv run aorus-rag fetch                 refresh the cached HTML
uv run aorus-rag build                 parse -> chunk -> embed -> index
uv run aorus-rag inspect               show the parsed spec table
uv run aorus-rag search "..."          retrieval only, no model needed
uv run aorus-rag ask "..."             full RAG answer, streamed
uv run aorus-rag eval-retrieval        dense vs bm25 vs hybrid comparison
uv run aorus-rag bench                 TTFT / TPS / answer quality
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import bench as benchmarks
from . import fetch as fetching
from .config import (
    CORPUS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    EMBEDDING_MODELS,
    GENERATION_MODELS,
    INDEX_PATH,
    RESULTS_DIR,
    RuntimeConfig,
)
from .embed import build_embedder
from .llm import build_llm
from .pipeline import RagPipeline, build_corpus, build_index, load_retriever
from .retrieve import Retriever

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _runtime_config(args) -> RuntimeConfig:
    return RuntimeConfig(
        gen_model=getattr(args, "model", DEFAULT_GENERATION_MODEL),
        embed_model=getattr(args, "embed_model", DEFAULT_EMBEDDING_MODEL),
        n_ctx=getattr(args, "n_ctx", 4096),
        kv_type=getattr(args, "kv_type", "q8_0"),
        n_gpu_layers=getattr(args, "n_gpu_layers", -1),
        top_k=getattr(args, "top_k", 4),
        max_tokens=getattr(args, "max_tokens", 384),
        temperature=getattr(args, "temperature", 0.2),
    )


HASHING_WARNING = (
    "!! Using the 'hashing' fallback embedder: it captures lexical overlap only\n"
    "!! and has NO semantic ability (it cannot match 螢幕多亮 to 'brightness').\n"
    "!! Retrieval numbers from it are a lexical-only lower bound, not a result.\n"
    "!! For real numbers: bash scripts/download_models.sh && "
    "uv run aorus-rag build --embed-model bge-m3"
)


def _warn_hashing() -> None:
    print(HASHING_WARNING, file=sys.stderr)


def _degraded(reason: str) -> None:
    print(
        f"note: dense retrieval unavailable ({reason})\n"
        "note: falling back to BM25 -- no model needed, Recall@5 = 1.000 on the\n"
        "note: eval set, but it cannot match paraphrases. To enable dense/hybrid:\n"
        "note:   bash scripts/download_models.sh bge-m3",
        file=sys.stderr,
    )


def _load_retriever(mode: str, embed_model: str | None = None) -> Retriever:
    r = load_retriever(mode, embed_model, on_degrade=_degraded)
    if r.embedder is not None and r.embedder.name == "hashing":
        _warn_hashing()
    return r


def _load_pipeline(args) -> RagPipeline:
    cfg = _runtime_config(args)
    retriever = _load_retriever(args.mode)
    llm = build_llm(
        model=args.model,
        backend=args.backend,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        kv_type=cfg.kv_type,
        seed=cfg.seed,
        server_url=getattr(args, "server_url", "http://127.0.0.1:8080"),
        verbose=getattr(args, "verbose", False),
    )
    return RagPipeline(retriever, llm, cfg)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_fetch(args) -> int:
    if args.status:
        for key, info in fetching.cache_status().items():
            state = "cached" if info["cached"] else "MISSING"
            size = f"{info['bytes'] / 1024:.0f} KB" if info["bytes"] else "-"
            print(f"{key:12s} {state:8s} {size:>9s}  {info['fetched_at'] or ''}")
        return 0
    paths = fetching.fetch_all(force=args.force)
    for key, path in paths.items():
        print(f"{key:12s} -> {path}")
    return 0


def cmd_build(args) -> int:
    t0 = time.perf_counter()
    chunks = build_corpus(offline=not args.refresh, force_fetch=args.refresh)
    parse_s = time.perf_counter() - t0

    kinds: dict[str, int] = {}
    for c in chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    print(f"corpus  {len(chunks)} chunks in {parse_s:.2f}s")
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"         {kind:11s} {n:4d}")
    print(f"         -> {CORPUS_PATH}")

    t1 = time.perf_counter()
    embedder = build_embedder(args.embed_model, n_gpu_layers=args.embed_gpu_layers)
    if embedder.name == "hashing":
        _warn_hashing()
    bundle = build_index(chunks, embedder)
    print(
        f"index   {bundle.embeddings.shape[0]} x {bundle.dim} "
        f"({bundle.embed_model}) in {time.perf_counter() - t1:.2f}s"
    )
    print(f"         -> {INDEX_PATH}")
    return 0


def cmd_inspect(args) -> int:
    from . import fetch, normalize, parse

    zh = parse.parse_spec_table(fetch.load_cached("spec_zh"))
    en = parse.parse_spec_table(fetch.load_cached("spec_en"))
    parse.validate_spec_items(zh, "spec_zh")
    parse.validate_spec_items(en, "spec_en")

    if args.facts:
        for fact in normalize.extract_facts(zh, en):
            print(f"{fact.fact_id:28s} {fact.label_zh} / {fact.label_en} = {fact.value}")
        return 0

    for z, e in zip(zh, en):
        print(f"\n[{z.index:2d}] {z.key}  /  {e.key}")
        for line in z.lines:
            print(f"      {line}")
        for note in z.footnotes:
            print(f"      (note) {note}")
    return 0


def cmd_search(args) -> int:
    retriever = _load_retriever(args.mode)
    t0 = time.perf_counter()
    hits = retriever.search(args.question, top_k=args.top_k)
    elapsed = (time.perf_counter() - t0) * 1000
    # Report what actually ran, not what was asked for: retrieval may have
    # degraded to BM25 when no embedder was available.
    print(f"{len(hits)} hits in {elapsed:.1f} ms  (mode={retriever.mode})\n")
    for h in hits:
        ranks = f"dense={h.dense_rank} bm25={h.bm25_rank}"
        print(f"[{h.rank + 1}] {h.score:.4f}  {h.chunk.kind:9s} {ranks}")
        print(f"     {h.chunk.text}")
    return 0


def cmd_ask(args) -> int:
    pipeline = _load_pipeline(args)

    # Reasoning models stream a <think> block before the answer. With thinking
    # switched off it is empty, but printing "<think></think>" to a user is
    # noise, so suppress it as it streams rather than after the fact.
    state = {"buf": "", "open": False, "done": False}

    def emit(piece: str) -> None:
        if state["done"]:
            sys.stdout.write(piece)
            sys.stdout.flush()
            return
        state["buf"] += piece
        buf = state["buf"]
        if "</think>" in buf:
            state["done"] = True
            tail = buf.split("</think>", 1)[1].lstrip()
            state["buf"] = ""
            if tail:
                sys.stdout.write(tail)
                sys.stdout.flush()
        elif (
            "<think>" in buf or "<think".startswith(buf.strip()[:6]) or buf.strip().startswith("<")
        ):
            state["open"] = True  # still inside (or possibly entering) the block
        else:
            state["done"] = True
            sys.stdout.write(buf)
            sys.stdout.flush()
            state["buf"] = ""

    result = pipeline.answer(
        args.question,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        on_token=None if args.no_stream else emit,
        use_rag=not args.no_rag,
    )
    from .llm import strip_thinking

    result.answer = strip_thinking(result.answer)
    if args.no_stream:
        print(result.answer)
    else:
        print()

    if args.show_context and result.hits:
        print("\n--- context ---")
        for i, h in enumerate(result.hits):
            print(f"[{i + 1}] {h.chunk.text}")

    s = result.stats
    print(
        f"\n--- retrieval {result.retrieval_s * 1000:.1f} ms | "
        f"TTFT {result.ttft_s:.3f} s (model {s.ttft_s:.3f} s"
        + (
            f", 首個答案 token {result.retrieval_s + s.ttft_answer_s:.3f} s"
            if s.thinking_tokens
            else ""
        )
        + ") | "
        f"{s.tps:.1f} tok/s decode | {s.e2e_tps:.1f} tok/s e2e | "
        f"{s.n_tokens} tokens | prompt {s.prompt_tokens} tokens"
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_eval_retrieval(args) -> int:
    questions = benchmarks.load_questions()
    payload = {"embed_model": None, "modes": {}}
    for mode in args.modes:
        retriever = _load_retriever(mode)
        payload["embed_model"] = retriever.embedder.name if retriever.embedder else None
        payload["modes"][mode] = benchmarks.evaluate_retrieval(retriever, questions)

    print(f"{'mode':8s} {'R@1':>7s} {'R@3':>7s} {'R@5':>7s} {'MRR':>7s} {'ms':>7s}")
    for mode, res in payload["modes"].items():
        r = res["recall"]
        print(
            f"{mode:8s} {r['@1']:7.3f} {r['@3']:7.3f} {r['@5']:7.3f} "
            f"{res['mrr']:7.3f} {res['latency_ms_median']:7.2f}"
        )

    out = RESULTS_DIR / args.output
    benchmarks.save_results(payload, out)
    print(f"\nwritten to {out}")
    return 0


def cmd_bench(args) -> int:
    pipeline = _load_pipeline(args)
    questions = benchmarks.load_questions()
    if args.limit:
        questions = questions[: args.limit]

    payload = {
        "config": {
            "model": args.model,
            "backend": args.backend,
            "n_ctx": args.n_ctx,
            "kv_type": args.kv_type,
            "n_gpu_layers": args.n_gpu_layers,
            "retriever": args.mode,
            "embed_model": pipeline.retriever.embedder.name
            if pipeline.retriever.embedder
            else None,
        },
        "runs": {},
    }

    for top_k in args.top_k_sweep:
        label = f"rag_top{top_k}"
        print(f"\n=== {label} ===")
        payload["runs"][label] = benchmarks.evaluate_generation(
            pipeline, questions, repeats=args.repeats, top_k=top_k
        )
        r = payload["runs"][label]
        print(
            f"TTFT {r['ttft_s_median']:.3f}s | {r['tps_median']:.1f} tok/s | "
            f"keyword {r['keyword_accuracy']:.2%} | refuse-on-negative "
            f"{r['refusal_rate_on_negatives']:.2%} | grounding {r['number_grounding']:.2%}"
        )

    if args.no_rag_control:
        print("\n=== no-RAG control ===")
        payload["runs"]["no_rag"] = benchmarks.evaluate_generation(
            pipeline, questions, repeats=1, use_rag=False
        )
        r = payload["runs"]["no_rag"]
        print(
            f"TTFT {r['ttft_s_median']:.3f}s | {r['tps_median']:.1f} tok/s | "
            f"keyword {r['keyword_accuracy']:.2%}"
        )

    out = RESULTS_DIR / args.output
    benchmarks.save_results(payload, out)
    print(f"\nwritten to {out}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aorus-rag",
        description="Hand-written RAG over the AORUS MASTER 16 AM6H spec sheet.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_model_args(sp, with_llm: bool = True) -> None:
        sp.add_argument("--mode", default="hybrid", choices=("dense", "bm25", "hybrid"))
        if with_llm:
            sp.add_argument(
                "--model", default=DEFAULT_GENERATION_MODEL, choices=sorted(GENERATION_MODELS)
            )
            sp.add_argument("--backend", default="in-process", choices=("in-process", "server"))
            sp.add_argument("--server-url", default="http://127.0.0.1:8080")
            sp.add_argument("--n-ctx", type=int, default=4096)
            sp.add_argument("--kv-type", default="q8_0", choices=("f16", "q8_0", "q5_1", "q4_0"))
            sp.add_argument("--n-gpu-layers", type=int, default=-1)
            sp.add_argument("--temperature", type=float, default=0.2)
            sp.add_argument("--verbose", action="store_true")

    sp = sub.add_parser("fetch", help="download / refresh the cached HTML")
    sp.add_argument("--force", action="store_true", help="re-download even if cached")
    sp.add_argument("--status", action="store_true", help="show cache state only")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("build", help="parse, chunk, embed and index")
    sp.add_argument(
        "--embed-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            f"one of {sorted(EMBEDDING_MODELS)}, or 'hashing' for a "
            "dependency-free lexical fallback used only for smoke tests"
        ),
    )
    sp.add_argument("--embed-gpu-layers", type=int, default=0)
    sp.add_argument("--refresh", action="store_true", help="re-fetch the pages first")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("inspect", help="print the parsed spec table")
    sp.add_argument("--facts", action="store_true", help="print derived atomic facts")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("search", help="retrieval only (no LLM required)")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=5)
    add_model_args(sp, with_llm=False)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="retrieve and generate, streamed")
    sp.add_argument("question")
    sp.add_argument("--top-k", type=int, default=4)
    sp.add_argument("--max-tokens", type=int, default=384)
    sp.add_argument("--no-stream", action="store_true")
    sp.add_argument("--no-rag", action="store_true", help="control condition: no context")
    sp.add_argument("--show-context", action="store_true")
    sp.add_argument("--json", action="store_true")
    add_model_args(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval-retrieval", help="dense vs bm25 vs hybrid (no LLM)")
    sp.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    sp.add_argument("--output", default="retrieval.json")
    sp.set_defaults(func=cmd_eval_retrieval)

    sp = sub.add_parser("bench", help="TTFT / TPS / answer quality")
    sp.add_argument("--repeats", type=int, default=3)
    sp.add_argument("--top-k-sweep", type=int, nargs="+", default=[4])
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--no-rag-control", action="store_true")
    sp.add_argument("--output", default="bench.json")
    sp.add_argument("--top-k", type=int, default=4)
    sp.add_argument("--max-tokens", type=int, default=384)
    add_model_args(sp)
    sp.set_defaults(func=cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        sys.stdout.flush()  # keep the error after whatever progress was printed
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
