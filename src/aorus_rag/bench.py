"""Evaluation: retrieval quality, generation quality, and latency.

Metric definitions are spelled out here rather than left implicit, because
"TPS" and "accuracy" both mean several different things in RAG write-ups.

Retrieval (no model required -- runs in milliseconds on CPU)
    recall@k   fraction of questions where any gold document appears in top-k
    mrr        mean reciprocal rank of the first gold document

Generation
    keyword accuracy   every ``must_include`` string present in the answer
    refusal rate       on ``negative`` questions, did the model decline
    number grounding   every number in the answer also appears in the context;
                       an ungrounded number is a hallucination, and this catches
                       it without needing a judge model

Latency
    see llm.StreamStats. Each question is run ``repeats`` times after one
    untimed warm-up, and the median is reported.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import QA_PATH
from .pipeline import RagPipeline
from .retrieve import Retriever

# Numbers worth grounding: 99Wh, 2560, 5.4, 1,000,000. Bare list indices and
# citation markers like "[1]" are excluded by stripping them first.
_CITATION = re.compile(r"\[\d+\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

REFUSAL_MARKERS = (
    "沒有這項資訊",
    "沒有提到",
    "未提供",
    "沒有提供",
    "查無",
    "無法確認",
    "資料中沒有",
    "not in the provided",
    "not provided",
    "does not specify",
    "no information",
    "not mentioned",
    "not listed",
    "cannot determine",
)


@dataclass
class EvalQuestion:
    id: str
    question: str
    lang: str
    type: str
    gold_docs: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_refuse: bool = False


def load_questions(path: Path = QA_PATH) -> list[EvalQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    with path.open(encoding="utf-8") as fh:
        return [EvalQuestion(**json.loads(line)) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[EvalQuestion],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Doc-level recall@k and MRR. Negative questions have no gold, so they are
    excluded here -- they are scored on refusal behaviour instead."""
    scored = [q for q in questions if q.gold_docs]
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    reciprocal: list[float] = []
    latencies: list[float] = []

    per_question = []
    for q in scored:
        t0 = time.perf_counter()
        results = retriever.search(q.question, top_k=max_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        docs = [h.chunk.doc_id for h in results]
        gold = set(q.gold_docs)
        first = next((i for i, d in enumerate(docs) if d in gold), None)
        for k in ks:
            if first is not None and first < k:
                hits_at[k] += 1
        reciprocal.append(1.0 / (first + 1) if first is not None else 0.0)
        per_question.append({"id": q.id, "first_gold_rank": first, "retrieved": docs})

    n = len(scored)
    return {
        "mode": retriever.mode,
        "n_questions": n,
        "recall": {f"@{k}": round(hits_at[k] / n, 4) if n else 0.0 for k in ks},
        "mrr": round(sum(reciprocal) / n, 4) if n else 0.0,
        "latency_ms_median": round(statistics.median(latencies), 3) if latencies else 0.0,
        "per_question": per_question,
    }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(m.lower() in lowered for m in REFUSAL_MARKERS)


def keyword_hit(answer: str, must_include: list[str]) -> bool:
    """All-of semantics, but each entry may be an alternatives group.

    ``["Left", "左"]`` in the eval set means "Left OR 左" for location answers;
    entries are treated as alternatives when they are short location/unit
    synonyms, so the set is written as a flat list and compared case-folded.
    """
    if not must_include:
        return True
    lowered = answer.lower()
    return all(term.lower() in lowered for term in must_include)


def keyword_hit_any(answer: str, must_include: list[str]) -> bool:
    if not must_include:
        return True
    lowered = answer.lower()
    return any(term.lower() in lowered for term in must_include)


def number_grounding(answer: str, context: str) -> tuple[int, int]:
    """(grounded, total) numbers appearing in the answer."""
    body = _CITATION.sub(" ", answer)
    numbers = _NUMBER.findall(body)
    if not numbers:
        return (0, 0)
    grounded = sum(
        1 for n in numbers if n in context or n.replace(",", "") in context.replace(",", "")
    )
    return (grounded, len(numbers))


def evaluate_generation(
    pipeline: RagPipeline,
    questions: list[EvalQuestion],
    repeats: int = 3,
    top_k: int | None = None,
    use_rag: bool = True,
    warmup: bool = True,
) -> dict:
    """Run every question ``repeats`` times, reporting medians."""
    from .prompt import render_context

    if warmup and questions:
        pipeline.answer(questions[0].question, top_k=top_k, use_rag=use_rag)

    rows = []
    for q in questions:
        runs = []
        answer = ""
        context = ""
        for _ in range(repeats):
            result = pipeline.answer(q.question, top_k=top_k, use_rag=use_rag)
            runs.append(result)
            answer = result.answer
            context = render_context(result.hits)

        ttft = statistics.median(r.ttft_s for r in runs)
        tps = statistics.median(r.stats.tps for r in runs)
        e2e = statistics.median(r.stats.e2e_tps for r in runs)
        retrieval_ms = statistics.median(r.retrieval_s * 1000 for r in runs)
        prompt_tokens = runs[-1].stats.prompt_tokens

        grounded, total_numbers = number_grounding(answer, context)
        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "lang": q.lang,
                "question": q.question,
                "answer": answer,
                "ttft_s": round(ttft, 4),
                "tps": round(tps, 2),
                "e2e_tps": round(e2e, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "prompt_tokens": prompt_tokens,
                "context_chars": runs[-1].context_chars,
                "keyword_ok": keyword_hit_any(answer, q.must_include)
                if q.type == "reasoning"
                else keyword_hit(answer, q.must_include),
                "refused": is_refusal(answer),
                "numbers_grounded": grounded,
                "numbers_total": total_numbers,
            }
        )

    answerable = [r for r in rows if r["type"] != "negative"]
    negatives = [r for r in rows if r["type"] == "negative"]
    num_total = sum(r["numbers_total"] for r in answerable)
    num_grounded = sum(r["numbers_grounded"] for r in answerable)

    def med(key: str) -> float:
        vals = [r[key] for r in rows if r[key]]
        return round(statistics.median(vals), 4) if vals else 0.0

    return {
        "backend": pipeline.llm.backend,
        "model": pipeline.llm.name,
        "use_rag": use_rag,
        "repeats": repeats,
        "top_k": top_k or pipeline.cfg.top_k,
        "n_questions": len(rows),
        "keyword_accuracy": round(sum(r["keyword_ok"] for r in answerable) / len(answerable), 4)
        if answerable
        else 0.0,
        "refusal_rate_on_negatives": round(sum(r["refused"] for r in negatives) / len(negatives), 4)
        if negatives
        else 0.0,
        "false_refusal_rate": round(sum(r["refused"] for r in answerable) / len(answerable), 4)
        if answerable
        else 0.0,
        "number_grounding": round(num_grounded / num_total, 4) if num_total else 1.0,
        "ttft_s_median": med("ttft_s"),
        "tps_median": med("tps"),
        "e2e_tps_median": med("e2e_tps"),
        "prompt_tokens_median": med("prompt_tokens"),
        "rows": rows,
    }


def save_results(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
