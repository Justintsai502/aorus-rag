"""Index, tokenizer and fusion behaviour. No model download required."""

from __future__ import annotations

import numpy as np
import pytest

from aorus_rag.chunk import Chunk
from aorus_rag.embed import HashingEmbedder
from aorus_rag.index import BM25Index, VectorIndex, rrf_fuse, tokenize
from aorus_rag.prompt import build_prompt, detect_language
from aorus_rag.retrieve import Retriever, embed_corpus


def test_tokenizer_keeps_alphanumeric_identifiers():
    tokens = tokenize("1 x Type-C with Thunderbolt™5, USB3.2 Gen2")
    assert "type-c" in tokens
    assert "usb3.2" in tokens


def test_tokenizer_emits_cjk_unigrams_and_bigrams():
    tokens = tokenize("螢幕更新率")
    assert "螢幕" in tokens  # bigram
    assert "更新" in tokens
    assert "螢" in tokens  # unigram


def test_vector_index_is_cosine():
    m = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    idx = VectorIndex(m)
    (top_i, top_s), *_ = idx.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert top_i == 0
    assert top_s == pytest.approx(1.0, abs=1e-5)


def test_vector_batch_matches_single():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(20, 8)).astype(np.float32)
    idx = VectorIndex(m)
    queries = rng.normal(size=(3, 8)).astype(np.float32)
    batched = idx.search_batch(queries, top_k=5)
    for q, expected in zip(queries, batched):
        single = idx.search(q, top_k=5)
        assert [i for i, _ in single] == [i for i, _ in expected]
        # GEMV and GEMM accumulate in a different order, so float32 scores
        # differ in the last couple of bits.
        for (_, a), (_, b) in zip(single, expected):
            assert a == pytest.approx(b, abs=1e-6)


def test_bm25_ranks_exact_term_first():
    docs = [tokenize(t) for t in ["電池 Li-ion 99Wh", "變壓器 330W", "螢幕 240Hz"]]
    bm25 = BM25Index(docs)
    assert bm25.search("99Wh", top_k=1)[0][0] == 0


def test_rrf_prefers_agreement_across_rankings():
    dense = [(5, 0.9), (1, 0.8)]
    sparse = [(1, 12.0), (9, 3.0)]
    fused = rrf_fuse([dense, sparse])
    assert fused[0][0] == 1  # ranked by both


def _toy_corpus() -> list[Chunk]:
    return [
        Chunk(
            "c1",
            "spec.battery",
            "fact",
            "電池容量 / Battery capacity: 99Wh",
            key_zh="電池容量",
            key_en="Battery capacity",
        ),
        Chunk(
            "c2",
            "spec.adapter",
            "fact",
            "變壓器功率 / Adapter power: 330W",
            key_zh="變壓器功率",
            key_en="Adapter power",
        ),
        Chunk(
            "c3",
            "spec.display",
            "fact",
            "螢幕更新率 / Display refresh rate: 240Hz",
            key_zh="螢幕更新率",
            key_en="Display refresh rate",
        ),
    ]


def test_retriever_finds_the_right_row_in_both_languages():
    chunks = _toy_corpus()
    emb = HashingEmbedder()
    bundle = embed_corpus(chunks, emb)
    r = Retriever(chunks, bundle, emb, mode="hybrid")
    assert r.search("電池容量是多少", top_k=1)[0].chunk.chunk_id == "c1"
    assert r.search("What is the adapter power?", top_k=1)[0].chunk.chunk_id == "c2"


def test_retriever_caps_chunks_per_document():
    chunks = [
        Chunk(f"d{i}", "spec.display", "spec_line", f"顯示器 / Display: line {i}") for i in range(5)
    ]
    emb = HashingEmbedder()
    r = Retriever(chunks, embed_corpus(chunks, emb), emb, mode="hybrid", max_per_doc=2)
    assert len(r.search("顯示器", top_k=5)) <= 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What is the battery capacity?", "en"),
        ("這台電池多大", "zh"),
        ("What's the 電池 capacity?", "en"),  # a couple of CJK nouns -> still English
        ("這台的 refresh rate 是多少", "zh"),
    ],
)
def test_language_detection(text, expected):
    assert detect_language(text) == expected


def test_prompt_trims_context_to_budget():
    chunks = _toy_corpus()
    emb = HashingEmbedder()
    r = Retriever(chunks, embed_corpus(chunks, emb), emb, mode="bm25")
    hits = r.search("電池", top_k=3)
    built = build_prompt("電池多大", hits, max_context_chars=30)
    assert built.context_chars <= 30
    assert len(built.used_hits) < len(hits)


def test_bm25_mode_needs_neither_index_nor_embedder():
    """The zero-dependency path must stay zero-dependency: passing no bundle and
    no embedder has to work, or `search --mode bm25` silently starts requiring a
    model download."""
    chunks = _toy_corpus()
    r = Retriever(chunks, None, None, mode="bm25")
    assert r.search("99Wh", top_k=1)[0].chunk.chunk_id == "c1"


def test_rescue_keeps_a_single_retriever_top_hit():
    """RRF scores consensus, so a chunk only one arm finds can be dropped.

    Measured case: the row holding "VESA DisplayHDR True Black 500" was dense's
    #1 and absent from BM25's list, so fusion pushed it to rank 8 and the answer
    became a refusal. The rescue must put it back inside top-k.
    """
    chunks = [Chunk(f"c{i}", f"d{i}", "spec_line", f"顯示器 / Display: 項目 {i}") for i in range(8)]
    chunks.append(
        Chunk("gold", "dgold", "spec_row", "顯示器 / Display: VESA DisplayHDR True Black 500")
    )
    emb = HashingEmbedder()
    r = Retriever(chunks, embed_corpus(chunks, emb), emb, mode="hybrid")
    # Force the situation: gold is dense's top hit, unseen by BM25.
    dense = [(8, 0.9)] + [(i, 0.5) for i in range(4)]
    sparse = [(0, 9.0), (1, 8.0), (2, 7.0)]
    selected = r._rescue_top_hits([(0, 0.03), (1, 0.02), (2, 0.01)], [dense, sparse], top_k=3)
    assert 8 in [i for i, _ in selected]
    assert len(selected) <= 3
