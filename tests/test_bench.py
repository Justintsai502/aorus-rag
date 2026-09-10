"""Scoring helpers -- these decide the numbers that end up in the README."""

from __future__ import annotations

from aorus_rag.bench import is_refusal, keyword_hit, number_grounding


def test_refusal_detection_both_languages():
    assert is_refusal("提供的規格資料中沒有這項資訊。")
    assert is_refusal("That information is not in the provided specifications.")
    assert not is_refusal("電池容量是 99Wh [1]。")


def test_keyword_hit_requires_every_term():
    assert keyword_hit("解析度是 2560 x 1600", ["2560", "1600"])
    assert not keyword_hit("解析度是 2560", ["2560", "1600"])


def test_keyword_hit_accepts_alternatives_group():
    """A bilingual system may answer "在左側" or "on the Left side"; both count."""
    assert keyword_hit("Thunderbolt 5 在左側 [1]。", [["Left", "左"]])
    assert keyword_hit("It is on the Left side [1].", [["Left", "左"]])
    assert not keyword_hit("它在機身側面 [1]。", [["Left", "左"]])


def test_keyword_hit_mixes_required_and_alternatives():
    spec = [["SO-DIMM", "記憶體"], ["M.2", "SSD"]]
    assert keyword_hit("可以加裝記憶體，也可以加第二顆 SSD [1]。", spec)
    assert not keyword_hit("可以加裝記憶體 [1]。", spec)


def test_number_grounding_flags_invented_numbers():
    context = "[1] 電池容量 / Battery capacity: 99Wh"
    grounded, total = number_grounding("電池是 99Wh [1]", context)
    assert (grounded, total) == (1, 1)  # the citation marker is not counted

    grounded, total = number_grounding("電池是 99Wh，約 26000mAh", context)
    assert total == 2 and grounded == 1  # 26000 is invented


def test_strip_thinking_removes_reasoning_block():
    """Qwen3 emits <think>...</think> even with thinking disabled (empty block).

    Scoring the monologue would credit keywords the user never sees and flag
    numbers the model was only considering as hallucinations.
    """
    from aorus_rag.llm import strip_thinking

    assert strip_thinking("<think>\n\n</think>\n\n電池容量為 99Wh。") == "電池容量為 99Wh。"
    assert strip_thinking("<think>用户问的是…</think>\n答案是 240Hz [1]。") == "答案是 240Hz [1]。"
    assert strip_thinking("沒有 think 的答案 [1]。") == "沒有 think 的答案 [1]。"
