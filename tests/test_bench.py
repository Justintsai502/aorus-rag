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


def test_number_grounding_flags_invented_numbers():
    context = "[1] 電池容量 / Battery capacity: 99Wh"
    grounded, total = number_grounding("電池是 99Wh [1]", context)
    assert (grounded, total) == (1, 1)  # the citation marker is not counted

    grounded, total = number_grounding("電池是 99Wh，約 26000mAh", context)
    assert total == 2 and grounded == 1  # 26000 is invented
