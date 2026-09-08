"""The parser is the only place where a silent error produces confident wrong
answers, so it gets the strictest tests."""

from __future__ import annotations

import pytest

from aorus_rag import fetch, normalize, parse

pytestmark = pytest.mark.skipif(
    not (fetch.RAW_DIR / "spec_zh.html").exists(),
    reason="cached HTML missing; run `uv run aorus-rag fetch`",
)


@pytest.fixture(scope="module")
def tables():
    zh = parse.parse_spec_table(fetch.load_cached("spec_zh"))
    en = parse.parse_spec_table(fetch.load_cached("spec_en"))
    return zh, en


def test_row_count_and_alignment(tables):
    zh, en = tables
    parse.validate_spec_items(zh, "spec_zh")
    parse.validate_spec_items(en, "spec_en")
    assert len(zh) == len(en) == parse.EXPECTED_SPEC_ROWS


def test_keys_are_translated_but_values_are_not(tables):
    """The zh and en pages share identical values; only keys differ.

    This is what makes the bilingual key anchor in chunk.py necessary -- a
    Chinese question has nothing Chinese to match against in the values.
    """
    zh, en = tables
    assert all(z.value == e.value for z, e in zip(zh, en))
    assert zh[1].key == "中央處理器" and en[1].key == "CPU"


def test_no_sibling_model_contamination(tables):
    """The page also carries BZH/BYH/BXH specs in a comparison widget."""
    zh, _ = tables
    blob = "\n".join(i.value for i in zh)
    for marker in parse.SIBLING_MARKERS:
        assert marker not in blob


def test_footnotes_are_separated_from_values(tables):
    zh, _ = tables
    gpu = next(i for i in zh if i.key == "顯示晶片")
    assert any("May vary by scenario" in n for n in gpu.footnotes)
    assert all("May vary by scenario" not in line for line in gpu.lines)


def test_known_values_survive_parsing(tables):
    _, en = tables
    by_key = {i.key: i.value for i in en}
    assert "99Wh" in by_key["Battery"]
    assert "2560" in by_key["Display"] and "240Hz" in by_key["Display"]
    assert "275HX" in by_key["CPU"]


def test_atomic_facts(tables):
    zh, en = tables
    facts = {f.fact_id: f.value for f in normalize.extract_facts(zh, en)}
    assert facts["battery.capacity"] == "99Wh"
    assert facts["display.refresh_rate"] == "240Hz"
    assert facts["display.contrast"] == "1,000,000:1"  # not "16:1" from "16:10"
    assert facts["adapter.power"] == "330W"
    assert facts["io.count.usb_c"] == "2"
    assert "Left" in facts["io.side.thunderbolt5"]
    assert "Right" in facts["io.side.thunderbolt4"]


def test_feature_page_excludes_site_navigation():
    blocks = parse.parse_feature_page(fetch.load_cached("feature_zh"))
    text = " ".join(p for b in blocks for p in b.paragraphs)
    assert "WINDFORCE" in text
    assert "主機板" not in text  # global nav menu label
