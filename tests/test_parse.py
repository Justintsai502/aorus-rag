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


# --------------------------------------------------------------------------
# Contamination guards
#
# BZH / BYH / BXH are SKUs of the AM6H (their URLs 302 to this page), differing
# only in GPU. The desktop comparison widget carries all three, values-only, in
# `div.spec-item-list` blocks. Merging them would put three mutually
# contradictory answers behind one question, so these guards get direct tests
# rather than relying on the happy path staying happy.
# --------------------------------------------------------------------------

CONTAMINANT_GPUS = ("5080", "5070")


def test_corpus_mentions_exactly_one_gpu(tables):
    """The most direct guard: the comparison widget's GPU values name no model,
    so the SIBLING_MARKERS check alone would not catch a parser regression."""
    zh, _ = tables
    blob = "\n".join(i.value for i in zh)
    assert "5090" in blob
    for gpu in CONTAMINANT_GPUS:
        assert gpu not in blob, f"RTX {gpu} leaked in from the comparison widget"


def test_comparison_widget_is_present_but_excluded():
    """Guards the guard: if the widget ever disappears from the page, this test
    fails and tells us the exclusion logic is no longer being exercised."""
    html = fetch.load_cached("spec_zh")
    assert html.count('<div class="spec-item-list"') == 51  # 17 rows x 3 SKUs
    assert html.count('<ul class="spec-item-list"') == 17  # AM6H itself
    assert "5080" in html and "5070" in html  # the contaminants are really there


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("extra row", lambda xs: xs + [xs[0]]),
        ("empty value", lambda xs: [parse.SpecItem(0, xs[0].key, [], [])] + xs[1:]),
        ("missing key", lambda xs: [parse.SpecItem(0, "", ["x"], [])] + xs[1:]),
        (
            "sku marker",
            lambda xs: [parse.SpecItem(0, xs[0].key, ["AORUS MASTER 16 BXH"], [])] + xs[1:],
        ),
    ],
)
def test_validation_rejects_contaminated_tables(tables, label, mutate):
    zh, _ = tables
    with pytest.raises(ValueError):
        parse.validate_spec_items(mutate(list(zh)), label)


def test_sku_variants_are_paired_with_their_model_code():
    """BZH/BYH/BXH are SKUs of the AM6H differing only in GPU. Their values live
    in unlabelled div blocks; the model names live in a separate subtitle. The
    parser must pair them by column order, or a value lands on the wrong model."""
    zh = parse.parse_spec_table(fetch.load_cached("spec_zh"))
    variants, differing = parse.parse_sku_variants(fetch.load_cached("spec_zh"))

    assert [v.code for v in variants] == ["BZH", "BYH", "BXH"]
    # Only the GPU row differs; the other sixteen are identical across SKUs.
    assert [zh[i].key for i in differing] == ["顯示晶片"]

    gpus = {v.code: v.values[differing[0]] for v in variants}
    assert "5090" in gpus["BZH"]
    assert "5080" in gpus["BYH"]
    assert "5070 Ti" in gpus["BXH"]


def test_sku_chunks_bind_every_value_to_a_model_code():
    from aorus_rag import chunk as chunking

    zh = parse.parse_spec_table(fetch.load_cached("spec_zh"))
    en = parse.parse_spec_table(fetch.load_cached("spec_en"))
    variants, differing = parse.parse_sku_variants(fetch.load_cached("spec_zh"))
    chunks = chunking.chunk_sku_variants(variants, differing, zh, en)

    assert len(chunks) == 1 + len(variants) * len(differing)
    for c in chunks:
        if c.chunk_id == "sku.overview":
            continue
        # The whole point: a GPU value never appears without its model code.
        assert c.meta["sku"] in c.text
