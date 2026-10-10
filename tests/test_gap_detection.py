"""Unit tests for ResearchGapEngine gap detection pipelines."""

import pytest

from researchlens.analysis.gap_engine import ResearchGapEngine
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


@pytest.fixture
def sample_profile():
    lim = ExtractedItem(
        value="A primary limitation of our approach is high memory footprint during long sequence training.",
        source_text="A primary limitation of our approach is high memory footprint during long sequence training.",
        page_number=6,
        section_name="discussion",
        chunk_id="chunk_6",
    )
    fw = ExtractedItem(
        value="In the future, we plan to extend evaluation to multilingual scientific corpora.",
        source_text="In the future, we plan to extend evaluation to multilingual scientific corpora.",
        page_number=8,
        section_name="conclusion",
        chunk_id="chunk_8",
    )
    ds = ExtractedItem(value="SQuAD v2.0", source_text="Evaluated on SQuAD v2.0.", page_number=4, section_name="experiments")
    mod = ExtractedItem(value="SciTransformer", source_text="We propose SciTransformer.", page_number=1, section_name="abstract")

    return ResearchProfile(
        document_id="doc_123",
        title="SciTransformer Paper",
        models=[mod],
        datasets=[ds],
        limitations=[lim],
        future_work=[fw],
    )


def test_gap_detection_explicit_limitations(sample_profile):
    engine = ResearchGapEngine(ablation_mode="full")
    gaps, hypotheses = engine.analyze_profile(sample_profile)

    assert len(gaps) > 0
    lim_gaps = [g for g in gaps if g.gap_type == "EXPLICIT_LIMITATION"]
    assert len(lim_gaps) == 1
    assert lim_gaps[0].explicit_or_inferred == "EXPLICIT"
    assert "memory footprint" in lim_gaps[0].statement.lower()
    assert lim_gaps[0].source_pages == [6]


def test_gap_detection_explicit_future_work(sample_profile):
    engine = ResearchGapEngine(ablation_mode="full")
    gaps, _ = engine.analyze_profile(sample_profile)

    fw_gaps = [g for g in gaps if g.gap_type == "EXPLICIT_FUTURE_WORK"]
    assert len(fw_gaps) == 1
    assert fw_gaps[0].explicit_or_inferred == "EXPLICIT"
    assert "multilingual" in fw_gaps[0].statement.lower()


def test_gap_detection_single_dataset(sample_profile):
    engine = ResearchGapEngine(ablation_mode="full")
    gaps, _ = engine.analyze_profile(sample_profile)

    ds_gaps = [g for g in gaps if g.gap_type == "UNDEREXPLORED_DATASET"]
    assert len(ds_gaps) == 1
    assert ds_gaps[0].explicit_or_inferred == "INFERRED"
    assert ds_gaps[0].category == "GENERALIZATION"
    assert "SQuAD" in ds_gaps[0].statement


def test_empty_profile_refusal():
    empty_profile = ResearchProfile(document_id="empty_doc", title="Empty Document")
    engine = ResearchGapEngine(ablation_mode="full")
    gaps, hypotheses = engine.analyze_profile(empty_profile)

    assert len(gaps) == 1
    assert gaps[0].gap_type == "REFRUSAL"
    assert gaps[0].statement == "Insufficient evidence to identify a reliable research gap."
    assert gaps[0].confidence_score == 0.0
    assert len(hypotheses) == 0
