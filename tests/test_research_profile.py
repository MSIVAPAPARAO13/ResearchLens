"""Unit tests for ResearchProfile schema, ExtractedItem data structures, and serialization."""

from researchlens.extraction.schema import (
    BaselineComparison,
    ExtractedItem,
    QuantitativeResultItem,
    ResearchProfile,
)


def test_extracted_item_serialization():
    item = ExtractedItem(
        value="SciTransformer",
        source_text="We propose SciTransformer for document intelligence.",
        page_number=3,
        section_name="methodology",
        chunk_id="c_01",
        confidence=0.92,
        confidence_level="High",
        extraction_method="pattern",
    )
    d = item.to_dict()
    assert d["value"] == "SciTransformer"
    assert d["page_number"] == 3
    assert d["section_name"] == "methodology"
    assert d["chunk_id"] == "c_01"
    assert d["confidence"] == 0.92
    assert d["confidence_level"] == "High"


def test_quantitative_result_serialization():
    res = QuantitativeResultItem(
        metric="Accuracy",
        value="94.2%",
        unit="%",
        source_text="Model achieved 94.2% accuracy.",
        page_number=6,
        section_name="results",
    )
    d = res.to_dict()
    assert d["metric"] == "Accuracy"
    assert d["value"] == "94.2%"
    assert d["unit"] == "%"


def test_baseline_comparison_serialization():
    comp = BaselineComparison(
        baseline_model="ResNet-50",
        proposed_model="SciTransformer",
        difference="+4.2%",
        source_text="Outperformed ResNet-50 by 4.2%.",
        page_number=7,
        section_name="results",
    )
    d = comp.to_dict()
    assert d["baseline_model"] == "ResNet-50"
    assert d["difference"] == "+4.2%"


def test_research_profile_to_dict():
    profile = ResearchProfile(
        document_id="doc_123",
        title="Sample Paper Title",
        year="2023",
        models=[ExtractedItem(value="BERT", page_number=2)],
    )
    d = profile.to_dict()
    assert d["document_id"] == "doc_123"
    assert d["title"] == "Sample Paper Title"
    assert d["year"] == "2023"
    assert len(d["models"]) == 1
    assert d["models"][0]["value"] == "BERT"
