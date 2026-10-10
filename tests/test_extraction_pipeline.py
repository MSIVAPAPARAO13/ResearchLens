"""Unit tests for ExtractionPipeline end-to-end execution and validation."""

import pytest

from researchlens.extraction.pipeline import ExtractionPipeline, extract_research_profile
from researchlens.extraction.schema import ResearchProfile
from researchlens.ingestion.document import ChunkData, Document, DocumentMetadata, Page


@pytest.fixture
def full_paper_document():
    page1 = Page(
        page_number=1,
        text="""SciTransformer: Multi-Scale Transformer for Document Intelligence
John Doe, Jane Smith
Department of Computer Science, Stanford University

Abstract:
Deep learning models for scientific document intelligence face significant challenges with domain-specific terminology. In this paper, we address the problem of automated scientific information extraction. We propose SciTransformer, a multi-scale self-attention transformer architecture.
""",
    )
    chunk_m = ChunkData(
        chunk_id="chunk_m",
        text="SciTransformer uses AdamW optimizer with cosine learning rate schedule on ImageNet-1K dataset.",
        token_count=16,
        section_name="methodology",
        page_number=3,
    )
    chunk_r = ChunkData(
        chunk_id="chunk_r",
        text="The model achieved 94.2% Accuracy and test RMSE of 0.284, outperforming ResNet-50 by 4.2%.",
        token_count=17,
        section_name="results",
        page_number=6,
    )
    chunk_d = ChunkData(
        chunk_id="chunk_d",
        text="A primary limitation of our approach is the quadratic memory footprint during long-sequence training. In the future, we plan to extend evaluation to multilingual papers.",
        token_count=24,
        section_name="discussion",
        page_number=8,
    )
    return Document(
        filename="scitransformer_paper.pdf",
        full_text=page1.text + "\n" + chunk_m.text + "\n" + chunk_r.text + "\n" + chunk_d.text,
        pages=[page1],
        chunks=[chunk_m, chunk_r, chunk_d],
        metadata=DocumentMetadata(title="SciTransformer: Multi-Scale Transformer for Document Intelligence"),
    )


def test_extraction_pipeline_execution(full_paper_document):
    pipeline = ExtractionPipeline()
    profile = pipeline.extract_profile(full_paper_document)

    assert isinstance(profile, ResearchProfile)
    assert profile.title == "SciTransformer: Multi-Scale Transformer for Document Intelligence"
    assert len(profile.problem) > 0
    assert len(profile.objective) > 0

    # Models & Datasets
    model_vals = [m.value for m in profile.models]
    dataset_vals = [d.value for d in profile.datasets]
    assert any("scitransformer" in m.lower() or "resnet" in m.lower() for m in model_vals)
    assert any("imagenet" in d.lower() for d in dataset_vals)

    # Results
    assert len(profile.results) > 0
    res_metrics = [r.metric.lower() for r in profile.results]
    assert any("accuracy" in m or "rmse" in m for m in res_metrics)

    # Limitations & Future work
    assert len(profile.limitations) > 0
    assert len(profile.future_work) > 0
    assert "quadratic memory" in profile.limitations[0].value.lower()
    assert "multilingual" in profile.future_work[0].value.lower()

    # Provenance
    assert profile.limitations[0].page_number == 8
    assert profile.limitations[0].chunk_id == "chunk_d"


def test_extract_research_profile_convenience(full_paper_document):
    profile = extract_research_profile(full_paper_document)
    assert isinstance(profile, ResearchProfile)
    assert profile.title != ""


def test_empty_document_pipeline_fallback():
    doc = Document(filename="empty.pdf", full_text="", chunks=[])
    profile = extract_research_profile(doc)

    assert profile.title == "Not detected"
    assert profile.problem[0].value == "Not detected"
    assert profile.research_questions[0].value == "Not explicitly stated"
    assert profile.models[0].value == "Not detected"
    assert profile.datasets[0].value == "Not detected"
    assert profile.limitations[0].value == "Not detected"
