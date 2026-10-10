"""Unit tests for baseline model and comparative finding extraction."""

import pytest

from researchlens.extraction.metrics import extract_metrics_results_baselines
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def baseline_document():
    chunk = ChunkData(
        chunk_id="c_b1",
        text="Our proposed model outperformed ResNet-50 by 4.2% on top-1 accuracy compared with SVM baseline.",
        token_count=16,
        section_name="results",
        page_number=6,
    )
    return Document(filename="baseline_paper.pdf", full_text=chunk.text, chunks=[chunk])


def test_extract_baseline_comparisons(baseline_document):
    _, _, baselines = extract_metrics_results_baselines(baseline_document)

    assert len(baselines) > 0
    b_models = [b.baseline_model.lower() for b in baselines]

    assert any("resnet" in bm for bm in b_models) or any("svm" in bm for bm in b_models)
    assert baselines[0].page_number == 6
    assert baselines[0].chunk_id == "c_b1"
    assert baselines[0].source_text != ""


def test_baseline_empty():
    doc = Document(filename="nobaseline.pdf", full_text="A study of algorithmic complexity.", chunks=[])
    _, _, baselines = extract_metrics_results_baselines(doc)
    assert len(baselines) == 0
