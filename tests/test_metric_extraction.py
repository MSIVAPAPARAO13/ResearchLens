"""Unit tests for metric name extraction."""

import pytest

from researchlens.extraction.metrics import extract_metrics_results_baselines
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def metric_document():
    chunk = ChunkData(
        chunk_id="c_m1",
        text="The model performance was measured using Accuracy, F1-Score, ROUGE-L, and BLEU-4 metrics.",
        token_count=15,
        section_name="experiments",
        page_number=4,
    )
    return Document(filename="metric_paper.pdf", full_text=chunk.text, chunks=[chunk])


def test_extract_metric_names(metric_document):
    metrics, _, _ = extract_metrics_results_baselines(metric_document)
    vals = [m.value for m in metrics]

    assert any("Accuracy" in v for v in vals)
    assert any("F1" in v for v in vals)
    assert any("Rouge" in v or "ROUGE" in v for v in vals)
    assert any("Bleu" in v or "BLEU" in v for v in vals)

    assert metrics[0].page_number == 4
    assert metrics[0].chunk_id == "c_m1"


def test_extract_metrics_missing():
    doc = Document(filename="empty.pdf", full_text="Qualitative discussion without quantitative evaluation.", chunks=[])
    metrics, _, _ = extract_metrics_results_baselines(doc)
    assert metrics[0].value == "Not detected"
    assert metrics[0].confidence == 0.0
