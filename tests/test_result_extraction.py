"""Unit tests for quantitative result relationship extraction."""

import pytest

from researchlens.extraction.metrics import extract_metrics_results_baselines
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def result_document():
    chunk = ChunkData(
        chunk_id="c_r1",
        text="On the classification benchmark, our proposed model achieved Accuracy of 94.2% and F1-score of 0.91.",
        token_count=18,
        section_name="results",
        page_number=7,
    )
    return Document(filename="result_paper.pdf", full_text=chunk.text, chunks=[chunk])


def test_extract_quantitative_results(result_document):
    _, results, _ = extract_metrics_results_baselines(result_document)

    assert len(results) >= 2
    r_map = {r.metric.lower(): r.value for r in results}

    assert "accuracy" in r_map
    assert "94.2%" in r_map["accuracy"] or "94.2" in r_map["accuracy"]

    assert "f1" in r_map or "f1-score" in r_map
    assert any("0.91" in val for k, val in r_map.items() if "f1" in k)

    first = results[0]
    assert first.page_number == 7
    assert first.chunk_id == "c_r1"
    assert first.source_text != ""


def test_extract_results_empty():
    doc = Document(filename="empty.pdf", full_text="", chunks=[])
    _, results, _ = extract_metrics_results_baselines(doc)
    assert len(results) == 0
