"""Unit tests for dataset and benchmark corpus extraction."""

import pytest

from researchlens.extraction.datasets import extract_dataset_items, extract_datasets
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def dataset_document():
    chunk1 = ChunkData(
        chunk_id="c_ds1",
        text="All experimental models were trained on the ImageNet-1K benchmark dataset.",
        token_count=12,
        section_name="experiments",
        page_number=5,
    )
    chunk2 = ChunkData(
        chunk_id="c_ds2",
        text="Evaluated on the SQuAD v2.0 dataset for question answering.",
        token_count=10,
        section_name="experiments",
        page_number=6,
    )
    return Document(
        filename="dataset_paper.pdf",
        full_text=chunk1.text + "\n" + chunk2.text,
        chunks=[chunk1, chunk2],
    )


def test_extract_dataset_items_provenance(dataset_document):
    items = extract_dataset_items(dataset_document)
    vals = [item.value for item in items]

    assert any("ImageNet" in v for v in vals)
    assert any("SQuAD" in v for v in vals)

    first = items[0]
    assert first.page_number == 5
    assert first.section_name == "experiments"
    assert first.chunk_id == "c_ds1"
    assert first.confidence > 0.8


def test_extract_datasets_string_compatibility(dataset_document):
    result = extract_datasets(dataset_document.full_text)
    assert isinstance(result, list)
    assert any("ImageNet" in r for r in result)


def test_extract_datasets_missing():
    doc = Document(filename="nodata.pdf", full_text="Theoretical study of quantum physics.", chunks=[])
    items = extract_dataset_items(doc)
    assert items[0].value == "Not detected"
    assert items[0].confidence == 0.0
