"""Unit tests for scientific methods, models, and algorithms extraction."""

import pytest

from researchlens.extraction.methods import (
    extract_methods,
    extract_models_methods_algorithms,
)
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def method_document():
    chunk1 = ChunkData(
        chunk_id="c1",
        text="We propose ResNet-50 architecture combined with multi-head attention mechanisms and fine-tuning.",
        token_count=16,
        section_name="methodology",
        page_number=3,
    )
    chunk2 = ChunkData(
        chunk_id="c2",
        text="The model was trained using AdamW optimizer with cosine learning rate schedule.",
        token_count=14,
        section_name="methodology",
        page_number=4,
    )
    return Document(
        filename="method.pdf",
        full_text=chunk1.text + "\n" + chunk2.text,
        chunks=[chunk1, chunk2],
    )


def test_extract_models_methods_algorithms(method_document):
    models, methods, algorithms = extract_models_methods_algorithms(method_document)

    model_vals = [m.value for m in models]
    method_vals = [m.value for m in methods]
    alg_vals = [a.value for a in algorithms]

    assert any("Resnet" in m or "RESNET" in m for m in model_vals)
    assert any("Fine-Tuning" in m for m in method_vals)
    assert any("Adamw" in a or "ADAMW" in a or "Attention" in a for a in alg_vals)

    # Check provenance preservation
    assert models[0].page_number is not None
    assert models[0].chunk_id is not None
    assert models[0].section_name == "methodology"


def test_extract_methods_backward_compatibility(method_document):
    str_list = extract_methods(method_document.full_text)
    assert isinstance(str_list, list)
    assert len(str_list) > 0
    assert "Resnet-50" in str_list or "Fine-Tuning" in str_list or "Adamw" in str_list


def test_extract_methods_empty_input():
    doc = Document(filename="empty.pdf", full_text="", chunks=[])
    models, methods, algorithms = extract_models_methods_algorithms(doc)
    assert models[0].value == "Not detected"
    assert methods[0].value == "Not detected"
    assert algorithms[0].value == "Not detected"
