"""Unit tests for PaperComparisonMatrix, paper matrix generation, and method/dataset tables."""

import pytest

from researchlens.analysis.literature_collection import LiteratureCollection, LiteratureDocument
from researchlens.analysis.paper_matrix import (
    generate_dataset_comparison_table,
    generate_method_comparison_table,
    generate_paper_matrix,
    generate_result_comparison_table,
)
from researchlens.extraction.schema import ExtractedItem, QuantitativeResultItem, ResearchProfile


@pytest.fixture
def sample_collection():
    prof1 = ResearchProfile(
        document_id="d1",
        title="Vision Transformer Paper",
        year=2022,
        models=[ExtractedItem("ViT", page_number=1)],
        datasets=[ExtractedItem("ImageNet", page_number=3)],
        metrics=[ExtractedItem("Accuracy", page_number=5)],
        results=[QuantitativeResultItem("Accuracy", "94.2%", model="ViT", dataset="ImageNet", page_number=5)],
    )
    prof2 = ResearchProfile(
        document_id="d2",
        title="ResNet Baseline Paper",
        year=2020,
        models=[ExtractedItem("ResNet-50", page_number=1)],
        datasets=[ExtractedItem("ImageNet", page_number=4)],
        metrics=[ExtractedItem("Accuracy", page_number=6)],
        results=[QuantitativeResultItem("Accuracy", "91.5%", model="ResNet-50", dataset="ImageNet", page_number=6)],
    )

    coll = LiteratureCollection(collection_id="col_matrix", collection_name="Matrix Test")
    coll.add_document(LiteratureDocument("d1", "vision.pdf", "Vision Transformer Paper", year=2022, profile=prof1))
    coll.add_document(LiteratureDocument("d2", "resnet.pdf", "ResNet Baseline Paper", year=2020, profile=prof2))
    return coll


def test_generate_paper_matrix(sample_collection):
    matrix = generate_paper_matrix(sample_collection)
    table_dict = matrix.to_table_dict()

    assert len(matrix.profiles) == 2
    assert "Paper Title" in table_dict
    assert table_dict["Paper Title"] == ["Vision Transformer Paper", "ResNet Baseline Paper"]


def test_generate_method_and_dataset_tables(sample_collection):
    m_table = generate_method_comparison_table(sample_collection)
    d_table = generate_dataset_comparison_table(sample_collection)

    assert len(m_table) >= 2
    assert any(m["Method / Model"] == "ViT" for m in m_table)
    assert any(m["Method / Model"] == "ResNet-50" for m in m_table)

    assert len(d_table) >= 1
    assert d_table[0]["Dataset"] == "ImageNet"
    assert d_table[0]["Paper Count"] == 2


def test_generate_result_comparison_table(sample_collection):
    res_table = generate_result_comparison_table(sample_collection)

    assert len(res_table) == 1
    row = res_table[0]
    assert row["Comparability Status"] == "Comparable"
    assert "Directly comparable" in row["Rationale"]
