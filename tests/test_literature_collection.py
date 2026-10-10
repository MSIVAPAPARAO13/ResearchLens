"""Unit tests for LiteratureCollection and LiteratureDocument dataclasses."""

from researchlens.analysis.literature_collection import (
    LiteratureCollection,
    LiteratureDocument,
    generate_deterministic_document_id,
)
from researchlens.extraction.schema import ResearchProfile


def test_deterministic_document_id():
    id1 = generate_deterministic_document_id("paper1.pdf", "Content text ABC")
    id2 = generate_deterministic_document_id("paper1.pdf", "Content text ABC")
    id3 = generate_deterministic_document_id("paper2.pdf", "Content text XYZ")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("doc_")


def test_literature_collection_management():
    coll = LiteratureCollection(collection_id="col_1", collection_name="Test Collection")
    prof1 = ResearchProfile(document_id="d1", title="Paper One", year=2023)
    doc1 = LiteratureDocument(document_id="d1", filename="p1.pdf", title="Paper One", year=2023, profile=prof1)

    coll.add_document(doc1)
    assert coll.paper_count == 1
    assert len(coll.profiles) == 1
    assert coll.get_document("d1") == doc1

    # Adding duplicate ID should not duplicate
    coll.add_document(doc1)
    assert coll.paper_count == 1

    d_dict = coll.to_dict()
    assert d_dict["collection_id"] == "col_1"
    assert d_dict["paper_count"] == 1
