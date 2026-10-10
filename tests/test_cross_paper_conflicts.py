"""Unit tests for cross-paper conflict detection and comparability checks."""

from researchlens.analysis.conflict_detector import detect_cross_paper_conflicts
from researchlens.analysis.literature_collection import LiteratureCollection, LiteratureDocument
from researchlens.extraction.schema import QuantitativeResultItem, ResearchProfile


def test_detect_result_conflict():
    r1 = QuantitativeResultItem("Accuracy", "94.2%", model="BERT", dataset="SQuAD", page_number=5)
    r2 = QuantitativeResultItem("Accuracy", "88.1%", model="BERT", dataset="SQuAD", page_number=6)

    prof1 = ResearchProfile(document_id="d1", title="Study A", results=[r1])
    prof2 = ResearchProfile(document_id="d2", title="Study B", results=[r2])

    coll = LiteratureCollection(collection_id="c_conf", collection_name="Conflict Test")
    coll.add_document(LiteratureDocument("d1", "a.pdf", "Study A", profile=prof1))
    coll.add_document(LiteratureDocument("d2", "b.pdf", "Study B", profile=prof2))

    conflicts = detect_cross_paper_conflicts(coll)

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.conflict_type == "RESULT_CONFLICT"
    assert c.comparability_status == "Comparable"
    assert c.value_a == "94.2%"
    assert c.value_b == "88.1%"


def test_detect_evaluation_difference():
    r1 = QuantitativeResultItem("Accuracy", "94.2%", model="BERT", dataset="ImageNet", page_number=5)
    r2 = QuantitativeResultItem("Accuracy", "88.1%", model="BERT", dataset="SQuAD", page_number=6)

    prof1 = ResearchProfile(document_id="d1", title="Study A", results=[r1])
    prof2 = ResearchProfile(document_id="d2", title="Study B", results=[r2])

    coll = LiteratureCollection(collection_id="c_diff", collection_name="Diff Test")
    coll.add_document(LiteratureDocument("d1", "a.pdf", "Study A", profile=prof1))
    coll.add_document(LiteratureDocument("d2", "b.pdf", "Study B", profile=prof2))

    conflicts = detect_cross_paper_conflicts(coll)

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.conflict_type == "EVALUATION_DIFFERENCE"
    assert c.comparability_status == "Not directly comparable"
