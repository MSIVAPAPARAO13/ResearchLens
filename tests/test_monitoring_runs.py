"""Unit tests for MonitoringRun execution with mocked external providers."""

import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

from researchlens.analysis.external_provider import ExternalPaper, UnifiedLiteratureSearchService
from researchlens.analysis.monitoring_service import ResearchMonitoringService
from researchlens.analysis.research_topic import ResearchTopic, TopicStore


@pytest.fixture
def mock_search_service():
    service = MagicMock(spec=UnifiedLiteratureSearchService)
    # Default mock papers
    p1 = ExternalPaper("p1", "Quantum Neural Networks for Chemistry", year=2024, doi="10.1000/qnn", provider="arXiv", citation_count=5)
    p2 = ExternalPaper("p2", "Variational Quantum Algorithms", year=2025, doi="10.1000/vqa", provider="Semantic Scholar", citation_count=12)
    service.search_all.return_value = ([p1, p2], [])
    return service


@pytest.fixture
def temp_store():
    d = tempfile.mkdtemp()
    store = TopicStore(storage_dir=d)
    yield store
    shutil.rmtree(d, ignore_errors=True)


def test_monitoring_service_first_run(mock_search_service, temp_store):
    monitoring = ResearchMonitoringService(
        search_service=mock_search_service,
        topic_store=temp_store,
    )

    topic = ResearchTopic(
        topic_id="top_quantum",
        name="Quantum ML",
        query="quantum machine learning neural networks",
    )

    run, alerts, gap_history = monitoring.execute_monitoring_run(topic)

    assert run.status == "COMPLETED"
    assert run.results_count == 2
    assert run.new_papers_count == 2
    assert len(topic.history) == 2
    assert len(topic.runs) == 1
    assert any(a.type == "NEW_PAPER" for a in alerts)


def test_monitoring_service_second_run_detects_one_new(mock_search_service, temp_store):
    monitoring = ResearchMonitoringService(
        search_service=mock_search_service,
        topic_store=temp_store,
    )

    topic = ResearchTopic(
        topic_id="top_quantum_2",
        name="Quantum ML",
        query="quantum machine learning neural networks",
    )

    # First run (finds p1, p2)
    monitoring.execute_monitoring_run(topic)

    # Second run returns p1 (existing) and p3 (new)
    p1 = ExternalPaper("p1", "Quantum Neural Networks for Chemistry", year=2024, doi="10.1000/qnn", provider="arXiv", citation_count=7)
    p3 = ExternalPaper("p3", "Fault-Tolerant Quantum Learning", year=2026, doi="10.1000/ftq", provider="Crossref", citation_count=0)
    mock_search_service.search_all.return_value = ([p1, p3], [])

    run2, alerts2, _ = monitoring.execute_monitoring_run(topic)

    assert run2.status == "COMPLETED"
    assert run2.results_count == 2
    assert run2.new_papers_count == 1
    assert run2.new_paper_ids == ["p3"]
    assert len(topic.history) == 3


def test_monitoring_service_partial_failure_handling(mock_search_service, temp_store):
    # Simulate one provider failing with warning error while other provides papers
    p1 = ExternalPaper("p1", "Resilient Quantum Study", year=2024, doi="10.1000/res", provider="arXiv")
    mock_search_service.search_all.return_value = ([p1], ["Semantic Scholar 429 Rate Limit"])

    monitoring = ResearchMonitoringService(
        search_service=mock_search_service,
        topic_store=temp_store,
    )

    topic = ResearchTopic("top_res", "Resilience Topic", "quantum")
    run, alerts, _ = monitoring.execute_monitoring_run(topic)

    assert run.status == "PARTIAL"
    assert len(run.errors) == 1
    assert run.results_count == 1
