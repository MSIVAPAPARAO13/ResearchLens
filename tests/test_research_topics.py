"""Unit tests for ResearchTopic, MonitoringConfig, PaperHistoryEntry, and TopicStore."""

import shutil
import tempfile

import pytest

from researchlens.analysis.research_topic import (
    PaperHistoryEntry,
    ResearchTopic,
    TopicStore,
)


@pytest.fixture
def temp_topic_store():
    temp_dir = tempfile.mkdtemp()
    store = TopicStore(storage_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_research_topic_creation_and_defaults():
    topic = ResearchTopic(
        topic_id="top_test_01",
        name="AI-based Alzheimer MRI diagnosis",
        query="Alzheimer MRI deep learning diagnosis",
        description="Tracking MRI architectures and datasets.",
        providers=["arXiv", "Semantic Scholar"],
        year_from=2022,
        year_to=2026,
        max_results=30,
    )

    assert topic.topic_id == "top_test_01"
    assert topic.name == "AI-based Alzheimer MRI diagnosis"
    assert topic.config.search_query == "Alzheimer MRI deep learning diagnosis"
    assert topic.config.max_results == 30
    assert topic.active is True
    assert len(topic.history) == 0
    assert len(topic.runs) == 0


def test_research_topic_serialization():
    topic = ResearchTopic(
        topic_id="top_ser_01",
        name="Graph Neural Networks for Drug Discovery",
        query="GNN molecular drug property prediction",
    )
    topic.history["p1"] = PaperHistoryEntry(
        paper_id="p1",
        first_seen="2026-10-01T00:00:00Z",
        last_seen="2026-10-08T00:00:00Z",
        providers=["arXiv"],
        title="GNN Drug Screening",
        authors=["Alice Doe"],
        year=2025,
        doi="10.1000/182",
        arxiv_id="2501.0001",
        citation_count=12,
    )

    d = topic.to_dict()
    assert d["topic_id"] == "top_ser_01"
    assert "p1" in d["history"]
    assert d["history"]["p1"]["title"] == "GNN Drug Screening"

    restored = ResearchTopic.from_dict(d)
    assert restored.topic_id == "top_ser_01"
    assert len(restored.history) == 1
    assert restored.history["p1"].doi == "10.1000/182"


def test_topic_store_crud_lifecycle(temp_topic_store):
    store = temp_topic_store

    # 1. Empty state
    assert len(store.list_topics()) == 0

    # 2. Save topic
    topic1 = ResearchTopic(
        topic_id="t1",
        name="Topic 1",
        query="Query 1",
    )
    topic2 = ResearchTopic(
        topic_id="t2",
        name="Topic 2",
        query="Query 2",
    )
    store.save_topic(topic1)
    store.save_topic(topic2)

    # 3. List
    topics = store.list_topics()
    assert len(topics) == 2
    ids = {t.topic_id for t in topics}
    assert "t1" in ids and "t2" in ids

    # 4. Load single
    loaded = store.load_topic("t1")
    assert loaded is not None
    assert loaded.name == "Topic 1"

    # 5. Delete
    deleted = store.delete_topic("t1")
    assert deleted is True
    assert store.load_topic("t1") is None
    assert len(store.list_topics()) == 1
