"""Unit tests for information extraction and research gap analysis."""

from researchlens.analysis.research_gaps import extract_research_gaps
from researchlens.extraction.datasets import extract_datasets
from researchlens.extraction.keywords import extract_keywords
from researchlens.extraction.methods import extract_methods
from researchlens.extraction.metrics import extract_metrics


def test_extract_keywords():
    text = (
        "Transformer architectures rely on multi-head attention mechanisms to process "
        "natural language sequences in parallel. Neural language models have achieved "
        "remarkable success across NLP benchmarks."
    )
    keywords = extract_keywords(text, top_k=5)
    assert len(keywords) > 0
    assert keywords != ["Not detected"]


def test_extract_methods():
    text = "We compare a Convolutional Neural Network with a Transformer and LSTM model using Adam optimizer."
    methods = extract_methods(text)
    assert "Transformer" in methods or "Convolutional Neural Network" in methods or "Lstm" in methods


def test_extract_datasets():
    text = "All experiments were evaluated on the ImageNet benchmark and the SQuAD v2.0 dataset."
    datasets = extract_datasets(text)
    assert any("imagenet" in d.lower() or "squad" in d.lower() for d in datasets)


def test_extract_metrics():
    text = "Our proposed architecture achieved an Accuracy of 94.2% and an F1-score of 91.5 on the test set."
    metrics = extract_metrics(text)
    assert "Accuracy" in metrics
    assert any("94.2" in val for val in metrics["Accuracy"])
    assert "F1-Score" in metrics


def test_empty_extractions_return_not_detected():
    empty_text = ""
    assert extract_keywords(empty_text) == ["Not detected"]
    assert extract_methods(empty_text) == ["Not detected"]
    assert extract_datasets(empty_text) == ["Not detected"]
    assert extract_metrics(empty_text) == {"result": ["Not detected"]}


def test_extract_research_gaps():
    text = (
        "A major limitation of our method is the high memory footprint required during training. "
        "For future work, we plan to investigate model quantization and pruning techniques."
    )
    gaps = extract_research_gaps(text)
    lims = gaps["potential_limitations"]
    futures = gaps["potential_future_directions"]

    assert any("limitation" in item.lower() for item in lims)
    assert any("future work" in item.lower() for item in futures)
    assert lims[0].startswith("[Potential Research Gap]")
    assert futures[0].startswith("[Potential Research Direction]")
