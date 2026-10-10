"""Unit tests for summarization interfaces with mocked model inference."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from researchlens.summarization.summarizer import (
    generate_executive_summary,
    generate_tldr,
    summarize_document,
    summarize_text,
)


@pytest.fixture
def mock_bart_components():
    """Mock the heavy transformer model and tokenizer to enable fast unit tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer output
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2054, 2003, 1037, 102]])
    }
    mock_tokenizer.decode.return_value = "This is a synthesized test summary."

    # Mock model generation output
    mock_model.generate.return_value = torch.tensor([[101, 2054, 2003, 1037, 102]])

    with patch(
        "researchlens.summarization.summarizer.load_model_and_tokenizer",
        return_value=(mock_model, mock_tokenizer, torch.device("cpu")),
    ):
        yield mock_model, mock_tokenizer


def test_summarize_text_mocked(mock_bart_components):
    mock_model, mock_tokenizer = mock_bart_components
    summary = summarize_text("This is an academic text about transformer self-attention.", max_length=50, min_length=20)
    assert summary == "This is a synthesized test summary."
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()


def test_summarize_empty_text():
    assert summarize_text("") == ""
    assert summarize_text("   ") == ""


def test_generate_tldr_mocked(mock_bart_components):
    tldr = generate_tldr("A comprehensive exploration into automated document understanding.")
    assert tldr == "This is a synthesized test summary."


def test_summarize_document_mocked():
    with patch("researchlens.summarization.summarizer.summarize_section", return_value="Section summary"):
        sections = {"abstract": "Abstract text", "results": "Results text"}
        results = summarize_document(sections)
        assert results == {
            "abstract": "Section summary",
            "results": "Section summary",
        }


def test_generate_executive_summary_mocked(mock_bart_components):
    section_summaries = {
        "abstract": "Summary of abstract.",
        "methodology": "Summary of methods.",
        "results": "Summary of results.",
    }
    exec_summary = generate_executive_summary(section_summaries)
    assert exec_summary == "This is a synthesized test summary."
