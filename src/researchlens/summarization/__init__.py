"""Summarization models and pipelines."""

from researchlens.summarization.hierarchical import (
    summarize_section,
    summarize_section_hierarchical,
)
from researchlens.summarization.model import (
    get_device,
    get_model,
    get_tokenizer,
    load_model_and_tokenizer,
)
from researchlens.summarization.summarizer import (
    generate_executive_summary,
    generate_tldr,
    summarize_document,
    summarize_text,
)

__all__ = [
    "get_device",
    "get_model",
    "get_tokenizer",
    "load_model_and_tokenizer",
    "summarize_text",
    "summarize_section",
    "summarize_section_hierarchical",
    "summarize_document",
    "generate_tldr",
    "generate_executive_summary",
]
