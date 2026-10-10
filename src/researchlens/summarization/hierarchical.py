"""Hierarchical two-level summarization pipeline for long research paper sections."""

from typing import Dict, Optional

from researchlens.config.settings import get_settings
from researchlens.preprocessing.chunker import chunk_text
from researchlens.preprocessing.cleaner import clean_section_text, filter_sentences_by_section
from researchlens.utils.logging import get_logger

logger = get_logger("hierarchical")


def summarize_section_hierarchical(
    section_name: str,
    section_text: str,
    custom_config: Optional[Dict[str, int]] = None,
) -> str:
    """Generate a synthesized section summary via a two-level hierarchical pipeline.

    1. Clean academic noise from raw text.
    2. Extractive sentence scoring by section-specific keywords.
    3. Token-aware text chunking.
    4. First-level summarization of individual chunks.
    5. Second-level synthesis of combined chunk summaries.
    """
    from researchlens.summarization.summarizer import summarize_text

    if not section_text or not section_text.strip():
        return ""

    settings = get_settings()
    config = custom_config or settings.section_summary_config.get(
        section_name.lower().strip(),
        {"max": settings.default_summary_length, "min": settings.min_summary_length},
    )

    max_len = config.get("max", settings.default_summary_length)
    min_len = config.get("min", settings.min_summary_length)

    # 1. Clean academic PDF noise
    cleaned_text = clean_section_text(section_text)

    # 2. Extract salient sentences based on section keywords
    filtered_text = filter_sentences_by_section(cleaned_text, section_name)
    if not filtered_text.strip():
        filtered_text = cleaned_text

    # 3. Chunk text into token bounds
    chunks = chunk_text(filtered_text, max_tokens=settings.chunk_size, overlap=settings.chunk_overlap)

    if not chunks:
        return ""

    if len(chunks) == 1:
        # If single chunk fits directly, generate one focused summary
        return summarize_text(chunks[0], max_length=max_len, min_length=min_len)

    logger.info(f"Summarizing section '{section_name}' across {len(chunks)} chunks")

    # 4. First-level summaries
    chunk_summaries = [
        summarize_text(chunk, max_length=max_len, min_length=min_len)
        for chunk in chunks
    ]

    # 5. Second-level summary
    combined = " ".join([s for s in chunk_summaries if s.strip()])
    if not combined.strip():
        return ""

    final_summary = summarize_text(
        combined,
        max_length=max_len,
        min_length=min_len,
    )

    return final_summary


# Backward compatibility alias
summarize_section = summarize_section_hierarchical
