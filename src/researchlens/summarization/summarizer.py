"""High-level transformer summarization interface and document synthesis."""

from typing import Dict, Optional, Union

from researchlens.config.settings import get_settings
from researchlens.summarization.hierarchical import summarize_section_hierarchical
from researchlens.summarization.model import load_model_and_tokenizer
from researchlens.utils.logging import get_logger

logger = get_logger("summarizer")


def summarize_text(
    text: str,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    num_beams: int = 6,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
) -> str:
    """Summarize a single text chunk using the BART conditional generation model.

    Maintains full backward compatibility with legacy summarizer.summarize_text.
    """
    if not text or not text.strip():
        return ""

    settings = get_settings()
    max_len = max_length or settings.default_summary_length
    min_len = min_length or settings.min_summary_length

    # Ensure min_len <= max_len
    if min_len > max_len:
        min_len = max(20, max_len // 2)

    model, tokenizer, device = load_model_and_tokenizer()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_input_tokens,
    )
    input_ids = inputs["input_ids"].to(device)

    # If input is very short, adjust generation bounds to avoid errors
    input_len = input_ids.shape[1]
    if input_len < min_len:
        min_len = max(10, input_len // 2)
    if input_len < max_len:
        max_len = max(min_len + 10, input_len)

    try:
        summary_ids = model.generate(
            input_ids,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_length=max_len,
            min_length=min_len,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    except Exception as exc:
        logger.error(f"Inference error during summarize_text: {exc}")
        raise


def summarize_section(
    section_name: str,
    section_text: str,
    custom_config: Optional[Dict[str, int]] = None,
) -> str:
    """Summarize an individual paper section using the hierarchical pipeline."""
    return summarize_section_hierarchical(section_name, section_text, custom_config)


def summarize_document(
    sections: Dict[str, str],
) -> Dict[str, str]:
    """Generate section summaries for all detected sections of a document."""
    results: Dict[str, str] = {}
    for section_name, text in sections.items():
        if text.strip():
            logger.info(f"Summarizing section: {section_name}")
            results[section_name] = summarize_section(section_name, text)
    return results


def generate_tldr(text: str, max_length: int = 50) -> str:
    """Generate an ultra-concise TL;DR sentence for a document or abstract."""
    return summarize_text(text, max_length=max_length, min_length=20, num_beams=4)


def generate_executive_summary(section_summaries: Union[Dict[str, str], str]) -> str:
    """Synthesize high-priority section summaries or raw text into an overarching executive summary."""
    if isinstance(section_summaries, str):
        if not section_summaries.strip():
            return ""
        return summarize_text(section_summaries, max_length=220, min_length=90, num_beams=5)

    priority_sections = ["abstract", "introduction", "methodology", "results", "conclusion"]
    saliency_text = []

    for key in priority_sections:
        for sec, summ in section_summaries.items():
            if key in sec.lower() and summ.strip():
                saliency_text.append(f"{sec.upper()}: {summ.strip()}")
                break

    if not saliency_text:
        # Fall back to any available summaries
        saliency_text = [s for s in section_summaries.values() if s.strip()]

    combined = " ".join(saliency_text)
    if not combined.strip():
        return ""

    return summarize_text(combined, max_length=220, min_length=90, num_beams=5)
