"""Experimental setup, training hyperparameter, and hardware environment extraction."""

import re
from typing import Any, List

from researchlens.extraction.schema import ExtractedItem

SETUP_PATTERNS = [
    r"\b(?:batch size(?: of)?\s*[0-9]+)\b",
    r"\b(?:learning rate(?: of)?\s*[0-9e\.-]+)\b",
    r"\b(?:trained for\s*[0-9]+\s*epochs?)\b",
    r"\b(?:optimizer(?: of| used| was)?\s*(?:adamw|adam|sgd|rmsprop))\b",
    r"\b(?:weight decay(?: of)?\s*[0-9\.-]+)\b",
    r"\b(?:trained on\s*[0-9]+\s*(?:nvidia\s*)?(?:a100|v100|rtx|gpu|tpu)s?)\b",
    r"\b(?:split into\s*[0-9]+(?::[0-9]+)+\s*(?:train/test|train/val|ratio)?)\b",
    r"\b(?:[0-9]+%?\s*train(?:ing)?,\s*[0-9]+%?\s*test(?:ing)?)\b",
    r"\b(?:5-fold|10-fold)\s*cross-validation\b",
]


def extract_experimental_setup(doc: Any) -> List[ExtractedItem]:
    """Extract evidence-grounded hyperparameter, data split, and hardware setup details."""
    setup_items: List[ExtractedItem] = []
    seen = set()

    chunks = getattr(doc, "chunks", [])
    for chunk in chunks:
        sec = getattr(chunk, "section_name", "").lower()
        if sec and sec not in ("methodology", "methods", "experiments", "experimental setup", "implementation", "results", "unknown", "document"):
            continue

        text = getattr(chunk, "text", "")
        for pat in SETUP_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                matched_text = m.group(0).strip()
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    # Extract surrounding sentence for context
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    context_sentence = matched_text
                    for s in sentences:
                        if matched_text.lower() in s.lower():
                            context_sentence = s.strip()
                            break

                    setup_items.append(
                        ExtractedItem(
                            value=matched_text,
                            source_text=context_sentence,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "experiments"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.92,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )

    if not setup_items:
        return [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ]

    return setup_items[:10]
