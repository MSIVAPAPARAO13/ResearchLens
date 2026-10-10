"""Scientific contribution statement extraction."""

import re
from typing import Any, List

from researchlens.extraction.schema import ExtractedItem

CONTRIBUTION_PATTERNS = [
    r"\b(?:our main contributions? (?:are|include))\b",
    r"\b(?:we make the following contributions?)\b",
    r"\b(?:in summary, our contributions? (?:are|include))\b",
    r"\b(?:the primary contribution of this paper is)\b",
    r"\b(?:we contribute)\b",
]


def extract_contributions(doc: Any) -> List[ExtractedItem]:
    """Identify explicit paper contribution statements with exact evidence provenance."""
    contributions: List[ExtractedItem] = []
    seen = set()

    chunks = getattr(doc, "chunks", [])
    for chunk in chunks:
        sec = getattr(chunk, "section_name", "").lower()
        if sec and sec not in ("abstract", "introduction", "conclusion", "unknown", "document"):
            continue

        text = getattr(chunk, "text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 25 or len(s_clean) > 400:
                continue

            s_lower = s_clean.lower()
            for pat in CONTRIBUTION_PATTERNS:
                if re.search(pat, s_lower):
                    if s_clean.lower() not in seen:
                        seen.add(s_clean.lower())
                        contributions.append(
                            ExtractedItem(
                                value=s_clean,
                                source_text=s_clean,
                                page_number=getattr(chunk, "page_number", None),
                                section_name=getattr(chunk, "section_name", "introduction"),
                                chunk_id=getattr(chunk, "chunk_id", None),
                                confidence=0.94,
                                confidence_level="High",
                                extraction_method="pattern",
                            )
                        )
                    break

    if not contributions:
        return [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ]

    return contributions[:5]
