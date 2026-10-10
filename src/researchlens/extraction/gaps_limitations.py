"""Limitation, research gap, and future work extraction with evidence provenance."""

import re
from typing import Any, List, Tuple

from researchlens.extraction.schema import ExtractedItem

LIMITATION_PATTERNS = [
    r"\b(?:limitation|limitations|drawback|drawbacks|shortcoming|shortcomings)\b",
    r"\b(?:is limited by|was limited to|remains challenging|remains an open question)\b",
    r"\b(?:constrained by|bottleneck|vulnerability|threats to validity)\b",
]

FUTURE_WORK_PATTERNS = [
    r"\b(?:future work|future research|further research|future direction|in the future)\b",
    r"\b(?:we plan to|we intend to|remains for future work|would benefit from)\b",
    r"\b(?:avenue for future|promising direction|open problem)\b",
]


def extract_limitations_and_future_work(doc: Any) -> Tuple[List[ExtractedItem], List[ExtractedItem]]:
    """Extract evidence-grounded limitations and explicit future work statements."""
    limitations: List[ExtractedItem] = []
    future_work: List[ExtractedItem] = []
    seen = set()

    chunks = getattr(doc, "chunks", []) if doc else []

    if not chunks:
        text = doc.full_text if hasattr(doc, "full_text") else str(doc or "")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_clean = s.strip()
            if 30 <= len(s_clean) <= 400:
                s_lower = s_clean.lower()
                for pat in LIMITATION_PATTERNS:
                    if re.search(pat, s_lower) and s_clean.lower() not in seen:
                        seen.add(s_clean.lower())
                        limitations.append(ExtractedItem(value=s_clean, source_text=s_clean, page_number=1, section_name="discussion"))
                        break
                for pat in FUTURE_WORK_PATTERNS:
                    if re.search(pat, s_lower) and s_clean.lower() not in seen:
                        seen.add(s_clean.lower())
                        future_work.append(ExtractedItem(value=s_clean, source_text=s_clean, page_number=1, section_name="conclusion"))
                        break
        if not limitations:
            limitations = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low")]
        if not future_work:
            future_work = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low")]
        return limitations[:5], future_work[:5]

    for chunk in chunks:
        sec = getattr(chunk, "section_name", "").lower()

        text = getattr(chunk, "text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 30 or len(s_clean) > 400:
                continue

            s_lower = s_clean.lower()

            # Limitations
            for pat in LIMITATION_PATTERNS:
                if re.search(pat, s_lower):
                    if s_clean.lower() not in seen:
                        seen.add(s_clean.lower())
                        conf = 0.92 if sec in ("discussion", "limitations", "conclusion") else 0.82
                        limitations.append(
                            ExtractedItem(
                                value=s_clean,
                                source_text=s_clean,
                                page_number=getattr(chunk, "page_number", None),
                                section_name=getattr(chunk, "section_name", "discussion"),
                                chunk_id=getattr(chunk, "chunk_id", None),
                                confidence=conf,
                                confidence_level="High" if conf > 0.85 else "Medium",
                                extraction_method="pattern",
                            )
                        )
                    break

            # Future Work
            for pat in FUTURE_WORK_PATTERNS:
                if re.search(pat, s_lower):
                    if s_clean.lower() not in seen:
                        seen.add(s_clean.lower())
                        conf = 0.92 if sec in ("conclusion", "future work", "discussion") else 0.82
                        future_work.append(
                            ExtractedItem(
                                value=s_clean,
                                source_text=s_clean,
                                page_number=getattr(chunk, "page_number", None),
                                section_name=getattr(chunk, "section_name", "conclusion"),
                                chunk_id=getattr(chunk, "chunk_id", None),
                                confidence=conf,
                                confidence_level="High" if conf > 0.85 else "Medium",
                                extraction_method="pattern",
                            )
                        )
                    break

    if not limitations:
        limitations = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]
    if not future_work:
        future_work = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    return limitations[:5], future_work[:5]
