"""Research Problem, Objective, Research Question, and Hypothesis extraction."""

import re
from typing import Any, List, Tuple

from researchlens.extraction.schema import ExtractedItem

PROBLEM_PATTERNS = [
    r"\b(?:address(?:es)? the problem of|focuses on the problem of|tackles the challenge of)\b",
    r"\b(?:significant challenge in|major bottleneck in|primary limitation of existing)\b",
    r"\b(?:motivated by the need to|problem of automated|difficulty of extracting)\b",
]

OBJECTIVE_PATTERNS = [
    r"\b(?:this paper proposes|we propose|this study presents|we present|we introduce)\b",
    r"\b(?:our objective is|the main objective of this study|we aim to|this paper investigates)\b",
    r"\b(?:the goal of this work|in this work, we|to address this challenge, we)\b",
]

RESEARCH_QUESTION_PATTERNS = [
    r"\b(?:research question|rq\s*[0-9]+)\b\s*[:\?\-]?\s*([^.!?\n]+[.!?])",
    r"\b(?:we investigate whether|we ask whether|how can we|what is the impact of)\b\s*([^.!?\n]+[.!?])",
]

HYPOTHESIS_PATTERNS = [
    r"\b(?:hypothesis|h\s*[0-9]+)\b\s*[:\?\-]?\s*([^.!?\n]+[.!?])",
    r"\b(?:we hypothesize that|our hypothesis is that)\b\s*([^.!?\n]+[.!?])",
]


def extract_problem_and_objective(doc: Any) -> Tuple[List[ExtractedItem], List[ExtractedItem]]:
    """Extract evidence-grounded research problem and research objective statements."""
    problems: List[ExtractedItem] = []
    objectives: List[ExtractedItem] = []

    chunks = getattr(doc, "chunks", [])
    if not chunks:
        return [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ], [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ]

    for chunk in chunks:
        sec = getattr(chunk, "section_name", "").lower()
        # Prefer Abstract, Introduction, Problem Statement, Motivation
        if sec and sec not in ("abstract", "introduction", "background", "motivation", "problem statement", "unknown", "document"):
            continue

        text = getattr(chunk, "text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 25 or len(s_clean) > 350:
                continue

            s_lower = s_clean.lower()

            # Problem extraction
            for pat in PROBLEM_PATTERNS:
                if re.search(pat, s_lower):
                    problems.append(
                        ExtractedItem(
                            value=s_clean,
                            source_text=s_clean,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "introduction"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.88,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )
                    break

            # Objective extraction
            for pat in OBJECTIVE_PATTERNS:
                if re.search(pat, s_lower):
                    objectives.append(
                        ExtractedItem(
                            value=s_clean,
                            source_text=s_clean,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "introduction"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.90,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )
                    break

    # Fallback to first paragraph of Abstract or Introduction if not detected
    if not problems and chunks:
        first_chunk = chunks[0]
        first_sentence = re.split(r"(?<=[.!?])\s+", first_chunk.text)[0].strip()
        if len(first_sentence) > 30:
            problems.append(
                ExtractedItem(
                    value=first_sentence,
                    source_text=first_sentence,
                    page_number=first_chunk.page_number,
                    section_name=first_chunk.section_name,
                    chunk_id=first_chunk.chunk_id,
                    confidence=0.60,
                    confidence_level="Medium",
                    extraction_method="heuristic_abstract",
                )
            )

    if not objectives and chunks:
        for chunk in chunks[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for s in sentences:
                if any(kw in s.lower() for kw in ["propose", "present", "introduce", "develop"]):
                    objectives.append(
                        ExtractedItem(
                            value=s.strip(),
                            source_text=s.strip(),
                            page_number=chunk.page_number,
                            section_name=chunk.section_name,
                            chunk_id=chunk.chunk_id,
                            confidence=0.70,
                            confidence_level="Medium",
                            extraction_method="heuristic_verb",
                        )
                    )
                    break
            if objectives:
                break

    if not problems:
        problems = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]
    if not objectives:
        objectives = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    return problems[:3], objectives[:3]


def extract_questions_and_hypotheses(doc: Any) -> Tuple[List[ExtractedItem], List[ExtractedItem]]:
    """Extract explicit research questions (RQs) and hypotheses (H1, H2). Return 'Not explicitly stated' if missing."""
    rqs: List[ExtractedItem] = []
    hypotheses: List[ExtractedItem] = []

    chunks = getattr(doc, "chunks", [])
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        # RQ Search
        for pat in RESEARCH_QUESTION_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                matched_str = m.group(0).strip()
                if len(matched_str) > 15:
                    rqs.append(
                        ExtractedItem(
                            value=matched_str,
                            source_text=matched_str,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "introduction"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.92,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )

        # Hypothesis Search
        for pat in HYPOTHESIS_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                matched_str = m.group(0).strip()
                if len(matched_str) > 15:
                    hypotheses.append(
                        ExtractedItem(
                            value=matched_str,
                            source_text=matched_str,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "introduction"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.92,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )

    if not rqs:
        rqs = [ExtractedItem("Not explicitly stated", confidence=0.0, confidence_level="Low", extraction_method="fallback")]
    if not hypotheses:
        hypotheses = [ExtractedItem("Not explicitly stated", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    return rqs[:5], hypotheses[:5]
