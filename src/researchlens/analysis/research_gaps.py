"""Research gap and limitation identification module."""

import re
from typing import Dict, List

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


def extract_research_gaps(text: str) -> Dict[str, List[str]]:
    """Identify explicit limitations and future research directions from text.

    All identified items are designated as potential suggestions rather than
    definitively validated scientific claims.
    """
    if not text or not text.strip():
        return {
            "potential_limitations": ["Not detected"],
            "potential_future_directions": ["Not detected"],
        }

    sentences = re.split(r"(?<=[.!?])\s+", text)

    limitations = []
    future_work = []

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 30 or len(s_clean) > 400:
            continue

        s_lower = s_clean.lower()

        # Check limitations
        for pat in LIMITATION_PATTERNS:
            if re.search(pat, s_lower):
                if s_clean not in limitations:
                    limitations.append(s_clean)
                break

        # Check future work
        for pat in FUTURE_WORK_PATTERNS:
            if re.search(pat, s_lower):
                if s_clean not in future_work:
                    future_work.append(s_clean)
                break

    return {
        "potential_limitations": (
            [f"[Potential Research Gap] {s}" for s in limitations[:5]]
            if limitations
            else ["Not detected"]
        ),
        "potential_future_directions": (
            [f"[Potential Research Direction] {s}" for s in future_work[:5]]
            if future_work
            else ["Not detected"]
        ),
    }
