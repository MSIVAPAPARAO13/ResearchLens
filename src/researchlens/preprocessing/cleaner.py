"""Text cleaning and extractive sentence filtering for academic paper sections."""

import re
from typing import Dict, List

from researchlens.utils.logging import get_logger

logger = get_logger("cleaner")

# Section-specific relevance keywords (preserved from original base)
SECTION_KEYWORDS: Dict[str, List[str]] = {
    "methodology": [
        "method", "sample", "questionnaire", "data", "analysis",
        "approach", "survey", "participants", "descriptive", "procedure",
        "architecture", "dataset", "baseline", "hyperparameters"
    ],
    "results": [
        "result", "showed", "found", "significant", "correlation",
        "impact", "effect", "coefficient", "accuracy", "performance",
        "score", "higher", "lower", "improved", "outperformed"
    ],
    "literature review": [
        "study", "research", "previous", "literature",
        "theory", "authors", "findings", "proposed", "prior", "work"
    ],
}

# Cross-section exclusion criteria (preserved from original base)
SECTION_EXCLUDE: Dict[str, List[str]] = {
    "methodology": ["result", "showed", "significant", "correlation"],
    "results": ["sample", "questionnaire", "survey", "demographic"],
}

_SPACY_NLP = None


def get_spacy_model():
    """Lazily load and cache the spaCy pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception as exc:
            logger.warning(f"Could not load spaCy model en_core_web_sm ({exc}). Falling back to regex.")
            _SPACY_NLP = False
    return _SPACY_NLP if _SPACY_NLP is not False else None


def clean_section_text(text: str) -> str:
    """Clean common academic PDF noise while preserving scientific acronyms.

    Refactored from original hierarchical.py:
    - Retains legitimate scientific acronyms (e.g. BERT, LSTM, CNN) instead of stripping [A-Z]{3,}.
    - Removes table/figure citation artifacts, p-values, URLs, and smart quotes.
    """
    if not text:
        return ""

    # Remove (Table 1), (Figure 2a) references
    text = re.sub(r"\((?:table|figure|fig\.)\s+\d+.*?\)", "", text, flags=re.IGNORECASE)

    # Remove p-value statistical notation and significance asterisks: e.g. 0.05*, p < 0.001
    text = re.sub(r"\d+\.\d+\*{1,3}", "", text)
    text = re.sub(r"\(2-tailed.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(p\s*[<>=]\s*0?\.\d+\)", "", text, flags=re.IGNORECASE)

    # Remove repetitive page header/footer fragments like 'back to contents'
    text = re.sub(r"back to\s+[a-z\s]+", "", text, flags=re.IGNORECASE)

    # Remove raw URLs
    text = re.sub(r"https?://\S+", "", text)

    # Normalize typographic smart quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def filter_sentences_by_section(text: str, section_name: str, max_sentences: int = 20) -> str:
    """Score and filter sentences by section keywords to retain high-salience content."""
    if not text:
        return ""

    section_key = section_name.lower().strip()
    nlp = get_spacy_model()

    if nlp is not None:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        # Regex fallback for sentence splitting if spaCy model is unavailable
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    include = SECTION_KEYWORDS.get(section_key, [])
    exclude = SECTION_EXCLUDE.get(section_key, [])

    if not include:
        return text

    scored = []
    for s in sentences:
        s_lower = s.lower()
        if any(e in s_lower for e in exclude):
            continue
        score = sum(k in s_lower for k in include)
        if score > 0:
            scored.append((score, s))

    if not scored:
        return text

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [s for _, s in scored[:max_sentences]]
    return " ".join(selected)
