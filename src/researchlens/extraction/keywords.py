"""Keyword and keyphrase extraction from academic text."""

from collections import Counter
from typing import List

from researchlens.preprocessing.cleaner import get_spacy_model

# Standard academic stopwords to exclude
STOPWORDS = {
    "et", "al", "paper", "study", "research", "results", "table", "figure",
    "method", "data", "using", "based", "proposed", "analysis", "system",
    "approach", "author", "authors", "model", "performance", "shown", "show"
}


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Extract top salient keyphrases using noun chunks and frequency counts."""
    if not text or not text.strip():
        return ["Not detected"]

    nlp = get_spacy_model()
    if nlp is not None:
        # Process a reasonable window to avoid OOM on huge documents
        doc = nlp(text[:25000])
        candidates = []
        for chunk in doc.noun_chunks:
            lemma = chunk.lemma_.lower().strip()
            # Filter short fragments and stopwords
            words = [w for w in lemma.split() if w not in STOPWORDS and len(w) > 2]
            if words:
                candidate = " ".join(words)
                if len(candidate) > 3 and not candidate.isnumeric():
                    candidates.append(candidate)

        if candidates:
            counts = Counter(candidates)
            return [phrase for phrase, _ in counts.most_common(top_k)]

    # Fallback to word frequency if spaCy noun chunks unavailable
    words = [w.lower().strip(".,;:()") for w in text.split() if len(w) > 4]
    filtered = [w for w in words if w not in STOPWORDS and w.isalpha()]
    counts = Counter(filtered)
    top_words = [w for w, _ in counts.most_common(top_k)]
    return top_words if top_words else ["Not detected"]
