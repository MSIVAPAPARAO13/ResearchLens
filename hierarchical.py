from nltk.tokenize import sent_tokenize
import re
from summarizer import summarize_text
from chunking import chunk_text

import spacy
nlp = spacy.load("en_core_web_sm")

SECTION_KEYWORDS = {
    "methodology": [
        "method", "sample", "questionnaire", "data", "analysis",
        "approach", "survey", "participants", "descriptive"
    ],
    "results": [
        "result", "showed", "found", "significant", "correlation",
        "impact", "effect", "coefficient"
    ],
    "literature review": [
        "study", "research", "previous", "literature",
        "theory", "authors", "findings"
    ],
}

SECTION_EXCLUDE = {
    "methodology": ["result", "showed", "significant", "correlation"],
    "results": ["method", "sample", "questionnaire", "approach"],
}


def filter_sentences_by_section(text, section_name, max_sentences=20):
    doc = nlp(text)
    sentences = [s.text for s in doc.sents]

    include = SECTION_KEYWORDS.get(section_name, [])
    exclude = SECTION_EXCLUDE.get(section_name, [])

    scored = []
    for s in sentences:
        s_lower = s.lower()
        if any(e in s_lower for e in exclude):
            continue
        score = sum(k in s_lower for k in include)
        if score > 0:
            scored.append((score, s))

    scored.sort(reverse=True)
    selected = [s for _, s in scored[:max_sentences]]

    return " ".join(selected) if selected else text


SECTION_SUMMARY_CONFIG = {
    "abstract": {"max": 120, "min": 60},
    "introduction": {"max": 160, "min": 80},
    "literature review": {"max": 200, "min": 120},
    "methodology": {"max": 240, "min": 140},
    "results": {"max": 240, "min": 140},
    "summary": {"max": 160, "min": 80},
    "ethics": {"max": 120, "min": 60},
    "conclusion": {"max": 160, "min": 80},
    "unknown": {"max": 120, "min": 60},
}


def clean_section_text(text):
    """
    Light cleaning to remove academic noise before summarization
    """
    text = re.sub(r"\(table\s+\d+.*?\)", "", text, flags=re.I)
    text = re.sub(r"\(figure\s+\d+.*?\)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b[A-Z]{3,}\b", "", text)   # AA, BBXX, DQXX
    text = re.sub(r"\d+\.\d+\*{1,2}", "", text)    # 0.134**, 0.000
    text = re.sub(r"\(2-tailed.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"back to.*", "", text, flags=re.I)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = text.replace("“", "").replace("”", "")
    return text.strip()


def summarize_section(section_name, section_text):
    config = SECTION_SUMMARY_CONFIG.get(
        section_name,
        {"max": 180, "min": 80}
    )

    #  CLEAN FIRST
    section_text = clean_section_text(section_text)

    # FILTER IMPORTANT KEYWORDS
    section_text = filter_sentences_by_section(section_text, section_name)

    #  THEN CHUNK
    chunks = chunk_text(section_text)

    # First-level summaries
    chunk_summaries = [
        summarize_text(
            chunk,
            max_length=config["max"],
            min_length=config["min"],
        )
        for chunk in chunks
    ]

    # Second-level summary
    combined = " ".join(chunk_summaries)

    final_summary = summarize_text(
        combined,
        max_length=config["max"],
        min_length=config["min"],
    )

    return final_summary
