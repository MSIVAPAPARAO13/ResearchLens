"""Scientific paper metadata extraction (Title, Authors, Organizations, Year, Keywords)."""

import re
from typing import Any, List, Optional, Tuple

from researchlens.extraction.schema import ExtractedItem
from researchlens.preprocessing.cleaner import get_spacy_model


def extract_title(doc: Any) -> ExtractedItem:
    """Extract paper title from PDF metadata or initial document text with provenance."""
    # 1. Check PDF metadata title
    if hasattr(doc, "metadata") and getattr(doc.metadata, "title", None):
        meta_title = doc.metadata.title.strip()
        if len(meta_title) > 8 and not meta_title.lower().startswith("microsoft word"):
            return ExtractedItem(
                value=meta_title,
                source_text=meta_title,
                page_number=1,
                section_name="header",
                confidence=0.95,
                confidence_level="High",
                extraction_method="pdf_metadata",
            )

    # 2. Extract from first page top text
    if hasattr(doc, "pages") and doc.pages:
        first_page_text = doc.pages[0].text.strip()
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        for line in lines[:5]:
            # Skip common header noise
            if re.search(r"\b(?:arxiv|journal|proceedings|vol|volume|issn|isbn|doi)\b", line, flags=re.IGNORECASE):
                continue
            if len(line) > 12 and not line.endswith(".pdf"):
                return ExtractedItem(
                    value=line,
                    source_text=line,
                    page_number=1,
                    section_name="header",
                    confidence=0.85,
                    confidence_level="High",
                    extraction_method="heading_layout",
                )

    return ExtractedItem(
        value="Not detected",
        source_text="No explicit title found in metadata or header",
        confidence=0.0,
        confidence_level="Low",
        extraction_method="fallback",
    )


def extract_authors_and_orgs(doc: Any) -> Tuple[List[ExtractedItem], List[ExtractedItem]]:
    """Extract author names and institutional affiliations using spaCy NER and header patterns."""
    if not hasattr(doc, "full_text") or not doc.full_text.strip():
        return [ExtractedItem("Not detected")], [ExtractedItem("Not detected")]

    header_text = doc.full_text[:5000]
    nlp = get_spacy_model()

    authors: List[ExtractedItem] = []
    orgs: List[ExtractedItem] = []

    if nlp is not None:
        spacy_doc = nlp(header_text)
        seen_authors = set()
        seen_orgs = set()

        for ent in spacy_doc.ents:
            val = ent.text.strip().replace("\n", " ")
            if len(val) < 3 or val.isnumeric() or "@" in val:
                continue

            if ent.label_ == "PERSON" and val.lower() not in seen_authors:
                # Basic sanity filter for author names (e.g. 2-4 words, capitalized)
                words = val.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                    seen_authors.add(val.lower())
                    authors.append(
                        ExtractedItem(
                            value=val,
                            source_text=ent.sent.text.strip() if ent.sent else val,
                            page_number=1,
                            section_name="header",
                            confidence=0.80,
                            confidence_level="Medium",
                            extraction_method="spacy_ner",
                        )
                    )
            elif ent.label_ == "ORG" and val.lower() not in seen_orgs:
                if any(kw in val.lower() for kw in ["univ", "inst", "lab", "dept", "department", "center", "centre", "corp", "inc"]):
                    seen_orgs.add(val.lower())
                    orgs.append(
                        ExtractedItem(
                            value=val,
                            source_text=ent.sent.text.strip() if ent.sent else val,
                            page_number=1,
                            section_name="header",
                            confidence=0.85,
                            confidence_level="High",
                            extraction_method="spacy_ner",
                        )
                    )

    if not authors:
        authors = [ExtractedItem(value="Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]
    if not orgs:
        orgs = [ExtractedItem(value="Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    return authors[:6], orgs[:6]


def extract_publication_year(doc: Any) -> Optional[str]:
    """Extract publication year from header/footer text or metadata."""
    header_text = doc.full_text[:4000] if hasattr(doc, "full_text") else ""
    matches = re.findall(r"\b(19[8-9][0-9]|20[0-2][0-9])\b", header_text)
    if matches:
        return matches[0]
    return None
