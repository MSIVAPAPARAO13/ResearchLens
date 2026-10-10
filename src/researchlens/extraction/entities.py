"""Named entity recognition for academic papers."""

from typing import Dict, List

from researchlens.preprocessing.cleaner import get_spacy_model


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract recognized academic entities (Authors, Organizations, Products)."""
    if not text or not text.strip():
        return {
            "authors": ["Not detected"],
            "organizations": ["Not detected"],
            "models_or_tools": ["Not detected"],
        }

    nlp = get_spacy_model()
    if nlp is None:
        return {
            "authors": ["Not detected"],
            "organizations": ["Not detected"],
            "models_or_tools": ["Not detected"],
        }

    # Inspect first 15,000 characters (typically header, author affiliation, abstract)
    doc = nlp(text[:15000])

    authors = set()
    orgs = set()
    products = set()

    for ent in doc.ents:
        val = ent.text.strip().replace("\n", " ")
        if len(val) < 3 or val.isnumeric():
            continue
        if ent.label_ == "PERSON":
            authors.add(val)
        elif ent.label_ == "ORG":
            orgs.add(val)
        elif ent.label_ in ("PRODUCT", "WORK_OF_ART"):
            products.add(val)

    return {
        "authors": sorted(list(authors))[:8] if authors else ["Not detected"],
        "organizations": sorted(list(orgs))[:8] if orgs else ["Not detected"],
        "models_or_tools": sorted(list(products))[:8] if products else ["Not detected"],
    }
