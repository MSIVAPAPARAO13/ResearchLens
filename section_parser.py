import re

SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "literature review": r"\bliterature review\b",
    "methodology": r"\b(methodology|methods|field research procedures)\b",
    "results": r"\b(results|presentation of research results)\b",
    "summary": r"\bsummary of results\b",
    "ethics": r"\bethical considerations\b",
    "conclusion": r"\bconclusion\b",
}


def split_into_sections(text):
    sections = {}
    current_section = "unknown"
    sections[current_section] = []

    for line in text.split("\n"):
        clean = line.strip().lower()

        # Detect section headers ANYWHERE near the start
        for section, pattern in SECTION_PATTERNS.items():
            # only look at first part of line
            if re.search(pattern, clean[:120]):
                current_section = section
                sections.setdefault(current_section, [])
                break

        sections[current_section].append(line)

    # Remove very small sections
    return {
        k: " ".join(v).strip()
        for k, v in sections.items()
        if len(" ".join(v)) > 400
    }
