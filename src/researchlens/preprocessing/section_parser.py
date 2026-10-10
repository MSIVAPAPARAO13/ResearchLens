"""Deterministic regex-based academic section boundary detection."""

import re
from typing import Dict, Optional

from researchlens.config.settings import get_settings

# Preserved for backward compatibility with legacy section_parser.py
SECTION_PATTERNS: Dict[str, str] = get_settings().section_patterns


def split_into_sections(
    text: str,
    patterns: Optional[Dict[str, str]] = None,
    min_length: Optional[int] = None,
) -> Dict[str, str]:
    """Parse raw academic text into canonical sections based on heading patterns.

    Maintains full backward compatibility with legacy section_parser.split_into_sections.
    """
    settings = get_settings()
    active_patterns = patterns or settings.section_patterns
    threshold = min_length if min_length is not None else settings.min_section_length

    sections: Dict[str, list[str]] = {}
    current_section = "unknown"
    sections[current_section] = []

    compiled_patterns = {
        sec: re.compile(pat, re.IGNORECASE) for sec, pat in active_patterns.items()
    }

    for line in text.split("\n"):
        clean = line.strip().lower()

        # Check the first 120 characters of the line for a section header
        detected_section = None
        for sec, regex in compiled_patterns.items():
            if regex.search(clean[:120]):
                detected_section = sec
                break

        if detected_section:
            current_section = detected_section
            sections.setdefault(current_section, [])

        sections[current_section].append(line)

    # Filter out empty or negligible sections according to configured threshold
    result: Dict[str, str] = {}
    for sec, lines in sections.items():
        joined_text = " ".join(lines).strip()
        if len(joined_text) >= threshold:
            result[sec] = joined_text

    return result
