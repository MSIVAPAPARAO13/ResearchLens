"""Text manipulation and normalization helpers."""

import re


def clean_whitespace(text: str) -> str:
    """Normalize irregular whitespace characters to single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Return count of whitespace-delimited words."""
    if not text:
        return 0
    return len(text.split())


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    """Safely truncate text to a maximum character threshold."""
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix
