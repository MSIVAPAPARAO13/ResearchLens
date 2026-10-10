"""Evidence provenance and source tracing models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvidenceSnippet:
    """Document evidence provenance pointing to raw text source."""
    claim_or_summary: str
    source_section: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    context_sentence: Optional[str] = None
    confidence_score: float = 1.0
