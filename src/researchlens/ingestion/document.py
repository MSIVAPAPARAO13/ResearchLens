"""Data structures and representations for academic documents."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Page:
    """Individual page representation within a document."""
    page_number: int
    text: str
    char_count: int = 0

    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.text)


@dataclass
class ChunkData:
    """Token-bounded text chunk with complete provenance metadata."""
    chunk_id: str
    text: str
    token_count: int
    section_name: str = "unknown"
    document_id: str = "doc"
    filename: str = "document.pdf"
    page_number: Optional[int] = None
    chunk_index: int = 0
    section: Optional[str] = None

    def __post_init__(self):
        if self.section is not None and self.section_name == "unknown":
            self.section_name = self.section
        elif self.section is None:
            self.section = self.section_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass
class SectionData:
    """Academic section containing aggregated text and child chunks."""
    name: str
    text: str
    summary: Optional[str] = None
    chunks: List[ChunkData] = field(default_factory=list)


@dataclass
class DocumentMetadata:
    """Extracted metadata describing the source PDF."""
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0


@dataclass
class Document:
    """Complete structured document representation."""
    filename: str
    full_text: str
    pages: List[Page] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    chunks: List[ChunkData] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
