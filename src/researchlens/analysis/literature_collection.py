"""Literature collection and multi-paper document representations."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from researchlens.extraction.schema import ResearchProfile
from researchlens.ingestion.document import Document


def generate_deterministic_document_id(filename: str, content: str = "") -> str:
    """Generate a deterministic document ID using SHA-256 hash of filename and content."""
    raw = f"{filename}::{content[:2000]}"
    return f"doc_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class LiteratureDocument:
    """Data representation of a single research paper within a collection."""

    document_id: str
    filename: str
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    profile: Optional[ResearchProfile] = None
    document_obj: Optional[Document] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LiteratureDocument to JSON-compatible dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "profile": self.profile.to_dict() if self.profile and hasattr(self.profile, "to_dict") else None,
        }


@dataclass
class LiteratureCollection:
    """Collection-level container aggregating multiple LiteratureDocument items."""

    collection_id: str
    collection_name: str
    documents: List[LiteratureDocument] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def paper_count(self) -> int:
        """Return total number of papers in the collection."""
        return len(self.documents)

    @property
    def profiles(self) -> List[ResearchProfile]:
        """Return list of non-null ResearchProfiles in the collection."""
        return [doc.profile for doc in self.documents if doc.profile is not None]

    def add_document(self, doc: LiteratureDocument) -> None:
        """Add a LiteratureDocument to the collection, avoiding duplicate document IDs."""
        if not any(d.document_id == doc.document_id for d in self.documents):
            self.documents.append(doc)

    def get_document(self, document_id: str) -> Optional[LiteratureDocument]:
        """Find and return a document by its document_id."""
        for d in self.documents:
            if d.document_id == document_id:
                return d
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LiteratureCollection to JSON-compatible dictionary."""
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "paper_count": self.paper_count,
            "created_at": self.created_at,
            "documents": [d.to_dict() for d in self.documents],
        }
