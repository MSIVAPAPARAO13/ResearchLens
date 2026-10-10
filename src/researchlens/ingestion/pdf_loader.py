"""PDF loading and robust text extraction layer."""

import os
from pathlib import Path
from typing import List, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from researchlens.ingestion.document import Document, DocumentMetadata, Page
from researchlens.utils.logging import get_logger

logger = get_logger("pdf_loader")


def extract_text_from_pdf(pdf_source: Union[str, Path, bytes]) -> str:
    """Extract raw text from a PDF file path or raw bytes buffer.

    Preserves backward compatibility with legacy pdf_utils.extract_text_from_pdf.
    Guarantees document handle closure.
    """
    doc = None
    try:
        if isinstance(pdf_source, (str, Path)):
            doc = fitz.open(str(pdf_source))
        elif isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

        full_text: List[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                full_text.append(page_text)

        return "\n".join(full_text)
    except Exception as exc:
        logger.error(f"Failed to extract text from PDF: {exc}")
        raise
    finally:
        if doc is not None:
            doc.close()


def load_pdf(pdf_source: Union[str, Path, bytes], filename: str = "document.pdf") -> Document:
    """Load a PDF source and return a structured Document instance with page records."""
    doc = None
    try:
        file_size = 0
        if isinstance(pdf_source, (str, Path)):
            path_obj = Path(pdf_source)
            filename = path_obj.name
            file_size = os.path.getsize(path_obj) if path_obj.exists() else 0
            doc = fitz.open(str(path_obj))
        elif isinstance(pdf_source, bytes):
            file_size = len(pdf_source)
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

        pages: List[Page] = []
        full_text_parts: List[str] = []

        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(Page(page_number=idx, text=text))
            if text:
                full_text_parts.append(text)

        meta = doc.metadata or {}
        metadata = DocumentMetadata(
            title=meta.get("title") or filename,
            author=meta.get("author") or "Unknown",
            page_count=len(pages),
            file_size_bytes=file_size,
        )

        return Document(
            filename=filename,
            full_text="\n".join(full_text_parts),
            pages=pages,
            metadata=metadata,
        )
    finally:
        if doc is not None:
            doc.close()
