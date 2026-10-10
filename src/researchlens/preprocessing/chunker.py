"""Token-aware and sentence-boundary preserving text chunker."""

import re
from typing import Any, List, Optional

from researchlens.config.settings import get_settings
from researchlens.ingestion.document import ChunkData
from researchlens.utils.logging import get_logger

logger = get_logger("chunker")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _estimate_or_count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Return exact token count if tokenizer is available, or estimated count via word ratio."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    # Standard rule of thumb: 1 word ~ 1.33 tokens
    return max(1, int(len(text.split()) * 1.33))


def chunk_text(
    text: str,
    max_tokens: Optional[int] = None,
    overlap: Optional[int] = None,
    tokenizer: Optional[Any] = None,
) -> List[str]:
    """Split text into token-bounded chunks while respecting sentence boundaries.

    Preserves full backward compatibility with legacy chunking.chunk_text(text, max_tokens=900).
    Eliminates quadratic re-tokenization.
    """
    settings = get_settings()
    target_max = max_tokens or settings.chunk_size
    target_overlap = overlap if overlap is not None else settings.chunk_overlap

    if not text or not text.strip():
        return []

    if tokenizer is None:
        try:
            from researchlens.summarization.model import get_tokenizer
            tokenizer = get_tokenizer()
        except Exception:
            tokenizer = None

    sentences = _split_into_sentences(text)
    if not sentences:
        words = text.split()
        if not words:
            return []
        sentences = [" ".join(words)]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_or_count_tokens(sentence, tokenizer)

        # Handle edge case where a single sentence exceeds target_max
        if sent_tokens > target_max:
            # Flush existing chunk if present
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            # Split large sentence by words
            words = sentence.split()
            word_chunk: List[str] = []
            for w in words:
                word_chunk.append(w)
                if _estimate_or_count_tokens(" ".join(word_chunk), tokenizer) >= target_max:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = []
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        if current_tokens + sent_tokens > target_max and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Apply overlap if configured
            if target_overlap > 0:
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    cnt = _estimate_or_count_tokens(s, tokenizer)
                    if overlap_tokens + cnt <= target_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += cnt
                    else:
                        break
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


def create_chunks(
    text: str,
    section_name: str = "unknown",
    max_tokens: Optional[int] = None,
    overlap: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    page_number: Optional[int] = None,
    document_id: str = "doc",
    filename: str = "document.pdf",
    start_index: int = 0,
) -> List[ChunkData]:
    """Generate structured ChunkData records with IDs, token counts, and section metadata."""
    raw_chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap, tokenizer=tokenizer)
    result: List[ChunkData] = []
    p_num = page_number or 1
    for idx, c in enumerate(raw_chunks):
        cnt = _estimate_or_count_tokens(c, tokenizer)
        c_idx = start_index + idx
        chunk_id = f"{document_id}_p{p_num:02d}_c{c_idx:02d}"
        result.append(
            ChunkData(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page_number=page_number,
                section_name=section_name,
                chunk_index=c_idx,
                text=c,
                token_count=cnt,
            )
        )
    return result


def _find_page_number(chunk_text_sample: str, pages: list) -> int:
    """Identify which document page contains the starting content of a chunk."""
    if not pages:
        return 1
    # Match the first 6-8 words of the chunk
    sample_words = " ".join(chunk_text_sample.split()[:7]).strip()
    if sample_words:
        for p in pages:
            if sample_words.lower() in p.text.lower():
                return p.page_number
    # Fallback to single words matching
    words = [w for w in chunk_text_sample.split()[:4] if len(w) > 3]
    for p in pages:
        if any(w.lower() in p.text.lower() for w in words):
            return p.page_number
    return 1


def create_document_chunks(
    document: Any,
    max_tokens: Optional[int] = None,
    overlap: Optional[int] = None,
    tokenizer: Optional[Any] = None,
) -> List[ChunkData]:
    """Create token-bounded chunks for a complete Document preserving page numbers and sections."""
    all_chunks: List[ChunkData] = []
    doc_id = getattr(document, "document_id", "doc")
    filename = getattr(document, "filename", "document.pdf")
    pages = getattr(document, "pages", [])
    sections = getattr(document, "sections", {})

    chunk_counter = 0

    if sections:
        for sec_name, sec_text in sections.items():
            if not sec_text.strip():
                continue
            raw_chunks = chunk_text(sec_text, max_tokens=max_tokens, overlap=overlap, tokenizer=tokenizer)
            for raw_chunk in raw_chunks:
                p_num = _find_page_number(raw_chunk, pages)
                cnt = _estimate_or_count_tokens(raw_chunk, tokenizer)
                chunk_id = f"{doc_id}_p{p_num:02d}_c{chunk_counter:02d}"
                all_chunks.append(
                    ChunkData(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        filename=filename,
                        page_number=p_num,
                        section_name=sec_name,
                        chunk_index=chunk_counter,
                        text=raw_chunk,
                        token_count=cnt,
                    )
                )
                chunk_counter += 1
    else:
        # Fall back to page-by-page chunking if no explicit sections detected
        for page in pages:
            if not page.text.strip():
                continue
            page_chunks = create_chunks(
                text=page.text,
                section_name="general",
                max_tokens=max_tokens,
                overlap=overlap,
                tokenizer=tokenizer,
                page_number=page.page_number,
                document_id=doc_id,
                filename=filename,
                start_index=chunk_counter,
            )
            all_chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

    document.chunks = all_chunks
    return all_chunks
