"""Local BM25 lexical retriever for exact scientific keyword and entity matching."""

import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from researchlens.config.settings import get_settings
from researchlens.retrieval.retriever import SearchResult
from researchlens.utils.logging import get_logger

logger = get_logger("bm25")


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize academic and scientific text preserving model names, abbreviations, and numbers."""
    if not text:
        return []
    # Match alphanumeric words and hyphenated terms (e.g., 'resnet-50', 'f1-score')
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text.lower())
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class BaseBM25Retriever(ABC):
    """Abstract interface for BM25 lexical retrieval."""

    @abstractmethod
    def index_chunks(self, chunks: List[Any]) -> None:
        """Build the lexical BM25 index over document chunks."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve top-k chunks matching the lexical query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear indexed chunks and reset the BM25 model."""
        pass


class BM25Retriever(BaseBM25Retriever):
    """Local in-memory BM25Okapi implementation preserving provenance and metadata."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self._chunks: List[Any] = []

    def index_chunks(self, chunks: List[Any]) -> None:
        """Index a list of ChunkData or dictionary objects."""
        if not chunks:
            self.clear()
            return

        self._chunks = list(chunks)
        tokenized_corpus = []

        for chunk in self._chunks:
            text = getattr(chunk, "text", "")
            if not text and isinstance(chunk, dict):
                text = chunk.get("text", "")
            tokenized_corpus.append(tokenize_for_bm25(text))

        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"Indexed {len(self._chunks)} chunks in local BM25Okapi model")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Execute lexical query scoring and return standardized SearchResults."""
        if self.bm25 is None or not self._chunks or not query or not query.strip():
            return []

        settings = get_settings()
        k = top_k or settings.bm25_candidate_k

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Get top-k indices sorted descending by score
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            # Only include non-zero scoring chunks unless all scores are 0
            if score <= 0.0 and len(results) > 0:
                continue

            chunk = self._chunks[idx]
            meta = (
                chunk.to_dict()
                if hasattr(chunk, "to_dict")
                else (dict(chunk) if isinstance(chunk, dict) else {})
            )
            c_id = getattr(chunk, "chunk_id", meta.get("chunk_id", f"chunk_{idx}"))
            text = getattr(chunk, "text", meta.get("text", ""))
            p_num = getattr(chunk, "page_number", meta.get("page_number"))
            s_name = getattr(chunk, "section_name", meta.get("section_name", "unknown"))

            meta["bm25_raw_score"] = score

            results.append(
                SearchResult(
                    doc_id=c_id,
                    text=text,
                    score=score,
                    page_number=p_num,
                    section_name=s_name,
                    metadata=meta,
                )
            )

        logger.info(f"BM25 retrieved {len(results)} chunks for query: '{query[:40]}...'")
        return results

    def clear(self) -> None:
        self.bm25 = None
        self._chunks.clear()
