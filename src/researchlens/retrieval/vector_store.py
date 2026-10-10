"""Vector storage abstraction layer and FAISS-CPU local implementation."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from researchlens.utils.logging import get_logger

logger = get_logger("vector_store")


@dataclass
class VectorDocument:
    """Indexed document payload stored inside the vector index."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0


class BaseVectorStore(ABC):
    """Abstract vector database interface."""

    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add and index documents in the store."""
        pass

    @abstractmethod
    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[VectorDocument]:
        """Perform nearest-neighbor retrieval given a query embedding."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed vectors and metadata."""
        pass

    def add_chunks(self, chunks: List[Any], embeddings: List[List[float]]) -> None:
        """Convenience helper to add ChunkData objects alongside their computed embeddings."""
        docs = []
        for chunk, emb in zip(chunks, embeddings):
            meta = (
                chunk.to_dict()
                if hasattr(chunk, "to_dict")
                else (chunk if isinstance(chunk, dict) else {})
            )
            c_id = getattr(chunk, "chunk_id", meta.get("chunk_id", "chunk"))
            text = getattr(chunk, "text", meta.get("text", ""))
            docs.append(
                VectorDocument(
                    doc_id=c_id,
                    text=text,
                    metadata=meta,
                    embedding=emb,
                )
            )
        self.add_documents(docs)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-CPU local vector index using normalized inner product (cosine similarity)."""

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self.index = None
        self._documents: List[VectorDocument] = []

    def _init_index_if_needed(self, dim: int) -> None:
        if self.index is None:
            import faiss
            self.dimension = dim
            logger.info(f"Initializing FAISS IndexFlatIP (cosine similarity) with dim={dim}")
            self.index = faiss.IndexFlatIP(dim)

    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Index vector documents using FAISS."""
        if not documents:
            return

        valid_docs = [d for d in documents if d.embedding is not None and len(d.embedding) > 0]
        if not valid_docs:
            return

        dim = len(valid_docs[0].embedding)
        self._init_index_if_needed(dim)

        import faiss
        vectors = np.array([d.embedding for d in valid_docs], dtype=np.float32)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self._documents.extend(valid_docs)
        logger.info(f"Added {len(valid_docs)} vectors to FAISS index. Total count: {self.index.ntotal}")

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[VectorDocument]:
        """Perform cosine similarity search over indexed chunks."""
        if self.index is None or self.index.ntotal == 0 or not query_embedding:
            return []

        import faiss
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results: List[VectorDocument] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            orig_doc = self._documents[idx]
            # Create search result copy with attached cosine similarity score
            doc_copy = VectorDocument(
                doc_id=orig_doc.doc_id,
                text=orig_doc.text,
                metadata=dict(orig_doc.metadata),
                embedding=orig_doc.embedding,
                score=float(score),
            )
            # Store score in metadata as well for direct inspection
            doc_copy.metadata["relevance_score"] = float(score)
            results.append(doc_copy)

        return results

    def clear(self) -> None:
        """Reset the FAISS index and stored document payloads."""
        self.index = None
        self._documents.clear()


class InMemoryVectorStore(BaseVectorStore):
    """Fallback in-memory cosine similarity vector index."""

    def __init__(self):
        self._documents: List[VectorDocument] = []

    def add_documents(self, documents: List[VectorDocument]) -> None:
        self._documents.extend(documents)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[VectorDocument]:
        if not self._documents:
            return []

        scored = []
        for doc in self._documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                doc_copy = VectorDocument(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    metadata=dict(doc.metadata),
                    embedding=doc.embedding,
                    score=float(score),
                )
                doc_copy.metadata["relevance_score"] = float(score)
                scored.append((score, doc_copy))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def clear(self) -> None:
        self._documents.clear()


def create_vector_store(store_type: str = "faiss") -> BaseVectorStore:
    """Factory creating the appropriate vector store implementation."""
    if store_type.lower() == "faiss":
        try:
            return FAISSVectorStore()
        except ImportError:
            logger.warning("FAISS not available. Falling back to InMemoryVectorStore.")
            return InMemoryVectorStore()
    return InMemoryVectorStore()
