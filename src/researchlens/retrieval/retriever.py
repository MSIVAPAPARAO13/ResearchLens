"""Retriever implementation orchestrating question embedding, vector search, and filtering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from researchlens.config.settings import get_settings
from researchlens.retrieval.embeddings import BaseEmbeddingModel
from researchlens.retrieval.vector_store import BaseVectorStore
from researchlens.utils.logging import get_logger

logger = get_logger("retriever")


@dataclass
class SearchResult:
    """Retrieved document fragment with relevance score and provenance metadata."""
    doc_id: str
    text: str
    score: float
    page_number: Optional[int] = None
    section_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    reranker_score: Optional[float] = None

    def __post_init__(self):
        if self.page_number is None and "page_number" in self.metadata:
            self.page_number = self.metadata["page_number"]
        if self.section_name == "unknown":
            self.section_name = self.metadata.get("section_name") or self.metadata.get("section", "unknown")
        if self.reranker_score is None and "reranker_score" in self.metadata:
            self.reranker_score = self.metadata["reranker_score"]

    @property
    def chunk_id(self) -> str:
        return self.doc_id

    @property
    def document_id(self) -> str:
        return self.metadata.get("document_id", "")

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    @property
    def retrieval_score(self) -> float:
        return self.score

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "chunk_id": self.doc_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": round(self.score, 4),
            "retrieval_score": round(self.score, 4),
            "page_number": self.page_number,
            "section_name": self.section_name,
            "metadata": self.metadata,
        }
        if self.reranker_score is not None:
            res["reranker_score"] = round(self.reranker_score, 4)
        return res


class BaseRetriever(ABC):
    """Abstract retriever contract."""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve most relevant chunks for a user query."""
        pass


class DenseRetriever(BaseRetriever):
    """Dense embedding retriever with score thresholding and metadata provenance preservation."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        similarity_threshold: Optional[float] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        settings = get_settings()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        settings = get_settings()
        k = top_k or settings.top_k

        query_vec = self.embedding_model.embed_query(query)
        docs = self.vector_store.similarity_search(query_vec, top_k=k)

        results: List[SearchResult] = []
        for doc in docs:
            score = getattr(doc, "score", 0.0)
            # Filter results if similarity threshold is configured and score is below it
            if self.similarity_threshold is not None and score < self.similarity_threshold:
                continue

            meta = doc.metadata or {}
            p_num = meta.get("page_number")
            s_name = meta.get("section_name") or meta.get("section", "unknown")

            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    page_number=p_num,
                    section_name=s_name,
                    metadata=meta,
                )
            )

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:40]}...'")
        return results
