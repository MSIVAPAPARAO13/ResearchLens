"""Embedding model abstraction layer and SentenceTransformer implementation."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch

from researchlens.config.settings import get_settings
from researchlens.utils.logging import get_logger

logger = get_logger("embeddings")

_EMBEDDING_MODEL_CACHE: Optional["SentenceTransformerEmbeddings"] = None


class BaseEmbeddingModel(ABC):
    """Abstract interface for document and query text embeddings."""

    @abstractmethod
    def embed_documents(self, texts: List[Union[str, Any]]) -> List[List[float]]:
        """Compute vector embeddings for a list of document texts or chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute vector embedding for a single user query."""
        pass

    def embed_text(self, text: str) -> List[float]:
        """Convenience alias for embed_query."""
        return self.embed_query(text)


class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    """SentenceTransformer wrapper supporting local caching, lazy load, and normalized vectors."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() and settings.device == "cuda" else "cpu")
        self._model = None

    @property
    def model(self):
        """Lazily initialize the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: List[Union[str, Any]]) -> List[List[float]]:
        """Compute normalized dense vector representations for documents/chunks."""
        if not texts:
            return []

        # Support list of string texts or objects with a .text attribute (e.g. ChunkData)
        extracted_texts: List[str] = [
            t.text if hasattr(t, "text") else (t["text"] if isinstance(t, dict) and "text" in t else str(t))
            for t in texts
        ]

        embeddings = self.model.encode(
            extracted_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Compute normalized vector representation for a query."""
        if not query or not query.strip():
            return [0.0] * self.dimension

        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic lightweight embedding model for testing and fast scaffolding."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_documents(self, texts: List[Union[str, Any]]) -> List[List[float]]:
        extracted = [
            t.text if hasattr(t, "text") else (t["text"] if isinstance(t, dict) and "text" in t else str(t))
            for t in texts
        ]
        return [
            [float((hash(t) + i) % 100) / 100.0 for i in range(self.dimension)]
            for t in extracted
        ]

    def embed_query(self, query: str) -> List[float]:
        return [float((hash(query) + i) % 100) / 100.0 for i in range(self.dimension)]


def get_embedding_model(
    model_name: Optional[str] = None,
    use_mock: bool = False,
) -> BaseEmbeddingModel:
    """Retrieve cached singleton embedding model instance."""
    global _EMBEDDING_MODEL_CACHE
    if use_mock:
        return MockEmbeddingModel()

    target_name = model_name or get_settings().embedding_model_name
    if (
        _EMBEDDING_MODEL_CACHE is None
        or _EMBEDDING_MODEL_CACHE.model_name != target_name
    ):
        _EMBEDDING_MODEL_CACHE = SentenceTransformerEmbeddings(model_name=target_name)
    return _EMBEDDING_MODEL_CACHE
