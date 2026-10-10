"""Hybrid retriever combining dense vector similarity with BM25 lexical matching."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from researchlens.config.settings import get_settings
from researchlens.retrieval.bm25 import BaseBM25Retriever
from researchlens.retrieval.retriever import BaseRetriever, SearchResult
from researchlens.utils.logging import get_logger

logger = get_logger("hybrid_retriever")


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize a dictionary of scores into the [0.0, 1.0] interval."""
    if not scores:
        return {}

    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)

    if max_v == min_v:
        # All scores equal: return 1.0 for positive scores, 0.0 otherwise
        return {k: (1.0 if max_v > 0 else 0.0) for k in scores}

    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class BaseHybridRetriever(ABC):
    """Abstract interface for hybrid multi-stage retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Execute hybrid search combining dense and lexical scoring."""
        pass


class HybridRetriever(BaseHybridRetriever):
    """Hybrid search combining Dense and BM25 retrievers with normalized linear score fusion."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseBM25Retriever,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        dense_candidate_k: Optional[int] = None,
        bm25_candidate_k: Optional[int] = None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        settings = get_settings()

        d_w = dense_weight if dense_weight is not None else settings.dense_weight
        b_w = bm25_weight if bm25_weight is not None else settings.bm25_weight

        # Normalize weights so they sum to 1.0
        total_w = d_w + b_w
        if total_w > 0:
            self.dense_weight = d_w / total_w
            self.bm25_weight = b_w / total_w
        else:
            self.dense_weight = 0.6
            self.bm25_weight = 0.4

        self.dense_candidate_k = dense_candidate_k or settings.dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k or settings.bm25_candidate_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        settings = get_settings()
        final_k = top_k or settings.top_k

        # 1. Retrieve first-stage candidate pools
        dense_candidates = self.dense_retriever.retrieve(query, top_k=self.dense_candidate_k)
        bm25_candidates = self.bm25_retriever.retrieve(query, top_k=self.bm25_candidate_k)

        if not dense_candidates and not bm25_candidates:
            return []

        # 2. Extract raw score dictionaries for normalization
        dense_raw = {c.doc_id: c.score for c in dense_candidates}
        bm25_raw = {c.doc_id: c.score for c in bm25_candidates}

        dense_norm = min_max_normalize(dense_raw)
        bm25_norm = min_max_normalize(bm25_raw)

        # 3. Fuse scores and merge chunk payloads
        chunk_map: Dict[str, SearchResult] = {}
        for c in dense_candidates:
            chunk_map[c.doc_id] = c
        for c in bm25_candidates:
            if c.doc_id not in chunk_map:
                chunk_map[c.doc_id] = c

        all_doc_ids = set(dense_raw.keys()) | set(bm25_raw.keys())
        fused_candidates: List[SearchResult] = []

        for doc_id in all_doc_ids:
            chunk = chunk_map[doc_id]
            d_score = dense_norm.get(doc_id, 0.0)
            b_score = bm25_norm.get(doc_id, 0.0)

            fused_score = (self.dense_weight * d_score) + (self.bm25_weight * b_score)

            meta = dict(chunk.metadata)
            meta["dense_raw_score"] = dense_raw.get(doc_id, 0.0)
            meta["dense_norm_score"] = round(d_score, 4)
            meta["bm25_raw_score"] = bm25_raw.get(doc_id, 0.0)
            meta["bm25_norm_score"] = round(b_score, 4)
            meta["hybrid_score"] = round(fused_score, 4)

            fused_candidates.append(
                SearchResult(
                    doc_id=doc_id,
                    text=chunk.text,
                    score=float(fused_score),
                    page_number=chunk.page_number,
                    section_name=chunk.section_name,
                    metadata=meta,
                )
            )

        # 4. Sort descending by fused hybrid score
        fused_candidates.sort(key=lambda x: x.score, reverse=True)
        return fused_candidates[:final_k]


class ScientificRetrievalPipeline:
    """Unified retrieval orchestrator supporting Dense, BM25, Hybrid, and Hybrid+Reranker."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseBM25Retriever,
        reranker: Optional[Any] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid_reranked",
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Dispatch query to selected retrieval strategy."""
        strat = strategy.lower().strip()
        settings = get_settings()

        if strat == "dense":
            return self.dense_retriever.retrieve(query, top_k=top_k)
        elif strat == "bm25":
            return self.bm25_retriever.retrieve(query, top_k=top_k)
        elif strat == "hybrid":
            return self.hybrid_retriever.retrieve(query, top_k=top_k)
        elif strat in ("hybrid_reranked", "hybrid_reranker", "hybrid + reranker"):
            pool_k = settings.dense_candidate_k + settings.bm25_candidate_k
            candidates = self.hybrid_retriever.retrieve(query, top_k=pool_k)
            if self.reranker:
                final_k = top_k or settings.rerank_top_k
                return self.reranker.rerank(query, candidates, top_k=final_k)
            return candidates[:top_k or settings.rerank_top_k]
        else:
            raise ValueError(
                f"Unknown retrieval strategy: '{strategy}'. "
                f"Choose from: 'dense', 'bm25', 'hybrid', 'hybrid_reranked'."
            )

