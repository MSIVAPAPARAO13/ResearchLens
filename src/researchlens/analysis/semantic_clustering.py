"""Semantic clustering of cross-paper limitations and future work statements into recurring themes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from researchlens.analysis.literature_collection import LiteratureCollection
from researchlens.extraction.schema import ExtractedItem
from researchlens.retrieval.embeddings import get_embedding_model


@dataclass
class LimitationCluster:
    """Dataclass representing a recurring limitation or future work theme across papers."""

    cluster_id: str
    theme_name: str
    category: str
    supporting_papers: List[str] = field(default_factory=list)
    supporting_evidence: List[ExtractedItem] = field(default_factory=list)
    confidence_level: str = "Medium"
    recurrence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LimitationCluster to a JSON-compatible dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "theme_name": self.theme_name,
            "category": self.category,
            "supporting_papers": self.supporting_papers,
            "supporting_evidence": [
                ev.to_dict() if hasattr(ev, "to_dict") else str(ev) for ev in self.supporting_evidence
            ],
            "confidence_level": self.confidence_level,
            "recurrence_count": self.recurrence_count,
        }


def _cosine_similarity_matrix(embeddings: List[List[float]]) -> np.ndarray:
    """Compute pair-wise cosine similarity matrix."""
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_arr = arr / norms
    return np.dot(norm_arr, norm_arr.T)


def cluster_limitations_and_future_work(
    collection: LiteratureCollection,
    threshold: float = 0.45,
    ablation_mode: str = "full_evidence",
) -> Tuple[List[LimitationCluster], List[LimitationCluster]]:
    """Cluster extracted limitations and future work across all papers into recurring themes.

    Ablation modes:
        - 'lexical_only': Exact word overlap matching.
        - 'semantic_only': Cosine similarity using SentenceTransformers embeddings.
        - 'lexical_semantic': Combined lexical overlap + semantic threshold.
        - 'full_evidence': Full semantic + section provenance-aware evidence clustering.
    """
    all_lims: List[Tuple[str, ExtractedItem]] = []
    all_fw: List[Tuple[str, ExtractedItem]] = []

    for doc in collection.documents:
        if not doc.profile:
            continue
        for lim in doc.profile.limitations:
            if lim.value not in ("Not detected", "Not explicitly stated"):
                all_lims.append((doc.title, lim))
        for fw in doc.profile.future_work:
            if fw.value not in ("Not detected", "Not explicitly stated"):
                all_fw.append((doc.title, fw))

    lim_clusters = _cluster_items(all_lims, "lim", threshold, ablation_mode)
    fw_clusters = _cluster_items(all_fw, "fw", threshold, ablation_mode)

    return lim_clusters, fw_clusters


def _cluster_items(
    items: List[Tuple[str, ExtractedItem]],
    prefix: str,
    threshold: float,
    ablation_mode: str,
) -> List[LimitationCluster]:
    if not items:
        return []

    texts = [item[1].value for item in items]

    if ablation_mode == "lexical_only":
        # Group by common key terms
        clusters_map: Dict[str, List[Tuple[str, ExtractedItem]]] = {}
        for paper_title, ev in items:
            key = ev.value.lower()[:30]
            if key not in clusters_map:
                clusters_map[key] = []
            clusters_map[key].append((paper_title, ev))

        clusters = []
        for idx, (k, group) in enumerate(clusters_map.items(), 1):
            papers = list(set(g[0] for g in group))
            evidence_items = [g[1] for g in group]
            clusters.append(
                LimitationCluster(
                    cluster_id=f"{prefix}_cluster_{idx}",
                    theme_name=evidence_items[0].value,
                    category=_categorize_text(evidence_items[0].value),
                    supporting_papers=papers,
                    supporting_evidence=evidence_items,
                    confidence_level="High" if len(papers) > 1 else "Medium",
                    recurrence_count=len(papers),
                )
            )
        return sorted(clusters, key=lambda c: c.recurrence_count, reverse=True)

    # Use SentenceTransformers embedding model
    emb_model = get_embedding_model()
    embeddings = emb_model.embed_documents(texts)
    sim_mat = _cosine_similarity_matrix(embeddings)

    visited = set()
    clusters = []
    cluster_idx = 1

    for i in range(len(items)):
        if i in visited:
            continue

        visited.add(i)
        current_group = [items[i]]

        for j in range(i + 1, len(items)):
            if j in visited:
                continue

            sim = float(sim_mat[i][j])
            if ablation_mode == "lexical_semantic":
                # Require both high similarity and string token overlap
                w_i = set(items[i][1].value.lower().split())
                w_j = set(items[j][1].value.lower().split())
                overlap = len(w_i.intersection(w_j)) / max(len(w_i), 1)
                is_match = sim >= threshold and overlap >= 0.25
            else:
                is_match = sim >= threshold

            if is_match:
                visited.add(j)
                current_group.append(items[j])

        papers = list(set(g[0] for g in current_group))
        evidence_items = [g[1] for g in current_group]

        theme_title = evidence_items[0].value
        if len(theme_title) > 80:
            theme_title = theme_title[:77] + "..."

        clusters.append(
            LimitationCluster(
                cluster_id=f"{prefix}_cluster_{cluster_idx}",
                theme_name=theme_title,
                category=_categorize_text(theme_title),
                supporting_papers=papers,
                supporting_evidence=evidence_items,
                confidence_level="High" if len(papers) > 1 else "Medium",
                recurrence_count=len(papers),
            )
        )
        cluster_idx += 1

    return sorted(clusters, key=lambda c: c.recurrence_count, reverse=True)


def _categorize_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("dataset", "data", "sample", "corpus", "benchmark")):
        return "DATASET"
    if any(k in t for k in ("generaliz", "domain", "transfer", "out-of-distribution")):
        return "GENERALIZATION"
    if any(k in t for k in ("memory", "gpu", "compute", "time", "speed", "latency", "efficiency")):
        return "EFFICIENCY"
    if any(k in t for k in ("baseline", "compar", "prior work")):
        return "BASELINE"
    if any(k in t for k in ("robust", "noise", "adversarial")):
        return "ROBUSTNESS"
    if any(k in t for k in ("ablation", "metric", "evaluat")):
        return "EVALUATION"
    return "METHODOLOGY"
