"""Information retrieval evaluation metrics: Precision@K, Recall@K, and MRR."""

from typing import List


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Precision at rank K."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for item in top_k if item in rel_set)
    return round(hits / k, 4)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Recall at rank K."""
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for item in top_k if item in rel_set)
    return round(hits / len(relevant_ids), 4)


def mean_reciprocal_rank(retrieved_ranked_lists: List[List[str]], relevant_sets: List[List[str]]) -> float:
    """Compute Mean Reciprocal Rank across a collection of queries."""
    if not retrieved_ranked_lists or len(retrieved_ranked_lists) != len(relevant_sets):
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_ranked_lists, relevant_sets):
        rel_set = set(relevant)
        rr = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if item in rel_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Hit Rate (Success@K): 1.0 if any relevant item is in top K, else 0.0."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    return 1.0 if any(item in rel_set for item in top_k) else 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K with binary relevance."""
    import math

    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in rel_set:
            dcg += 1.0 / math.log2(i + 2)  # rank 1 gives log2(2) = 1

    idcg = 0.0
    num_relevant = min(k, len(rel_set))
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)

    if idcg <= 0.0:
        return 0.0

    return round(dcg / idcg, 4)

