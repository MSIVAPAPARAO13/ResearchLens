"""Evaluation metrics and benchmark runner package."""

from researchlens.evaluation.evaluation_runner import EvaluationRunner
from researchlens.evaluation.pipeline import (
    AutomatedEvaluationPipeline,
    NLPEvaluationTemplates,
)
from researchlens.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from researchlens.evaluation.summarization_metrics import compute_rouge_scores

__all__ = [
    "compute_rouge_scores",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "hit_rate_at_k",
    "ndcg_at_k",
    "EvaluationRunner",
    "AutomatedEvaluationPipeline",
    "NLPEvaluationTemplates",
]
