"""Summarization evaluation metrics (ROUGE-1, ROUGE-2, ROUGE-L)."""

from typing import Dict, List

from rouge_score import rouge_scorer


def compute_rouge_scores(
    predictions: List[str], references: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute mean ROUGE-1, ROUGE-2, and ROUGE-L scores for predictions against references."""
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must equal number of references")

    if not predictions:
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    accumulators = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    }

    count = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for metric, val in scores.items():
            accumulators[metric]["precision"] += val.precision
            accumulators[metric]["recall"] += val.recall
            accumulators[metric]["fmeasure"] += val.fmeasure

    return {
        metric: {
            "precision": round(vals["precision"] / count, 4),
            "recall": round(vals["recall"] / count, 4),
            "fmeasure": round(vals["fmeasure"] / count, 4),
        }
        for metric, vals in accumulators.items()
    }
