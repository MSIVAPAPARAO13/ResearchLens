"""Transparent heuristic gap scoring and confidence calibration module."""

from typing import Tuple

from researchlens.analysis.research_gap_schema import ResearchGap


def calculate_gap_score(gap: ResearchGap) -> Tuple[float, str]:
    """Calculate transparent heuristic score and confidence level for a ResearchGap.

    Scoring Formula:
        score = 0.30 * evidence_strength
              + 0.20 * specificity
              + 0.20 * recurrence
              + 0.15 * impact
              + 0.15 * researchability

    Returns:
        Tuple of (confidence_score, confidence_level)
    """
    # 1. Evidence Strength (0.0 to 1.0)
    if gap.explicit_or_inferred == "EXPLICIT":
        evidence_strength = 0.95 if gap.supporting_evidence else 0.85
    elif gap.gap_type in ("MISSING_EVALUATION", "MISSING_COMPARISON"):
        evidence_strength = 0.75 if len(gap.supporting_evidence) >= 2 else 0.65
    elif gap.supporting_evidence:
        evidence_strength = 0.70
    else:
        evidence_strength = 0.30

    # 2. Specificity (0.0 to 1.0)
    has_method = bool(gap.affected_method and gap.affected_method != "Not detected")
    has_dataset = bool(gap.affected_dataset and gap.affected_dataset != "Not detected")
    has_metric = bool(gap.affected_metric and gap.affected_metric != "Not detected")

    spec_count = sum([has_method, has_dataset, has_metric])
    if spec_count >= 2:
        specificity = 0.90
    elif spec_count == 1:
        specificity = 0.75
    elif len(gap.statement.split()) >= 8:
        specificity = 0.60
    else:
        specificity = 0.40

    # 3. Recurrence / Evidence Count (0.0 to 1.0)
    ev_count = len(gap.supporting_evidence)
    if ev_count >= 3 or gap.cross_paper_frequency > 1:
        recurrence = 0.95
    elif ev_count == 2:
        recurrence = 0.80
    elif ev_count == 1:
        recurrence = 0.65
    else:
        recurrence = 0.35

    # 4. Impact (0.0 to 1.0)
    if gap.category in ("GENERALIZATION", "METHODOLOGY", "EVALUATION", "ROBUSTNESS"):
        impact = 0.85
    elif gap.category in ("DATASET", "BASELINE", "REPRODUCIBILITY"):
        impact = 0.75
    else:
        impact = 0.60

    # 5. Researchability (0.0 to 1.0)
    if gap.potential_direction and len(gap.potential_direction.split()) >= 6:
        researchability = 0.90
    elif gap.potential_direction:
        researchability = 0.75
    else:
        researchability = 0.50

    # Weighted Sum
    score = (
        0.30 * evidence_strength
        + 0.20 * specificity
        + 0.20 * recurrence
        + 0.15 * impact
        + 0.15 * researchability
    )

    score = round(min(max(score, 0.0), 1.0), 4)

    # Confidence Level Mapping
    if score >= 0.78:
        level = "High"
    elif score >= 0.55:
        level = "Medium"
    else:
        level = "Low"

    return score, level
