"""Scientific comparability logic for cross-paper result and metric evaluations."""

from typing import Tuple


def check_scientific_comparability(
    metric_a: str,
    dataset_a: str,
    metric_b: str,
    dataset_b: str,
    task_a: str = "",
    task_b: str = "",
) -> Tuple[bool, str]:
    """Determine whether two empirical findings across distinct papers are scientifically comparable.

    Rule:
        - Never rank or directly compare raw numbers unless task, dataset, and metric align.
        - If metrics or datasets differ, label 'Not directly comparable' with explicit reason.

    Returns:
        Tuple of (is_comparable, explanation_reason)
    """
    m_a = (metric_a or "").strip().lower()
    m_b = (metric_b or "").strip().lower()
    d_a = (dataset_a or "").strip().lower()
    d_b = (dataset_b or "").strip().lower()

    if not m_a or not m_b or m_a in ("not detected", "unknown") or m_b in ("not detected", "unknown"):
        return False, "Missing metric specifications for comparative analysis."

    if m_a != m_b:
        return False, f"Incompatible evaluation metrics ('{metric_a}' vs '{metric_b}')."

    if not d_a or not d_b or d_a in ("not detected", "unknown") or d_b in ("not detected", "unknown"):
        return False, "Dataset context unverified; numerical comparison may be misleading."

    # Normalize dataset names (e.g. squad v2.0 vs squad)
    def norm_ds(ds: str) -> str:
        return ds.replace("v1.1", "").replace("v2.0", "").replace("-1k", "").strip()

    if norm_ds(d_a) != norm_ds(d_b):
        return False, f"Distinct evaluation datasets ('{dataset_a}' vs '{dataset_b}'). Results are not directly comparable."

    return True, f"Directly comparable: Both studies report '{metric_a}' on '{dataset_a}'."
