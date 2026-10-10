"""Evaluation Regression Detection (Phase 14)."""

from typing import Any, Dict, List, Optional


class RegressionDetector:
    """Detects empirical performance regressions against baselines or historical thresholds."""

    # Default degradation thresholds (e.g., 0.05 = 5% absolute drop allowed)
    DEFAULT_THRESHOLDS = {
        "mrr": 0.03,
        "f1": 0.05,
        "accuracy": 0.05,
        "precision": 0.05,
        "recall": 0.05,
        "rouge1": 0.04,
        "rouge2": 0.04,
        "rougel": 0.04,
    }

    @staticmethod
    def evaluate_regression(
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compare candidate metrics against baseline metrics and detect degradation.

        Returns status: PASS or REGRESSION_DETECTED.
        Explicitly: Do not automatically reject code.
        """
        active_thresholds = dict(RegressionDetector.DEFAULT_THRESHOLDS)
        if thresholds:
            active_thresholds.update({k.lower(): v for k, v in thresholds.items()})

        regressions: List[Dict[str, Any]] = []
        comparisons: Dict[str, Any] = {}

        for metric_name, base_val in baseline_metrics.items():
            clean_m = metric_name.lower().replace("-", "").replace("_", "")
            curr_val = current_metrics.get(metric_name)

            if curr_val is None:
                # Try normalized match
                for cand_k, cand_v in current_metrics.items():
                    if cand_k.lower().replace("-", "").replace("_", "") == clean_m:
                        curr_val = cand_v
                        break

            if curr_val is not None:
                delta = curr_val - base_val
                allowed_drop = active_thresholds.get(clean_m, 0.05)
                # If negative delta exceeds allowed degradation drop
                has_regression = delta < -allowed_drop

                comp_info = {
                    "metric": metric_name,
                    "baseline": base_val,
                    "current": curr_val,
                    "delta": round(delta, 4),
                    "allowed_drop": allowed_drop,
                    "status": "REGRESSION" if has_regression else "OK",
                }
                comparisons[metric_name] = comp_info

                if has_regression:
                    regressions.append(comp_info)

        status = "REGRESSION_DETECTED" if regressions else "PASS"

        return {
            "status": status,
            "regressions_count": len(regressions),
            "regressions": regressions,
            "comparisons": comparisons,
            "message": (
                f"Evaluation regression detected on {len(regressions)} metrics."
                if regressions
                else "No empirical regressions detected."
            ),
        }
