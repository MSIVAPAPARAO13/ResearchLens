"""Tests for Evaluation Regression Detection (Phase 14)."""

from researchlens.execution.regression import RegressionDetector


def test_no_regression_when_metrics_improve_or_stay_within_threshold():
    """Verify evaluation returns PASS when metrics meet or exceed baseline."""
    baseline = {"MRR": 0.85, "F1": 0.82}
    current = {"MRR": 0.86, "F1": 0.81}  # F1 dropped 0.01 (within 0.05 threshold)

    res = RegressionDetector.evaluate_regression(current_metrics=current, baseline_metrics=baseline)
    assert res["status"] == "PASS"
    assert res["regressions_count"] == 0


def test_regression_detected_when_threshold_exceeded():
    """Verify REGRESSION_DETECTED when performance drops beyond threshold."""
    baseline = {"MRR": 0.85, "F1": 0.82}
    current = {"MRR": 0.75, "F1": 0.80}  # MRR dropped 0.10 (exceeds 0.03 threshold)

    res = RegressionDetector.evaluate_regression(current_metrics=current, baseline_metrics=baseline)
    assert res["status"] == "REGRESSION_DETECTED"
    assert res["regressions_count"] == 1
    reg_metric = res["regressions"][0]
    assert reg_metric["metric"] == "MRR"
    assert reg_metric["delta"] == -0.10


def test_custom_thresholds_override():
    """Verify custom regression threshold overrides."""
    baseline = {"accuracy": 0.90}
    current = {"accuracy": 0.88}  # drop of 0.02

    # With strict 0.01 threshold -> REGRESSION_DETECTED
    strict_res = RegressionDetector.evaluate_regression(
        current_metrics=current,
        baseline_metrics=baseline,
        thresholds={"accuracy": 0.01},
    )
    assert strict_res["status"] == "REGRESSION_DETECTED"

    # With lenient 0.05 threshold -> PASS
    lenient_res = RegressionDetector.evaluate_regression(
        current_metrics=current,
        baseline_metrics=baseline,
        thresholds={"accuracy": 0.05},
    )
    assert lenient_res["status"] == "PASS"
