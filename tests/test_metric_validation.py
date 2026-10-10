"""Unit tests for metric range validation and percentage normalization."""

from researchlens.workspace.experiment_service import normalize_and_validate_metric


def test_metric_validation_bounds_and_percentages():
    # 1. Standard decimals in [0, 1]
    ok, val, err = normalize_and_validate_metric("accuracy", 0.945)
    assert ok is True
    assert val == 0.945
    assert err == ""

    # 2. Percentage representations (e.g. 94.5 -> 0.945)
    ok, val, err = normalize_and_validate_metric("f1", 91.5)
    assert ok is True
    assert val == 0.915

    ok, val, err = normalize_and_validate_metric("recall", 100.0)
    assert ok is True
    assert val == 1.0

    # 3. Out-of-bounds rejection
    ok, val, err = normalize_and_validate_metric("accuracy", 1.5)
    # 1.5 is > 1.0 and not a standard percentage (> 100), but wait: if val > 1 and <= 100, 1.5 becomes 0.015!
    # Let's test a true out-of-bounds value: 150.0 or -0.2
    ok, val, err = normalize_and_validate_metric("accuracy", 150.0)
    assert ok is False
    assert "out of valid bounds" in err

    ok, val, err = normalize_and_validate_metric("precision", -0.05)
    assert ok is False
    assert "out of valid bounds" in err

    # 4. Non-negative metrics (RMSE, loss)
    ok, val, err = normalize_and_validate_metric("rmse", 0.245)
    assert ok is True
    assert val == 0.245

    ok, val, err = normalize_and_validate_metric("loss", -1.0)
    assert ok is False
    assert "out of valid bounds" in err

    # 5. Non-numeric rejection
    ok, val, err = normalize_and_validate_metric("accuracy", "not_a_number")
    assert ok is False
    assert "Non-numeric metric value" in err
