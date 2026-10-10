"""Unit tests for baseline comparison and scientific comparability enforcement."""

from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import ExperimentResult


def test_baseline_comparison_and_comparability():
    # Proposed model results
    proposed = [
        ExperimentResult("r_prop_1", "run_01", "RoBERTa-large", "SQuAD", "f1", 0.915),
        ExperimentResult("r_prop_2", "run_01", "RoBERTa-large", "SQuAD", "exact_match", 0.840),
        ExperimentResult("r_prop_3", "run_01", "RoBERTa-large", "BioASQ", "f1", 0.880),  # No baseline for BioASQ
    ]

    # Baseline model results
    baselines = [
        ExperimentResult("r_base_1", "run_00", "BERT-base", "SQuAD", "f1", 0.895),
        ExperimentResult("r_base_2", "run_00", "BERT-base", "SQuAD", "exact_match", 0.810),
    ]

    comps = ExperimentService.compare_baseline(proposed, baselines)
    assert len(comps) == 3

    # Check SQuAD F1 comparison
    f1_comp = next(c for c in comps if c["dataset"] == "SQuAD" and c["metric"] == "f1")
    assert f1_comp["comparability"] == "DIRECTLY_COMPARABLE"
    assert f1_comp["proposed_value"] == 0.915
    assert f1_comp["baseline_value"] == 0.895
    assert round(f1_comp["absolute_diff"], 4) == 0.0200
    # Relative improvement: (0.020 / 0.895) * 100 = 2.23%
    assert 2.20 <= f1_comp["relative_improvement_pct"] <= 2.25

    # Check BioASQ where no baseline match exists
    bio_comp = next(c for c in comps if c["dataset"] == "BioASQ")
    assert bio_comp["comparability"] == "NO_BASELINE_MATCH"
    assert bio_comp["baseline_value"] is None
    assert bio_comp["absolute_diff"] is None
