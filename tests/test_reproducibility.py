"""Unit tests for experiment reproducibility assessment."""

from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import ExperimentRun


def test_reproducibility_assessment_levels():
    # 1. Complete reproducibility metadata
    complete_run = ExperimentRun(
        run_id="run_complete",
        experiment_id="exp_01",
        environment="Ubuntu 22.04 LTS, Python 3.11",
        hardware="NVIDIA H100 80GB SXM5",
        software_versions={"torch": "2.2.0", "transformers": "4.38.0"},
        seed=42,
    )
    rep_complete = ExperimentService.assess_reproducibility(complete_run)
    assert rep_complete.completeness_status == "COMPLETE"
    assert len(rep_complete.missing_fields) == 0

    # 2. Partial metadata (missing seed and hardware)
    partial_run = ExperimentRun(
        run_id="run_partial",
        experiment_id="exp_01",
        environment="macOS Sonoma, Python 3.11",
        software_versions={"torch": "2.2.0"},
    )
    rep_partial = ExperimentService.assess_reproducibility(partial_run)
    assert rep_partial.completeness_status == "PARTIAL"
    assert "random_seed" in rep_partial.missing_fields
    assert "hardware_specification" in rep_partial.missing_fields

    # 3. Insufficient metadata (empty run)
    empty_run = ExperimentRun(
        run_id="run_empty",
        experiment_id="exp_01",
    )
    rep_insufficient = ExperimentService.assess_reproducibility(empty_run)
    assert rep_insufficient.completeness_status == "INSUFFICIENT"
    assert len(rep_insufficient.missing_fields) >= 3
    # Metadata values must NOT be fabricated!
    assert rep_insufficient.metadata_summary["seed"] == "NOT PROVIDED"
    assert rep_insufficient.metadata_summary["hardware"] == "NOT PROVIDED"
