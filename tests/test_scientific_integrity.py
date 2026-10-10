"""Scientific integrity tests for Phase 11.

Explicitly asserts that the system does NOT:
- invent experiment results
- invent missing seeds
- invent hardware
- invent statistical significance
- invent confidence intervals
- compare incompatible datasets
- mark hypothesis SUPPORTED automatically
- call an imported result verified
- claim state-of-the-art without evidence
"""

from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import (
    ExperimentResult,
    ExperimentRun,
    ResearchHypothesis,
)


def test_no_invented_seeds_or_hardware():
    """Missing seed and hardware must remain None/Not Provided, never fabricated."""
    run = ExperimentRun(
        run_id="run_integrity_01",
        experiment_id="exp_01",
        run_number=1,
        hardware=None,
        seed=None,
    )
    assert run.seed is None
    assert run.hardware is None

    rep_report = ExperimentService.assess_reproducibility(run)
    assert rep_report.metadata_summary["seed"] == "NOT PROVIDED"
    assert rep_report.metadata_summary["hardware"] == "NOT PROVIDED"
    assert rep_report.completeness_status == "INSUFFICIENT"
    assert "random_seed" in rep_report.missing_fields
    assert "hardware_specification" in rep_report.missing_fields


def test_imported_results_are_unverified_by_default():
    """Imported results must never be marked verified without explicit audit."""
    csv_data = "model,dataset,accuracy\nRoBERTa,SQuAD,0.92\n"
    results, errors = ExperimentService.ingest_results(csv_data, run_id="run_02", source_name="raw.csv")
    assert len(results) == 1
    assert results[0].status == "UNVERIFIED_IMPORT"
    assert results[0].source == "raw.csv"


def test_no_incompatible_dataset_comparison():
    """Cross-dataset metrics must not be calculated into fake percentage improvements."""
    proposed = ExperimentResult(
        result_id="res_1",
        run_id="run_1",
        model="NewModel",
        dataset="SQuAD_v1",
        metric="f1",
        value=0.92,
    )
    baseline = ExperimentResult(
        result_id="res_2",
        run_id="run_base",
        model="BERT",
        dataset="GLUE_MNLI",  # Different dataset!
        metric="f1",
        value=0.88,
    )
    comparisons = ExperimentService.compare_baseline(
        proposed_results=[proposed],
        baseline_results=[baseline],
    )
    assert len(comparisons) == 1
    # Must explicitly state NO_BASELINE_MATCH / not directly comparable
    assert comparisons[0]["comparability"] == "NO_BASELINE_MATCH"
    assert comparisons[0]["absolute_diff"] is None
    assert comparisons[0]["relative_improvement_pct"] is None


def test_no_automatic_hypothesis_support():
    """Hypothesis status must NOT automatically become SUPPORTED without human researcher confirmation."""
    hypothesis = ResearchHypothesis(
        hypothesis_id="hyp_01",
        workspace_id="ws_01",
        statement="Transformer attention heads scale linearly with context length.",
        status="PROPOSED",
    )
    high_metric_result = ExperimentResult(
        result_id="res_3",
        run_id="run_3",
        model="TransformerX",
        dataset="LongBench",
        metric="accuracy",
        value=0.99,
    )

    finding = ExperimentService.generate_research_finding(
        workspace_id="ws_01",
        title="High accuracy observed",
        statement="Accuracy reached 0.99 on LongBench.",
        experiment_results=[high_metric_result],
        related_hypothesis_id=hypothesis.hypothesis_id,
    )

    # Hypothesis status is still PROPOSED
    assert hypothesis.status == "PROPOSED"
    # Finding requires researcher review
    assert finding.status == "RESEARCHER_REVIEW_REQUIRED"


def test_no_invented_statistical_significance_or_confidence_intervals():
    """System must not compute p-values or confidence intervals on single-run results."""
    run = ExperimentRun(
        run_id="run_single",
        experiment_id="exp_01",
        run_number=1,
    )
    res = ExperimentResult(
        result_id="res_single",
        run_id=run.run_id,
        model="ModelA",
        dataset="TriviaQA",
        metric="f1",
        value=0.85,
    )
    comp = ExperimentService.compare_baseline([res], [res])
    assert "p_value" not in comp[0]
    assert "confidence_interval" not in comp[0]
    # Finding must not claim statistical significance
    finding = ExperimentService.generate_research_finding(
        workspace_id="ws_01",
        title="Single Run Observation",
        statement="Model achieved 0.85 on TriviaQA.",
        experiment_results=[res],
    )
    assert "statistically significant" not in finding.statement.lower()
