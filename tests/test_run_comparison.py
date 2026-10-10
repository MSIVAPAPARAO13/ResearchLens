"""Unit tests for longitudinal run comparison diffs."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.monitoring_service import detect_paper_changes


def test_compare_runs_diff():
    # Run 1 papers
    run1_papers = [
        ExternalPaper("p1", "Foundational Deep Learning", doi="10.1000/fdl", citation_count=50),
        ExternalPaper("p2", "Early Neural Networks", doi="10.1000/enn", citation_count=100),
    ]

    # Run 2 papers (p1 with updated citations, p2 missing, p3 newly added)
    run2_papers = [
        ExternalPaper("p1", "Foundational Deep Learning", doi="10.1000/fdl", citation_count=75),
        ExternalPaper("p3", "Modern State Space Models", doi="10.1000/ssm", citation_count=5),
    ]

    history_from_run1 = {f"doi:{p.doi}": p for p in run1_papers}

    # Detect what changed from Run 1 to Run 2
    new_papers, updated_papers, existing_unmod = detect_paper_changes(
        run2_papers, history_from_run1
    )

    assert len(new_papers) == 1
    assert new_papers[0].doi == "10.1000/ssm"

    assert len(updated_papers) == 1
    p_up, diffs = updated_papers[0]
    assert p_up.doi == "10.1000/fdl"
    assert diffs["citation_count"]["new"] == 75


def test_experiment_run_comparison_statistics():
    from researchlens.workspace.experiment_service import ExperimentService
    from researchlens.workspace.schema import ExperimentResult

    # Simulate 3 runs of an experiment with multiple seeds
    results = [
        ExperimentResult(result_id="r1", run_id="run_1", model="ModelA", dataset="SQuAD", metric="f1", value=0.910, seed=42),
        ExperimentResult(result_id="r2", run_id="run_2", model="ModelA", dataset="SQuAD", metric="f1", value=0.914, seed=123),
        ExperimentResult(result_id="r3", run_id="run_3", model="ModelA", dataset="SQuAD", metric="f1", value=0.920, seed=456),
    ]

    stats = ExperimentService.calculate_run_statistics(results)
    assert "f1" in stats
    f1_stat = stats["f1"]
    assert f1_stat["count"] == 3
    assert f1_stat["mean"] == 0.9147
    assert f1_stat["min"] == 0.910
    assert f1_stat["max"] == 0.920
    assert f1_stat["std"] > 0.0
    assert f1_stat["has_repeated_runs"] is True
    # Ensure no fabricated significance or p-value
    assert "p_value" not in f1_stat
