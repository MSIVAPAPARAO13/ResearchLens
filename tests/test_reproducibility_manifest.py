"""Unit tests for Reproducibility Manifest and Completeness Score (Phase 12)."""

from researchlens.execution.provenance import (
    calculate_reproducibility_completeness_score,
    generate_reproducibility_manifest,
)
from researchlens.workspace.schema import (
    DatasetReference,
    ExperimentArtifact,
    ExperimentPlan,
    ExperimentRun,
)


def test_reproducibility_manifest_and_score():
    plan = ExperimentPlan(
        experiment_id="exp_01",
        name="Reproducibility Test",
        dataset="SQuAD",
        model="RoBERTa",
        entrypoint="scripts/experiments/deterministic_demo.py",
        configuration={"lr": 1e-4, "batch_size": 32},
    )
    run = ExperimentRun(
        run_id="run_rep_01",
        experiment_id="exp_01",
        git_commit_sha="a1b2c3d4e5",
        hardware="NVIDIA H100 80GB",
        seed=42,
        duration_seconds=12.5,
    )
    env_snapshot = {
        "python_version": "3.11.8",
        "git_commit": "a1b2c3d4e5",
        "hardware": "NVIDIA H100 80GB",
        "package_versions": {"torch": "2.2.0"},
    }
    artifacts = [
        ExperimentArtifact("art_1", "run_rep_01", "results.json", "results.json", "METRICS", 100, "abc1234"),
    ]
    ds_ref = DatasetReference(
        dataset_id="ds_01",
        name="SQuAD",
        version="v2.0",
        hash="sha256_dataset_hash",
    )

    manifest = generate_reproducibility_manifest(
        experiment=plan,
        run=run,
        env_snapshot=env_snapshot,
        artifacts=artifacts,
        dataset_ref=ds_ref,
    )

    assert manifest["run_id"] == "run_rep_01"
    assert manifest["code_commit"] == "a1b2c3d4e5"
    assert manifest["dataset_version"] == "v2.0"
    assert manifest["dataset_hash"] == "sha256_dataset_hash"
    assert manifest["seed"] == 42

    score = manifest["reproducibility_score"]
    assert score["available_fields"] == 10
    assert score["total_fields"] == 10
    assert score["status"] == "COMPLETE"
    assert score["score_label"] == "10/10 metadata fields available"


def test_reproducibility_completeness_partial_and_insufficient():
    # Incomplete manifest (missing code commit, seed, hardware, dataset hash)
    manifest_partial = {
        "code_commit": "NOT_AVAILABLE",
        "dataset": "GLUE",
        "dataset_hash": "NOT PROVIDED",
        "model": "BERT",
        "model_version": "1.0",
        "configuration": {"lr": 1e-4},
        "seed": "SEED NOT PROVIDED",
        "python_version": "3.11",
        "hardware": "NOT PROVIDED",
        "artifacts": [{"name": "out.txt"}],
    }
    score_p = calculate_reproducibility_completeness_score(manifest_partial)
    assert score_p["status"] == "PARTIAL"
    assert score_p["available_fields"] >= 4
