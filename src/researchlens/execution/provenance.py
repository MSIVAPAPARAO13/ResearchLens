"""Reproducibility manifest creation and provenance auditing (Phase 12)."""

from typing import Any, Dict, List, Optional

from researchlens.workspace.schema import (
    DatasetReference,
    ExperimentArtifact,
    ExperimentPlan,
    ExperimentRun,
)

CANONICAL_REPRODUCIBILITY_FIELDS = [
    "code_commit",
    "dataset",
    "dataset_hash",
    "model",
    "model_version",
    "configuration",
    "seed",
    "python_version",
    "hardware",
    "artifacts",
]


def generate_reproducibility_manifest(
    experiment: ExperimentPlan,
    run: ExperimentRun,
    env_snapshot: Dict[str, Any],
    artifacts: List[ExperimentArtifact],
    dataset_ref: Optional[DatasetReference] = None,
) -> Dict[str, Any]:
    """Generate the canonical execution reproducibility manifest (reproducibility.json)."""
    # Dataset information
    ds_name = dataset_ref.name if dataset_ref else (experiment.dataset or "NOT PROVIDED")
    ds_hash = dataset_ref.hash if dataset_ref and dataset_ref.hash else "NOT PROVIDED"
    ds_ver = dataset_ref.version if dataset_ref else "NOT PROVIDED"

    # Artifact hashes
    artifact_entries = [
        {"name": a.name, "type": a.type, "sha256": a.sha256, "size_bytes": a.size_bytes}
        for a in artifacts
    ]
    artifact_hashes = {a.name: a.sha256 for a in artifacts}

    code_val = experiment.entrypoint or run.entrypoint or "NOT PROVIDED"
    commit_val = run.git_commit_sha if run.git_commit_sha != "NOT_AVAILABLE" else env_snapshot.get("git_commit", "NOT_AVAILABLE")

    manifest = {
        "run_id": run.run_id,
        "experiment_id": experiment.experiment_id,
        "entrypoint": code_val,
        "code": code_val,
        "commit": commit_val,
        "code_commit": commit_val,
        "dataset": ds_name,
        "dataset_version": ds_ver,
        "dataset_hash": ds_hash,
        "model": experiment.model or "NOT PROVIDED",
        "model_version": experiment.variables.get("model_version", "1.0") if experiment.variables else "1.0",
        "configuration": experiment.configuration,
        "seed": run.seed if run.seed is not None else "SEED NOT PROVIDED",
        "environment": run.environment or env_snapshot.get("os", "NOT PROVIDED"),
        "python_version": env_snapshot.get("python_version", "NOT PROVIDED"),
        "package_versions_count": len(env_snapshot.get("package_versions", {})),
        "hardware": run.hardware or env_snapshot.get("hardware", "NOT PROVIDED"),
        "execution_time_seconds": run.duration_seconds,
        "execution_backend": getattr(run, "execution_backend", "LOCAL") or "LOCAL",
        "execution_image": getattr(run, "execution_image", None) or "NOT_PROVIDED",
        "image_digest": getattr(run, "image_digest", None) or "NOT_PROVIDED",
        "MLflow_run_id": getattr(run, "mlflow_run_id", None) or "NOT_AVAILABLE",
        "artifact_hashes": artifact_hashes,
        "artifacts_count": len(artifacts),
        "artifacts": artifact_entries,
    }

    # Audit completeness
    completeness = calculate_reproducibility_completeness_score(manifest)
    manifest["reproducibility_score"] = completeness

    return manifest


def calculate_reproducibility_completeness_score(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the Reproducibility Completeness Score (e.g. 8/10 metadata fields available).

    Note: This is an evidence availability count, NOT a statistical probability.
    """
    field_checks = {
        "code_commit": bool(manifest.get("code_commit") and manifest["code_commit"] not in ["NOT_AVAILABLE", "NOT PROVIDED"]),
        "dataset": bool(manifest.get("dataset") and manifest["dataset"] != "NOT PROVIDED"),
        "dataset_hash": bool(manifest.get("dataset_hash") and manifest["dataset_hash"] != "NOT PROVIDED"),
        "model": bool(manifest.get("model") and manifest["model"] != "NOT PROVIDED"),
        "model_version": bool(manifest.get("model_version") and manifest["model_version"] != "NOT PROVIDED"),
        "configuration": bool(manifest.get("configuration")),
        "seed": bool(manifest.get("seed") and manifest["seed"] != "SEED NOT PROVIDED"),
        "python_version": bool(manifest.get("python_version") and manifest["python_version"] != "NOT PROVIDED"),
        "hardware": bool(manifest.get("hardware") and manifest["hardware"] != "NOT PROVIDED"),
        "artifacts": bool(manifest.get("artifacts")),
    }

    available_count = sum(1 for v in field_checks.values() if v)
    total_fields = len(field_checks)

    if available_count == total_fields:
        status = "COMPLETE"
    elif available_count >= 6:
        status = "PARTIAL"
    else:
        status = "INSUFFICIENT"

    checklist = [{"field": k, "present": v, "value": manifest.get(k, "NOT PROVIDED")} for k, v in field_checks.items()]

    return {
        "score": available_count,
        "available_fields": available_count,
        "total_fields": total_fields,
        "score_label": f"{available_count}/{total_fields} metadata fields available",
        "ratio": available_count / total_fields if total_fields else 0.0,
        "status": status,
        "checks": field_checks,
        "checklist": checklist,
    }
