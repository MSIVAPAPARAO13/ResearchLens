"""Reproducibility Replay, Difference Analysis, and Drift Detection (Phase 14)."""

from typing import Any, Dict, List, Optional

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExperimentRun,
)

logger = get_logger("reproducibility_service")


class ReproducibilityDiffService:
    """Computes exact diffs between an original experiment run and a reproduced run."""

    @staticmethod
    def compare_runs(
        original_run: ExperimentRun,
        replay_run: ExperimentRun,
        original_manifest: Optional[Dict[str, Any]] = None,
        replay_manifest: Optional[Dict[str, Any]] = None,
        original_metrics: Optional[Dict[str, float]] = None,
        replay_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compare two runs across metrics, parameters, dataset hash, model version, code commit, environment, and artifacts.

        Returns status MATCH, DIFFERENCE, or MISSING for each dimension.
        """
        diff_report: Dict[str, Any] = {
            "original_run_id": original_run.run_id,
            "replay_run_id": replay_run.run_id,
            "overall_reproduced": True,
            "categories": {},
        }

        def check_field(field_name: str, orig_val: Any, replay_val: Any) -> Dict[str, Any]:
            if orig_val is None or orig_val in ["NOT_AVAILABLE", "NOT PROVIDED", ""]:
                status = "MISSING"
                is_match = False
            elif orig_val == replay_val:
                status = "MATCH"
                is_match = True
            else:
                status = "DIFFERENCE"
                is_match = False

            return {
                "field": field_name,
                "original": orig_val,
                "replay": replay_val,
                "status": status,
                "is_match": is_match,
            }

        # 1. Code Commit
        orig_commit = original_run.git_commit_sha
        replay_commit = replay_run.git_commit_sha
        diff_report["categories"]["code_commit"] = check_field("code_commit", orig_commit, replay_commit)

        # 2. Seed
        diff_report["categories"]["seed"] = check_field("seed", original_run.seed, replay_run.seed)

        # 3. Environment
        orig_env = original_run.environment
        replay_env = replay_run.environment
        diff_report["categories"]["environment"] = check_field("environment", orig_env, replay_env)

        # 4. Backend
        diff_report["categories"]["execution_backend"] = check_field(
            "execution_backend",
            getattr(original_run, "execution_backend", "LOCAL"),
            getattr(replay_run, "execution_backend", "LOCAL"),
        )

        # 5. Dataset Hash
        orig_ds_hash = original_manifest.get("dataset_hash") if original_manifest else None
        replay_ds_hash = replay_manifest.get("dataset_hash") if replay_manifest else None
        diff_report["categories"]["dataset_hash"] = check_field("dataset_hash", orig_ds_hash, replay_ds_hash)

        # 6. Model Version
        orig_model_ver = original_manifest.get("model_version") if original_manifest else None
        replay_model_ver = replay_manifest.get("model_version") if replay_manifest else None
        diff_report["categories"]["model_version"] = check_field("model_version", orig_model_ver, replay_model_ver)

        # 7. Metrics comparison
        metrics_diff: Dict[str, Any] = {}
        if original_metrics and replay_metrics:
            all_keys = set(original_metrics.keys()).union(set(replay_metrics.keys()))
            for m in all_keys:
                orig_m = original_metrics.get(m)
                rep_m = replay_metrics.get(m)
                if orig_m is None or rep_m is None:
                    m_status = "MISSING"
                elif abs(orig_m - rep_m) < 1e-4:
                    m_status = "MATCH"
                else:
                    m_status = "DIFFERENCE"
                metrics_diff[m] = {
                    "original": orig_m,
                    "replay": rep_m,
                    "diff": (rep_m - orig_m) if (orig_m is not None and rep_m is not None) else None,
                    "status": m_status,
                }
        diff_report["metrics"] = metrics_diff

        # 8. Artifacts comparison
        orig_arts = original_manifest.get("artifact_hashes", {}) if original_manifest else {}
        replay_arts = replay_manifest.get("artifact_hashes", {}) if replay_manifest else {}
        art_diff: Dict[str, Any] = {}
        for art_name in set(orig_arts.keys()).union(set(replay_arts.keys())):
            ohash = orig_arts.get(art_name)
            rhash = replay_arts.get(art_name)
            if not ohash or not rhash:
                astatus = "MISSING"
            elif ohash == rhash:
                astatus = "MATCH"
            else:
                astatus = "DIFFERENCE"
            art_diff[art_name] = {
                "original_hash": ohash,
                "replay_hash": rhash,
                "status": astatus,
            }
        diff_report["artifacts"] = art_diff

        # Determine overall reproducibility
        mismatches = [
            v["status"]
            for v in diff_report["categories"].values()
            if v["status"] in ["DIFFERENCE", "MISSING"] and v["field"] not in ["code_commit", "environment"]
        ]
        diff_report["overall_reproduced"] = len(mismatches) == 0

        return diff_report


class ReproducibilityDriftDetector:
    """Detects deviations in dataset, code, dependencies, model, or configuration.

    Explicitly: Do not call this model drift.
    """

    @staticmethod
    def detect_drift(
        original_manifest: Dict[str, Any],
        current_dataset_hash: Optional[str] = None,
        current_code_commit: Optional[str] = None,
        current_model_version: Optional[str] = None,
        current_configuration: Optional[Dict[str, Any]] = None,
        current_dependencies: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify drift events between the recorded baseline run and current workspace state."""
        drift_events: List[Dict[str, Any]] = []

        # 1. Dataset Drift
        orig_ds_hash = original_manifest.get("dataset_hash")
        if orig_ds_hash and orig_ds_hash not in ["NOT_AVAILABLE", "NOT PROVIDED"]:
            if current_dataset_hash and current_dataset_hash != orig_ds_hash:
                drift_events.append({
                    "category": "DATASET",
                    "status": "DATASET_CHANGED",
                    "original": orig_ds_hash,
                    "current": current_dataset_hash,
                    "message": f"Dataset hash changed from '{orig_ds_hash}' to '{current_dataset_hash}'.",
                })

        # 2. Code Drift
        orig_commit = original_manifest.get("code_commit") or original_manifest.get("commit")
        if orig_commit and orig_commit not in ["NOT_AVAILABLE", "NOT PROVIDED"]:
            if current_code_commit and current_code_commit != orig_commit:
                drift_events.append({
                    "category": "CODE",
                    "status": "CODE_CHANGED",
                    "original": orig_commit,
                    "current": current_code_commit,
                    "message": f"Source code HEAD commit changed from '{orig_commit}' to '{current_code_commit}'.",
                })

        # 3. Model Drift
        orig_model_ver = original_manifest.get("model_version")
        if orig_model_ver and orig_model_ver not in ["NOT_AVAILABLE", "NOT PROVIDED"]:
            if current_model_version and current_model_version != orig_model_ver:
                drift_events.append({
                    "category": "MODEL",
                    "status": "MODEL_CHANGED",
                    "original": orig_model_ver,
                    "current": current_model_version,
                    "message": f"Model version changed from '{orig_model_ver}' to '{current_model_version}'.",
                })

        # 4. Configuration Drift
        orig_config = original_manifest.get("configuration", {})
        if current_configuration is not None and orig_config:
            if orig_config != current_configuration:
                drift_events.append({
                    "category": "CONFIGURATION",
                    "status": "CONFIGURATION_CHANGED",
                    "original": orig_config,
                    "current": current_configuration,
                    "message": "Experiment configuration parameters do not match original baseline.",
                })

        # 5. Dependency Drift
        if current_dependencies and "package_versions" in original_manifest:
            orig_pkgs = original_manifest.get("package_versions", {})
            changed_pkgs = {}
            for pkg, ver in orig_pkgs.items():
                cur_ver = current_dependencies.get(pkg)
                if cur_ver and cur_ver != ver:
                    changed_pkgs[pkg] = {"original": ver, "current": cur_ver}
            if changed_pkgs:
                drift_events.append({
                    "category": "DEPENDENCIES",
                    "status": "DEPENDENCIES_CHANGED",
                    "changed_packages": changed_pkgs,
                    "message": f"Detected {len(changed_pkgs)} modified dependency versions.",
                })

        return drift_events
