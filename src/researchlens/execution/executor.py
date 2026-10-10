"""Controlled, sandboxed experiment executor with audit logging and artifact capture (Phase 12)."""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from researchlens.execution.artifacts import discover_artifacts
from researchlens.execution.backends import get_execution_backend
from researchlens.execution.environment import (
    capture_environment_snapshot,
    get_gpu_info,
)
from researchlens.execution.process_monitor import classify_failure
from researchlens.execution.provenance import generate_reproducibility_manifest
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger
from researchlens.workspace.experiment_service import (
    ExperimentService,
    normalize_and_validate_metric,
)
from researchlens.workspace.schema import (
    ExecutionJob,
    ExecutionProfile,
    ExperimentResult,
    ExperimentRun,
    ResearchFinding,
    current_iso_time,
    generate_id,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("experiment_executor")


class ExperimentExecutor:
    """Safely executes registered experiment entrypoints in sandboxed environments (LOCAL, DOCKER, REMOTE)."""

    def __init__(self, store: WorkspaceStore):
        self.store = store
        self.registry_service = ExperimentRegistryService(store)

    def execute_experiment(
        self,
        workspace_id: str,
        experiment_id: str,
        profile_id: Optional[str] = None,
        configuration_id: Optional[str] = None,
        actor_id: str = "system",
        seed: Optional[int] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        image: Optional[str] = None,
        network_policy: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ExperimentRun], str]:
        """Execute a registered experiment across LOCAL, DOCKER, or REMOTE backends."""
        # 1. Verify permissions (OWNER or RESEARCHER)
        if not self.store.has_permission(workspace_id, actor_id, "RESEARCHER"):
            return None, f"Permission denied: Actor '{actor_id}' is not permitted to execute experiments."

        # 2. Retrieve experiment plan
        experiment = self.store.get_experiment(workspace_id, experiment_id)
        if not experiment:
            return None, f"Experiment plan '{experiment_id}' not found."

        entrypoint = experiment.entrypoint or "scripts/experiments/deterministic_demo.py"

        # 3. Verify entrypoint authorization in registry
        is_allowed, reg_err = self.registry_service.validate_execution_allowed(workspace_id, entrypoint)
        if not is_allowed:
            return None, f"Execution blocked: {reg_err}"

        # 4. Validate profile
        profile = self.store.get_execution_profile(workspace_id, profile_id) if profile_id else None
        if not profile:
            profile = ExecutionProfile(
                profile_id="prof_default",
                name="Default Local Profile",
                python_version=sys.version.split()[0],
                environment_type="LOCAL",
            )

        # Check GPU requirement constraint
        if profile.gpu != "None":
            gpu_state = get_gpu_info()
            if not gpu_state.get("available", False):
                blocked_run = ExperimentRun(
                    run_id=generate_id("run"),
                    experiment_id=experiment_id,
                    status="BLOCKED",
                    notes="Requested GPU is not available on this host environment.",
                )
                self.store.save_run(blocked_run, workspace_id)
                return blocked_run, "Execution blocked: Requested GPU is not available."

        # 5. Retrieve configuration
        config_obj = self.store.get_configuration(workspace_id, configuration_id) if configuration_id else None
        params = dict(experiment.configuration)
        if config_obj:
            params.update(config_obj.parameters)
        if custom_params:
            params.update(custom_params)

        effective_seed = seed if seed is not None else experiment.variables.get("seed", 42)
        try:
            effective_seed = int(effective_seed)
        except (ValueError, TypeError):
            effective_seed = None

        # Resolve backend & image selection
        effective_backend = (
            backend
            or (custom_params and custom_params.get("execution_backend"))
            or getattr(experiment, "execution_backend", None)
            or (profile and getattr(profile, "environment_type", None))
            or "LOCAL"
        ).strip().upper()

        effective_image = (
            image
            or (custom_params and custom_params.get("execution_image"))
            or getattr(experiment, "execution_image", None)
            or (effective_backend == "DOCKER" and "researchlens/nlp-experiment:1.0")
        )

        effective_net_policy = (
            network_policy
            or (custom_params and custom_params.get("network_policy"))
            or "DISABLED"
        ).strip().upper()

        effective_resources = dict(resources or getattr(experiment, "resource_requirements", {}) or {})

        # 6. Initialize run record and sandboxed workspace
        run_id = generate_id("run")
        run_dir = self.store.get_run_dir(workspace_id, run_id)
        artifacts_dir = os.path.join(run_dir, "artifacts")

        # 7. Capture Environment Snapshot
        env_snapshot = capture_environment_snapshot(working_dir=run_dir)
        env_snapshot["execution_profile"] = profile.to_dict()
        with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
            json.dump(env_snapshot, f, indent=2)

        # Save configuration used
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        # 8. Construct argument array (shell=False)
        cmd_args = [sys.executable, entrypoint, "--output-dir", artifacts_dir]
        if effective_seed is not None:
            cmd_args.extend(["--seed", str(effective_seed)])
        if experiment.dataset:
            cmd_args.extend(["--dataset", str(experiment.dataset)])

        for p_key, p_val in params.items():
            if p_key in ["seed", "dataset", "output-dir", "output_dir", "execution_backend", "execution_image", "network_policy"]:
                continue
            clean_k = str(p_key).strip().replace("_", "-")
            cmd_args.extend([f"--{clean_k}", str(p_val)])

        # Security check: validate arguments against shell metacharacters
        is_safe_args, arg_err = SecurityValidator.validate_command_arguments(cmd_args)
        if not is_safe_args:
            return None, f"Security rejection: {arg_err}"

        # 9. Instantiate ExecutionJob (Phase 14)
        job_id = generate_id("job")
        job = ExecutionJob(
            job_id=job_id,
            workspace_id=workspace_id,
            experiment_id=experiment_id,
            run_id=run_id,
            backend=effective_backend,
            image=effective_image,
            command=cmd_args,
            resources=effective_resources,
            environment=SecurityValidator.sanitize_env_vars(dict(os.environ)),
            network_policy=effective_net_policy,
            timeout=experiment.timeout_seconds,
            status="CREATED",
        )
        job.transition_to("CREATED", "Execution job initialized")
        self.store.save_job(job, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action="JOB_CREATED",
            target_type="EXECUTION_JOB",
            target_id=job.job_id,
            metadata={"backend": effective_backend, "experiment_id": experiment_id},
        )

        # 10. Instantiate and validate backend
        backend_obj = get_execution_backend(effective_backend, store=self.store)

        job.transition_to("VALIDATING", f"Validating on {effective_backend} backend")
        is_backend_valid, backend_val_err = backend_obj.validate(job, experiment)
        if not is_backend_valid:
            job.transition_to("BLOCKED", backend_val_err)
            self.store.save_job(job, workspace_id)
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="JOB_FAILED",
                target_type="EXECUTION_JOB",
                target_id=job.job_id,
                metadata={"reason": backend_val_err},
            )
            blocked_run = ExperimentRun(
                run_id=run_id,
                experiment_id=experiment_id,
                status="BLOCKED",
                notes=f"Backend validation failed: {backend_val_err}",
                execution_backend=effective_backend,
                execution_image=effective_image,
                job_id=job_id,
            )
            self.store.save_run(blocked_run, workspace_id)
            return blocked_run, f"Execution blocked: {backend_val_err}"

        # 11. Prepare Backend
        job.transition_to("PREPARING", "Preparing execution environment")
        prep_ok, prep_err = backend_obj.prepare(job, run_dir)
        if not prep_ok:
            job.transition_to("FAILED", prep_err)
            self.store.save_job(job, workspace_id)
            return None, f"Backend preparation failed: {prep_err}"

        # Serialize execution_job.json before execution
        with open(os.path.join(run_dir, "execution_job.json"), "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, indent=2)

        # 12. Start Execution
        job.transition_to("RUNNING", "Started backend execution")
        self.store.save_job(job, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action="JOB_STARTED",
            target_type="EXECUTION_JOB",
            target_id=job.job_id,
            metadata={"backend": effective_backend},
        )
        if effective_backend == "DOCKER":
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="CONTAINER_STARTED",
                target_type="CONTAINER",
                target_id=job.job_id,
                metadata={"image": effective_image, "network": effective_net_policy},
            )

        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            run_number=len(self.store.list_runs(workspace_id)) + 1,
            started_at=current_iso_time(),
            status="RUNNING",
            environment=f"{env_snapshot['os']} {env_snapshot['os_release']}, Python {env_snapshot['python_version']} [{effective_backend}]",
            hardware=f"CPU: {env_snapshot['cpu_count']} Cores | GPU: {env_snapshot['gpu']['name']}",
            software_versions={
                "python": env_snapshot["python_version"],
            },
            seed=effective_seed,
            git_commit_sha=env_snapshot["git_commit"],
            entrypoint=entrypoint,
            command_executed=" ".join(cmd_args),
            execution_backend=effective_backend,
            execution_image=effective_image,
            job_id=job_id,
        )
        self.store.save_run(run, workspace_id)

        exec_res = backend_obj.start(job, run_dir, timeout_seconds=experiment.timeout_seconds)

        if effective_backend == "DOCKER":
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="CONTAINER_COMPLETED",
                target_type="CONTAINER",
                target_id=job.job_id,
                metadata={"exit_code": exec_res.get("exit_code")},
            )

        duration = exec_res.get("duration_seconds", 0.0)
        exit_code = exec_res.get("exit_code", 1)
        stdout_content = exec_res.get("stdout", "")
        stderr_content = exec_res.get("stderr", "")
        timed_out = exec_res.get("timed_out", False)

        if "image_digest" in exec_res:
            run.image_digest = exec_res["image_digest"]

        run.completed_at = current_iso_time()
        run.duration_seconds = duration
        run.exit_code = exit_code

        # 13. Handle timeout / failure
        if timed_out:
            job.transition_to("TIMEOUT", "Process execution timed out")
            self.store.save_job(job, workspace_id)
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="JOB_FAILED",
                target_type="EXECUTION_JOB",
                target_id=job.job_id,
                metadata={"failure_category": "TIMEOUT"},
            )
            run.status = "TIMEOUT"
            run.failure_category = "TIMEOUT"
            self.store.save_run(run, workspace_id)
            return run, "Experiment execution timed out."

        if exit_code != 0:
            run.status = "FAILED"
            run.failure_category = classify_failure(exit_code, stderr=stderr_content, stdout=stdout_content)
            job.transition_to("FAILED", f"Exit code {exit_code}: {run.failure_category}")
            self.store.save_job(job, workspace_id)
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="JOB_FAILED",
                target_type="EXECUTION_JOB",
                target_id=job.job_id,
                metadata={"failure_category": run.failure_category, "exit_code": exit_code},
            )
            self.store.save_run(run, workspace_id)
            return run, f"Experiment failed with exit code {exit_code} ({run.failure_category})."

        # 12. Validate Output Contract (results.json)
        results_file_path = os.path.join(artifacts_dir, "results.json")
        if not os.path.exists(results_file_path):
            # check root of run_dir as fallback
            if os.path.exists(os.path.join(run_dir, "results.json")):
                results_file_path = os.path.join(run_dir, "results.json")
            else:
                run.status = "FAILED"
                run.failure_category = "FAILED_OUTPUT_CONTRACT"
                self.store.save_run(run, workspace_id)
                return run, "Output contract failed: 'results.json' not produced by experiment."

        try:
            with open(results_file_path, "r", encoding="utf-8") as f:
                results_payload = json.load(f)
            if not isinstance(results_payload, dict) or "metrics" not in results_payload:
                run.status = "FAILED"
                run.failure_category = "FAILED_OUTPUT_CONTRACT"
                self.store.save_run(run, workspace_id)
                return run, "Output contract failed: 'results.json' missing 'metrics' object."
        except Exception as e:
            run.status = "FAILED"
            run.failure_category = "FAILED_OUTPUT_CONTRACT"
            self.store.save_run(run, workspace_id)
            return run, f"Failed to parse results.json: {e}"

        # 13. Discover artifacts and compute hashes
        job.transition_to("UPLOADING", "Registering output artifacts")
        discovered_arts = discover_artifacts(artifacts_dir, run_id=run_id)
        for art in discovered_arts:
            self.store.save_artifact(art, workspace_id)
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="ARTIFACT_UPLOADED",
                target_type="ARTIFACT",
                target_id=art.artifact_id,
                metadata={"name": art.name, "sha256": art.sha256},
            )
        run.artifact_ids = [a.artifact_id for a in discovered_arts]

        # 14. Ingest Results
        model_name = experiment.model or "ProposedModel"
        dataset_name = experiment.dataset or "EvaluationDataset"
        ingested_results: List[ExperimentResult] = []

        for metric_name, val_raw in results_payload["metrics"].items():
            is_valid, norm_val, err_msg = normalize_and_validate_metric(metric_name, val_raw)
            if is_valid:
                res_obj = ExperimentResult(
                    result_id=generate_id("res"),
                    run_id=run_id,
                    model=model_name,
                    dataset=dataset_name,
                    metric=metric_name,
                    value=norm_val,
                    split="test",
                    seed=effective_seed,
                    source=f"execution:{entrypoint}",
                    status="UNVERIFIED_IMPORT",
                )
                self.store.save_result(res_obj, workspace_id)
                ingested_results.append(res_obj)

        # 15. Generate Reproducibility Manifest
        dataset_ref = None
        if experiment.dataset_id:
            for d in self.store.list_dataset_refs(workspace_id):
                if d.dataset_id == experiment.dataset_id:
                    dataset_ref = d
                    break

        manifest = generate_reproducibility_manifest(
            experiment=experiment,
            run=run,
            env_snapshot=env_snapshot,
            artifacts=discovered_arts,
            dataset_ref=dataset_ref,
        )
        with open(os.path.join(run_dir, "reproducibility.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        reports_dir = os.path.join(run_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "reproducibility.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2)
        with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
            json.dump(env_snapshot, f, indent=2)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, indent=2)

        # 16. Baseline comparison & Research Finding Candidate
        if ingested_results and experiment.baseline:
            all_results = self.store.list_results(workspace_id)
            baseline_results = [r for r in all_results if r.model == experiment.baseline and r.run_id != run_id]
            if baseline_results:
                comparisons = ExperimentService.compare_baseline(
                    proposed_results=ingested_results,
                    baseline_results=baseline_results,
                )
                comparable = [c for c in comparisons if c.get("comparability") == "DIRECTLY_COMPARABLE"]
                if comparable:
                    comp_top = comparable[0]
                    finding = ResearchFinding(
                        finding_id=generate_id("find"),
                        workspace_id=workspace_id,
                        title=f"Empirical Evaluation: {experiment.model} vs {experiment.baseline}",
                        statement=(
                            f"Model '{experiment.model}' achieved {comp_top['proposed_value']:.4f} {comp_top['metric']} "
                            f"vs baseline '{experiment.baseline}' ({comp_top['baseline_value']:.4f}) on {comp_top['dataset']} "
                            f"(Difference: {comp_top['absolute_diff']:+.4f}, {comp_top['relative_improvement_pct']:+.2f}%)."
                        ),
                        finding_type="COMPARISON",
                        evidence_ids=[],
                        experiment_result_ids=[r.result_id for r in ingested_results],
                        related_hypothesis_id=experiment.hypothesis_id,
                        confidence_level="MEDIUM",
                        status="PENDING_RESEARCHER_REVIEW",
                    )
                    self.store.save_finding(finding)

        run.status = "COMPLETED"
        self.store.save_run(run, workspace_id)

        job.transition_to("COMPLETED", "Execution completed successfully")
        self.store.save_job(job, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action="JOB_COMPLETED",
            target_type="EXECUTION_JOB",
            target_id=job.job_id,
            metadata={"duration": duration, "status": "COMPLETED"},
        )

        # 17. MLflow Synchronization (Phase 13 & 14)
        try:
            from researchlens.mlops.sync import MLflowSyncService
            sync_service = MLflowSyncService(self.store)
            sync_ok, _ = sync_service.sync_run(
                workspace_id=workspace_id,
                run=run,
                experiment=experiment,
                results=ingested_results,
                artifacts=discovered_arts,
                manifest=manifest,
                actor_id=actor_id,
            )
            if sync_ok:
                self.store.log_audit(
                    workspace_id=workspace_id,
                    actor_id=actor_id,
                    action="MLFLOW_REMOTE_SYNCED",
                    target_type="EXPERIMENT_RUN",
                    target_id=run.run_id,
                    metadata={"mlflow_run_id": run.mlflow_run_id},
                )
        except Exception as e:
            logger.warning(f"MLflow synchronization skipped or encountered error: {e}")

        # 18. Log audit event
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action="EXECUTE",
            target_type="EXPERIMENT_RUN",
            target_id=run.run_id,
            metadata={"experiment_id": experiment_id, "duration": duration, "status": run.status},
        )

        return run, ""

    def reproduce_run(
        self,
        workspace_id: str,
        original_run_id: str,
        actor_id: str = "researcher",
    ) -> Tuple[Optional[ExperimentRun], str]:
        """[Reproduce Run] Loads original run config/dataset/model/seed/backend/image and creates a NEW ExperimentRun.

        Guarantees: Never modifies original run.
        """
        orig_run = self.store.get_run(workspace_id, original_run_id)
        if not orig_run:
            return None, f"Original run '{original_run_id}' not found."

        experiment = self.store.get_experiment(workspace_id, orig_run.experiment_id)
        if not experiment:
            return None, f"Original experiment plan '{orig_run.experiment_id}' not found."

        orig_seed = orig_run.seed
        orig_backend = getattr(orig_run, "execution_backend", "LOCAL") or "LOCAL"
        orig_image = getattr(orig_run, "execution_image", None)

        new_run, err = self.execute_experiment(
            workspace_id=workspace_id,
            experiment_id=experiment.experiment_id,
            actor_id=actor_id,
            seed=orig_seed,
            backend=orig_backend,
            image=orig_image,
        )

        if new_run:
            new_run.notes = f"Replay of run '{original_run_id}'"
            self.store.save_run(new_run, workspace_id)
            self.store.log_audit(
                workspace_id=workspace_id,
                actor_id=actor_id,
                action="REPRODUCIBILITY_REPLAYED",
                target_type="EXPERIMENT_RUN",
                target_id=new_run.run_id,
                metadata={"original_run_id": original_run_id},
            )

        return new_run, err

    def cancel_run(
        self,
        workspace_id: str,
        run_id: str,
        actor_id: str = "system",
        reason: str = "User cancelled run.",
    ) -> Tuple[bool, str]:
        """Cancel an active or queued run (restricted to OWNER or RESEARCHER)."""
        if not self.store.has_permission(workspace_id, actor_id, "RESEARCHER"):
            return False, f"Permission denied: Actor '{actor_id}' is not authorized to cancel runs."

        run = self.store.get_run(workspace_id, run_id)
        if not run:
            return False, f"Run '{run_id}' not found."

        if run.status in ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"]:
            return False, f"Run '{run_id}' is already finalized ({run.status})."

        run.status = "CANCELLED"
        run.cancelled_by = actor_id
        run.cancelled_at = current_iso_time()
        run.cancellation_reason = reason
        self.store.save_run(run, workspace_id)

        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=actor_id,
            action="CANCEL",
            target_type="EXPERIMENT_RUN",
            target_id=run_id,
            metadata={"reason": reason},
        )
        return True, ""
