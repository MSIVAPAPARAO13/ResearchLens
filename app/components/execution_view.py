"""Streamlit view component for Reproducible Experiment Execution & MLOps Integration (Phase 12)."""

import os
from typing import Optional

import streamlit as st

from researchlens.execution.artifacts import verify_artifact_hash
from researchlens.execution.environment import capture_environment_snapshot, get_gpu_info
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.provenance import calculate_reproducibility_completeness_score
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.utils.logging import get_logger
from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import (
    ExecutionProfile,
    ExperimentPlan,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("execution_view")


def render_execution_view(store: Optional[WorkspaceStore] = None) -> None:
    """Render the ⚙️ Experiment Execution dashboard with 8 subtabs."""
    st.markdown("## ⚙️ Safe & Reproducible Experiment Execution")
    st.markdown(
        "Execute registered scientific experiments in sandboxed local environments, "
        "capture environment provenance and execution logs, hash output artifacts, "
        "and audit computational reproducibility."
    )

    if store is None:
        store = WorkspaceStore()

    workspaces = store.list_workspaces()
    if not workspaces:
        st.info("Please create a Research Workspace in the 🧪 Research Workspace tab first.")
        return

    ws_options = {f"{w.name} ({w.workspace_id})": w for w in workspaces}
    selected_ws_label = st.selectbox(
        "Active Research Workspace",
        list(ws_options.keys()),
        key="exec_ws_selector",
    )
    current_ws = ws_options[selected_ws_label]
    ws_id = current_ws.workspace_id

    registry_service = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    subtabs = st.tabs([
        "📋 Registry",
        "🚀 Run Experiment",
        "⚡ Active Runs",
        "📜 Run History",
        "📊 Results",
        "📦 Artifacts",
        "🔍 Reproducibility",
        "📈 Comparison",
    ])

    # 1. REGISTRY
    with subtabs[0]:
        st.markdown("### 📋 Approved Experiment Registry")
        st.caption("Only experiments explicitly registered from approved directories are permitted to execute.")

        with st.expander("➕ Register New Experiment Entrypoint"):
            with st.form("register_experiment_form"):
                reg_name = st.text_input("Experiment Name", placeholder="Deterministic Baseline Demo")
                reg_entrypoint = st.text_input(
                    "Entrypoint Script Path",
                    value="scripts/experiments/deterministic_demo.py",
                    help="Must reside in scripts/experiments/ or approved directories. .py only.",
                )
                reg_desc = st.text_area("Description", placeholder="Synthetic benchmark experiment with deterministic metrics.")
                reg_version = st.text_input("Version", value="1.0.0")
                if st.form_submit_button("Register Entrypoint", use_container_width=True):
                    if reg_name.strip() and reg_entrypoint.strip():
                        entry, err = registry_service.register_experiment(
                            workspace_id=ws_id,
                            name=reg_name.strip(),
                            entrypoint=reg_entrypoint.strip(),
                            description=reg_desc.strip(),
                            version=reg_version.strip(),
                            owner="user",
                        )
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Registered experiment '{entry.name}' ({entry.experiment_id}) successfully.")
                            st.rerun()

        entries = registry_service.list_registered_experiments(ws_id)
        if not entries:
            st.info("No experiment entrypoints registered in this workspace yet.")
        else:
            for entry in entries:
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.markdown(f"**{entry.name}** (`v{entry.version}`)")
                    c1.caption(f"Entrypoint: `{entry.entrypoint}` | Owner: {entry.owner}")
                    c2.markdown(f"Status: `{'ALLOWED' if entry.allowed else 'BLOCKED'}`")
                    c3.markdown(f"Runs: {getattr(entry, 'run_count', 0)}")
                    with c4:
                        if entry.allowed:
                            if st.button("Block", key=f"blk_{entry.experiment_id}"):
                                registry_service.toggle_allowed(ws_id, entry.experiment_id, False)
                                st.rerun()
                        else:
                            if st.button("Allow", key=f"alw_{entry.experiment_id}"):
                                registry_service.toggle_allowed(ws_id, entry.experiment_id, True)
                                st.rerun()
                st.divider()

    # 2. RUN EXPERIMENT
    with subtabs[1]:
        st.markdown("### 🚀 Launch Controlled Experiment")
        st.caption("Execution is strictly sandboxed. Every run requires explicit researcher confirmation.")

        entries = [e for e in registry_service.list_registered_experiments(ws_id) if e.allowed]
        if not entries:
            st.warning("No allowed experiment entrypoints available. Please register and allow an entrypoint in Registry.")
        else:
            entry_map = {f"{e.name} ({e.entrypoint})": e for e in entries}
            chosen_entry_label = st.selectbox("Select Registered Experiment", list(entry_map.keys()), key="launch_entry_sel")
            chosen_entry = entry_map[chosen_entry_label]

            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                run_dataset = st.text_input("Dataset Name", value="SQuAD_v2", key="launch_dataset")
                run_model = st.text_input("Model Identifier", value="DistilBERT-QA", key="launch_model")
                run_seed = st.number_input("Random Seed (Optional)", min_value=0, max_value=999999, value=42, key="launch_seed")
            with col_cfg2:
                run_timeout = st.number_input("Timeout (Seconds)", min_value=1, max_value=86400, value=300, key="launch_timeout")
                device_choice = st.selectbox("Target Compute Device", ["cpu", "cuda"], index=0, key="launch_device")
                run_extra_args = st.text_input("Additional Parameters (Key=Value space-separated)", value="", key="launch_extra_args")

            # Environment Pre-flight Summary
            st.markdown("#### Pre-flight Environment Validation")
            gpu_info = get_gpu_info()
            p1, p2, p3 = st.columns(3)
            p1.metric("Python Runtime", capture_environment_snapshot().get("python_version", "3.13"))
            p2.metric("GPU Available", "Yes" if gpu_info.get("available") else "No (CPU Mode)")
            p3.metric("Timeout Limit", f"{run_timeout}s")

            st.markdown("---")
            if st.button("▶️ Start Experiment", type="primary", use_container_width=True, key="launch_btn"):
                with st.spinner("Preparing sandbox and executing experiment process..."):
                    # Create or reuse plan
                    plan = ExperimentPlan(
                        experiment_id=chosen_entry.experiment_id,
                        name=chosen_entry.name,
                        entrypoint=chosen_entry.entrypoint,
                        dataset=run_dataset,
                        model=run_model,
                        timeout_seconds=int(run_timeout),
                    )
                    store.save_experiment(plan, ws_id)

                    profile = ExecutionProfile(
                        profile_id="profile_default",
                        name="Local Runtime Profile",
                        hardware=device_choice,
                    )

                    extra_args_list = run_extra_args.split() if run_extra_args.strip() else None

                    run, err = executor.execute_experiment(
                        workspace_id=ws_id,
                        experiment_id=plan.experiment_id,
                        actor_id="researcher",
                        seed=int(run_seed),
                        execution_profile=profile,
                        extra_args=extra_args_list,
                    )

                    if err or not run or run.status != "COMPLETED":
                        st.error(f"Experiment execution finished with status: {run.status if run else 'FAILED'}. Error: {err or run.failure_category if run else 'Unknown'}")
                    else:
                        st.success(f"Experiment execution COMPLETED! Run ID: `{run.run_id}` (Duration: {run.duration_seconds:.2f}s)")
                        st.rerun()

    # 3. ACTIVE RUNS
    with subtabs[2]:
        st.markdown("### ⚡ Active / In-Flight Runs")
        all_runs = store.list_runs(ws_id)
        active_runs = [r for r in all_runs if r.status in ("QUEUED", "PREPARING", "RUNNING")]
        if not active_runs:
            st.info("No experiments currently running in background.")
        else:
            for ar in active_runs:
                c_a1, c_a2, c_a3, c_a4 = st.columns([3, 2, 2, 2])
                c_a1.markdown(f"**Run `{ar.run_id}`** ({ar.status})")
                c_a2.markdown(f"Started: {ar.started_at or 'Just now'}")
                c_a3.markdown(f"Duration: {ar.duration_seconds:.1f}s")
                with c_a4:
                    if st.button("Cancel Run", key=f"cnc_{ar.run_id}"):
                        executor.cancel_run(ws_id, ar.run_id, actor_id="researcher", reason="Manual cancellation by researcher")
                        st.warning("Cancelled run.")
                        st.rerun()

    # 4. RUN HISTORY
    with subtabs[3]:
        st.markdown("### 📜 Experiment Run History")
        runs = store.list_runs(ws_id)
        if not runs:
            st.info("No recorded experiment runs yet.")
        else:
            for run in runs:
                with st.expander(f"Run `{run.run_id}` — Status: {run.status} | Duration: {run.duration_seconds:.2f}s | Code: `{run.git_commit_sha[:8]}`"):
                    r_c1, r_c2 = st.columns(2)
                    with r_c1:
                        st.write(f"**Entrypoint:** `{run.entrypoint}`")
                        st.write(f"**Exit Code:** `{run.exit_code}`")
                        st.write(f"**Failure Category:** `{run.failure_category or 'NONE'}`")
                    with r_c2:
                        st.write(f"**Started:** {run.started_at}")
                        st.write(f"**Completed:** {run.completed_at}")
                        st.write(f"**Artifacts Count:** {len(run.artifact_ids)}")

                    # Logs viewer
                    run_dir = store.get_run_dir(ws_id, run.run_id)
                    stdout_path = os.path.join(run_dir, "logs", "stdout.log")
                    stderr_path = os.path.join(run_dir, "logs", "stderr.log")

                    if os.path.exists(stdout_path):
                        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                            stdout_txt = f.read()
                        st.text_area("stdout.log", stdout_txt, height=120, key=f"out_{run.run_id}")

                    if os.path.exists(stderr_path):
                        with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                            stderr_txt = f.read()
                        if stderr_txt.strip():
                            st.text_area("stderr.log", stderr_txt, height=80, key=f"err_{run.run_id}")

                    # Re-run button
                    if st.button("🔄 [Re-run Experiment]", key=f"rerun_{run.run_id}"):
                        with st.spinner("Executing new run instance..."):
                            new_run, err = executor.execute_experiment(
                                workspace_id=ws_id,
                                experiment_id=run.experiment_id,
                                actor_id="researcher",
                            )
                            if err or not new_run or new_run.status != "COMPLETED":
                                st.error(f"Re-run failed: {err}")
                            else:
                                st.success(f"Created new Run: `{new_run.run_id}`")
                                st.rerun()

    # 5. RESULTS
    with subtabs[4]:
        st.markdown("### 📊 Ingested Results & Findings")
        results = store.list_results(ws_id)
        if not results:
            st.info("No experiment results recorded yet.")
        else:
            st.dataframe(
                [
                    {
                        "Run ID": r.run_id,
                        "Model": r.model,
                        "Dataset": r.dataset,
                        "Metric": r.metric,
                        "Value": r.value,
                        "Split": r.split,
                        "Seed": r.seed,
                    }
                    for r in results
                ],
                use_container_width=True,
            )

            # Candidate Findings
            findings = [f for f in store.list_findings(ws_id) if f.status == "PENDING_RESEARCHER_REVIEW"]
            if findings:
                st.markdown("#### 💡 Candidate Findings Generated by Runs")
                for f in findings:
                    with st.container():
                        st.markdown(f"**Finding candidate:** {f.title}")
                        st.caption(f.statement)
                        cf1, cf2 = st.columns(2)
                        with cf1:
                            if st.button("Accept Finding (SUPPORTED)", key=f"acc_{f.finding_id}"):
                                f.status = "SUPPORTED"
                                store.save_finding(f, ws_id)
                                st.success("Finding marked as SUPPORTED.")
                                st.rerun()
                        with cf2:
                            if st.button("Reject Finding (NOT_SUPPORTED)", key=f"rej_{f.finding_id}"):
                                f.status = "NOT_SUPPORTED"
                                store.save_finding(f, ws_id)
                                st.warning("Finding rejected.")
                                st.rerun()

    # 6. ARTIFACTS
    with subtabs[5]:
        st.markdown("### 📦 Artifact Management & Integrity")
        artifacts = store.list_artifacts(ws_id)
        if not artifacts:
            st.info("No artifacts captured from experiment runs.")
        else:
            for art in artifacts:
                with st.container():
                    a1, a2, a3, a4 = st.columns([3, 2, 3, 2])
                    a1.markdown(f"**{art.name}** (`{art.type}`)")
                    a2.caption(f"Size: {art.size_bytes} bytes")
                    a3.caption(f"SHA-256: `{art.sha256[:16]}...`")
                    with a4:
                        if st.button("Verify Hash", key=f"vhash_{art.artifact_id}"):
                            is_valid = verify_artifact_hash(art.path, art.sha256)
                            if is_valid:
                                st.success("✓ SHA-256 Verified!")
                            else:
                                st.error("✗ Hash mismatch detected!")
                st.divider()

    # 7. REPRODUCIBILITY
    with subtabs[6]:
        st.markdown("### 🔍 Reproducibility Audit & Checklist")
        runs = store.list_runs(ws_id)
        completed_runs = [r for r in runs if r.status == "COMPLETED"]
        if not completed_runs:
            st.info("Execute an experiment run to generate a reproducibility manifest.")
        else:
            chosen_run_id = st.selectbox(
                "Select Run for Audit",
                [r.run_id for r in completed_runs],
                key="rep_run_sel",
            )
            run_obj = next(r for r in completed_runs if r.run_id == chosen_run_id)

            manifest_path = os.path.join(store.get_run_dir(ws_id, run_obj.run_id), "reports", "reproducibility.json")
            if os.path.exists(manifest_path):
                import json
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                score_info = calculate_reproducibility_completeness_score(manifest)

                m1, m2 = st.columns(2)
                m1.metric("Completeness Score", f"{score_info['score']}/{score_info['total_fields']}")
                m2.metric("Audit Status", score_info["status"])

                st.markdown("#### Reproducibility Checklist")
                for item in score_info["checklist"]:
                    mark = "✓" if item["present"] else "✗"
                    st.write(f"{mark} **{item['field']}**: `{item['value']}`")

                with st.expander("📄 View Canonical Manifest JSON"):
                    st.json(manifest)
            else:
                st.warning("Reproducibility manifest not found for this run.")

    # 8. COMPARISON
    with subtabs[7]:
        st.markdown("### 📈 Multi-Run & Seed Comparison")
        results = store.list_results(ws_id)
        if len(results) < 2:
            st.info("Execute at least two runs to compute statistical aggregations (mean, std, min, max).")
        else:
            stats = ExperimentService.calculate_run_statistics(results)
            if not stats:
                st.info("No matching metrics across multiple runs found.")
            else:
                rows = []
                for key, stat in stats.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        metric_name, ds_name = key
                    else:
                        metric_name = str(key)
                        ds_name = stat.get("dataset", "All Datasets")
                    rows.append({
                        "Metric": metric_name,
                        "Dataset": ds_name,
                        "Runs Count": stat.get("count", 0),
                        "Mean": round(stat.get("mean", 0.0), 4),
                        "Std Dev": round(stat.get("std", 0.0), 4),
                        "Min": round(stat.get("min", 0.0), 4),
                        "Max": round(stat.get("max", 0.0), 4),
                    })
                st.dataframe(rows, use_container_width=True)
                st.caption("Note: Statistics are computed purely over empirical runs. No statistical significance is claimed.")
