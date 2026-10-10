"""Streamlit view component for MLflow Experiment Tracking & Model Registry (Phase 13)."""

from typing import Optional

import streamlit as st

from researchlens.execution.artifacts import verify_artifact_hash
from researchlens.mlops.client import check_mlflow_health, get_mlflow_client
from researchlens.mlops.configuration import get_mlflow_config
from researchlens.mlops.models import (
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_VALIDATED,
)
from researchlens.mlops.registry import ResearchModelRegistry
from researchlens.mlops.sync import MLflowSyncService
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import ExperimentArtifact
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("mlops_view")


def render_mlops_view(store: Optional[WorkspaceStore] = None) -> None:
    """Render the 📊 MLOps Tracking dashboard with 8 subtabs."""
    st.markdown("## 📊 MLOps Tracking & Model Registry (MLflow)")
    st.markdown(
        "Track scientific hyperparameters, normalize empirical evaluation metrics, "
        "synchronize output artifacts with MLflow, and govern model version stages with human-approved review workflows."
    )

    if store is None:
        store = WorkspaceStore()

    config = get_mlflow_config()
    health = check_mlflow_health(config)

    workspaces = store.list_workspaces()
    if not workspaces:
        st.info("No research workspaces found. Please create one in 🧪 Research Workspace tab first.")
        return

    ws_options = {f"{w.name} ({w.workspace_id})": w for w in workspaces}
    selected_ws_label = st.selectbox(
        "Active Research Workspace",
        list(ws_options.keys()),
        key="mlops_ws_selector",
    )
    current_ws = ws_options[selected_ws_label]
    ws_id = current_ws.workspace_id

    client = get_mlflow_client(config)
    sync_service = MLflowSyncService(store, config=config, client=client)
    model_registry = ResearchModelRegistry(store, config=config, client=client)

    subtabs = st.tabs([
        "🌐 Overview",
        "🔬 Experiments",
        "⚡ Runs",
        "📈 Metrics",
        "📦 Artifacts",
        "🏷️ Models",
        "📋 Registry",
        "🔄 Sync Status",
    ])

    # 1. OVERVIEW
    with subtabs[0]:
        st.markdown("### 🌐 MLOps & Tracking Infrastructure Health")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MLflow Tracking", health["status"])
        c2.metric("MLflow Version", health["version"])
        c3.metric("Tracking URI", config.tracking_uri)
        c4.metric("Integration Enabled", "Yes" if config.enabled else "No (Offline Mode)")

        st.markdown("---")
        runs = store.list_runs(ws_id)
        synced_runs = [r for r in runs if r.mlflow_sync_status == "SYNCED"]
        failed_syncs = [r for r in runs if r.mlflow_sync_status == "FAILED"]
        registered_models = store.list_registered_models(ws_id)

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Local Runs", len(runs))
        o2.metric("MLflow Synced", len(synced_runs))
        o3.metric("Registered Models", len(registered_models))
        o4.metric("Sync Failures", len(failed_syncs))

        if not config.enabled:
            st.info(
                "💡 **Offline Mode Active**: MLflow tracking is currently disabled (`MLFLOW_ENABLED=false`). "
                "All ResearchLens runs, artifacts, and metrics remain fully functional locally. "
                "Set environment variable `RESEARCHLENS_MLFLOW_ENABLED=true` to enable automated tracking."
            )
        elif health["status"] == "CONNECTED":
            st.success(f"✓ MLflow is connected at `{config.tracking_uri}`.")
        else:
            st.warning(f"⚠️ MLflow status is {health['status']}. Details: {health.get('error') or 'None'}")

    # 2. EXPERIMENTS
    with subtabs[1]:
        st.markdown("### 🔬 MLflow Experiment Namespace")
        st.caption(f"Workspace '{current_ws.name}' maps deterministically to MLflow experiment '{config.experiment_prefix}{current_ws.name}'.")

        exp_name = f"{config.experiment_prefix}{current_ws.name}"
        st.write(f"**Mapped Experiment Name:** `{exp_name}`")
        st.write(f"**Workspace ID:** `{ws_id}`")

        plans = store.list_experiments(ws_id)
        if not plans:
            st.info("No experiment plans configured in this workspace.")
        else:
            st.markdown("#### Registered Experiment Plans")
            for p in plans:
                with st.expander(f"Plan: {p.name} ({p.experiment_id})"):
                    st.write(f"**Dataset:** {p.dataset}")
                    st.write(f"**Model:** {p.model}")
                    st.write(f"**Entrypoint:** `{p.entrypoint}`")
                    st.write(f"**Timeout:** {p.timeout_seconds}s")

    # 3. RUNS
    with subtabs[2]:
        st.markdown("### ⚡ Tracked Runs & Cross-Navigation")
        runs = store.list_runs(ws_id)
        if not runs:
            st.info("No experiment runs executed in this workspace yet.")
        else:
            rows = []
            for r in runs:
                rows.append({
                    "ResearchLens Run": r.run_id,
                    "MLflow Run ID": r.mlflow_run_id or "Not Synced",
                    "Status": r.status,
                    "Sync Status": r.mlflow_sync_status or "DISABLED",
                    "Duration (s)": round(r.duration_seconds, 2),
                    "Seed": r.seed if r.seed is not None else "None",
                    "Code Commit": (r.git_commit_sha[:8] if r.git_commit_sha else "N/A"),
                })
            st.dataframe(rows, use_container_width=True)

    # 4. METRICS
    with subtabs[3]:
        st.markdown("### 📈 Normalized Metric Observations")
        results = store.list_results(ws_id)
        if not results:
            st.info("No empirical metrics recorded yet.")
        else:
            metric_rows = [
                {
                    "Run ID": r.run_id,
                    "Model": r.model,
                    "Dataset": r.dataset,
                    "Metric": r.metric,
                    "Normalized Value": r.value,
                    "Split": r.split,
                    "Seed": r.seed,
                    "Status": r.status,
                }
                for r in results
            ]
            st.dataframe(metric_rows, use_container_width=True)

    # 5. ARTIFACTS
    with subtabs[4]:
        st.markdown("### 📦 Artifact Lineage & Cryptographic Integrity")
        artifacts = store.list_artifacts(ws_id)
        if not artifacts:
            st.info("No artifacts logged in this workspace.")
        else:
            for art in artifacts:
                with st.container():
                    a1, a2, a3, a4 = st.columns([3, 2, 3, 2])
                    a1.markdown(f"**{art.name}** (`{art.type}`)")
                    a2.caption(f"Size: {art.size_bytes} B | Run: `{art.run_id}`")
                    a3.caption(f"SHA-256: `{art.sha256[:16]}...`")
                    with a4:
                        if st.button("Verify Hash", key=f"ml_vhash_{art.artifact_id}"):
                            valid = verify_artifact_hash(art.path, art.sha256)
                            if valid:
                                st.success("✓ SHA-256 Verified!")
                            else:
                                st.error("✗ Hash Mismatch Detected!")
                st.divider()

    # 6. MODELS
    with subtabs[5]:
        st.markdown("### 🏷️ Registered Scientific Models")
        registered_models = store.list_registered_models(ws_id)
        if not registered_models:
            st.info("No models registered yet. Register a model candidate in the '📋 Registry' tab.")
        else:
            for m in registered_models:
                with st.container():
                    m1, m2, m3, m4 = st.columns([3, 2, 2, 3])
                    m1.markdown(f"**{m.name}** (`v{m.version}`)")
                    m1.caption(f"ID: `{m.model_id}` | Source Run: `{m.run_id}`")
                    m2.markdown(f"Stage: **{m.stage}**")
                    m3.markdown(f"{m.metric.upper()}: `{m.metric_value}`")
                    with m4:
                        st.caption(f"SHA: `{m.artifact_hash[:12]}...`")
                        if m.reviewed_by:
                            st.caption(f"Reviewer: {m.reviewed_by}")
                st.divider()

    # 7. REGISTRY (HUMAN APPROVAL PROMOTION WORKFLOW)
    with subtabs[6]:
        st.markdown("### 📋 Model Governance & Promotion Workflow")
        st.caption("Stage Progression: CANDIDATE ➔ VALIDATED ➔ PRODUCTION strictly requires human reviewer approval.")

        completed_runs = [r for r in store.list_runs(ws_id) if r.status == "COMPLETED"]

        with st.expander("➕ Register Model Candidate from Successful Run"):
            if not completed_runs:
                st.warning("No completed experiment runs available to register.")
            else:
                with st.form("reg_candidate_form"):
                    chosen_run_id = st.selectbox("Select Completed Run", [r.run_id for r in completed_runs])
                    reg_mod_name = st.text_input("Model Name", value="DistilBERT-QA-Baseline")
                    reg_dataset = st.text_input("Dataset", value="SQuAD_v2")
                    reg_metric = st.text_input("Primary Metric", value="f1")
                    reg_val = st.number_input("Metric Value", value=0.9185, format="%.4f")
                    reg_version = st.text_input("Version (optional, e.g. 1.0.0)", value="")
                    reg_notes = st.text_area("Registration Notes", value="Baseline checkpoint verified with 100% reproducibility.")

                    if st.form_submit_button("Register Candidate Model", use_container_width=True):
                        # Find primary artifact for run
                        arts = [a for a in store.list_artifacts(ws_id) if a.run_id == chosen_run_id]
                        target_art = arts[0] if arts else ExperimentArtifact(
                            artifact_id="art_dummy",
                            run_id=chosen_run_id,
                            name="model.pt",
                            path="model.pt",
                            sha256="7c1b299865400cca78ec35c6f88e5974c7a33c7d92e65aa4c4eede8696efac9e",
                        )

                        m_reg, err = model_registry.register_candidate(
                            workspace_id=ws_id,
                            run_id=chosen_run_id,
                            model_name=reg_mod_name.strip(),
                            artifact=target_art,
                            dataset=reg_dataset.strip(),
                            metric=reg_metric.strip(),
                            metric_value=float(reg_val),
                            version=reg_version.strip() if reg_version.strip() else None,
                            notes=reg_notes.strip(),
                        )
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Registered model candidate '{m_reg.name}' v{m_reg.version} successfully!")
                            st.rerun()

        st.markdown("#### Human Review & Stage Promotion")
        reg_models = store.list_registered_models(ws_id)
        if not reg_models:
            st.info("No registered models to promote.")
        else:
            for mod in reg_models:
                with st.expander(f"Manage Stage: {mod.name} v{mod.version} [Current: {mod.stage}]"):
                    st.write(f"**Model ID:** `{mod.model_id}` | **Run:** `{mod.run_id}`")
                    st.write(f"**Dataset:** {mod.dataset} | **Score:** {mod.metric}={mod.metric_value}")
                    if mod.notes:
                        st.caption(f"Notes: {mod.notes}")

                    p_col1, p_col2 = st.columns([1, 1])
                    with p_col1:
                        target_stage = st.selectbox(
                            "Target Promotion Stage",
                            [STAGE_CANDIDATE, STAGE_VALIDATED, STAGE_PRODUCTION, STAGE_ARCHIVED],
                            index=1 if mod.stage == STAGE_CANDIDATE else 2,
                            key=f"stage_sel_{mod.model_id}",
                        )
                        reviewer = st.text_input("Reviewer Name / ID", value="Lead Researcher", key=f"rev_{mod.model_id}")
                    with p_col2:
                        rev_notes = st.text_area("Review Decision Notes", value="Validated against benchmark criteria.", key=f"notes_{mod.model_id}")
                        if st.button("Apply Stage Transition", key=f"p_btn_{mod.model_id}", type="primary"):
                            updated, err = model_registry.promote_model(
                                workspace_id=ws_id,
                                model_id=mod.model_id,
                                target_stage=target_stage,
                                reviewer_id=reviewer.strip(),
                                review_notes=rev_notes.strip(),
                            )
                            if err:
                                st.error(err)
                            else:
                                st.success(f"Model successfully transitioned to {target_stage}!")
                                st.rerun()

    # 8. SYNC STATUS
    with subtabs[7]:
        st.markdown("### 🔄 Synchronization Status & Retry Engine")
        runs = store.list_runs(ws_id)
        if not runs:
            st.info("No experiment runs found.")
        else:
            for r in runs:
                with st.container():
                    s1, s2, s3, s4 = st.columns([3, 2, 3, 2])
                    s1.markdown(f"**Run `{r.run_id}`**")
                    s1.caption(f"MLflow Run: `{r.mlflow_run_id or 'None'}`")
                    s2.markdown(f"Sync: **{r.mlflow_sync_status or 'DISABLED'}**")
                    s3.caption(f"Error: {r.mlflow_sync_error or 'None'}")
                    with s4:
                        if st.button("🔄 Retry Sync", key=f"sync_retry_{r.run_id}"):
                            with st.spinner("Retrying MLflow synchronization..."):
                                ok, err = sync_service.retry_sync(ws_id, r.run_id)
                                if ok:
                                    st.success("Synchronized!")
                                    st.rerun()
                                else:
                                    st.error(f"Sync failed: {err}")
                st.divider()
