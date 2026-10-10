"""Research Automation Streamlit View (Phase 15).

Provides interactive interface for guided, evidence-grounded research automation:
- Topic & literature collection
- Evidence-backed research gaps
- Research question & hypothesis editor
- Experiment planning & run-count preview
- Approval gate before execution
- Execution progress & logs
- Evaluation & comparison results
- Finding candidates & human review actions
- Complete provenance & audit history
"""

import streamlit as st

from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.workspace.store import WorkspaceStore


def render_research_automation_view(store: WorkspaceStore) -> None:
    """Render the Research Automation tab interface."""
    st.markdown("## 🤖 Research Automation: Guided Scientific Workflow")
    st.markdown(
        "> **Scientific Integrity Standard:** ResearchLens enforces explicit stage boundaries: "
        "`EXECUTED` $\\neq$ `REPRODUCED` $\\neq$ `EVALUATED` $\\neq$ `COMPARABLE` $\\neq$ "
        "`RESEARCH FINDING` $\\neq$ `SCIENTIFIC CONCLUSION`. "
        "All generated proposals are literature-grounded candidates; execution requires explicit "
        "human approval and findings require formal researcher review."
    )

    workspaces = store.list_workspaces()
    if not workspaces:
        st.warning("No workspaces found. Please create a workspace in **🧪 Research Workspace** first.")
        return

    # Workspace selector
    ws_options = {ws.name: ws.workspace_id for ws in workspaces}
    selected_ws_name = st.selectbox("Select Active Workspace", list(ws_options.keys()), key="auto_ws_sel")
    ws_id = ws_options[selected_ws_name]

    orchestrator = ResearchWorkflowOrchestrator(store)
    workflows = store.list_workflows(ws_id)

    col_w1, col_w2 = st.columns([2, 1])
    with col_w2:
        st.markdown("#### ➕ Initialize Workflow")
        new_title = st.text_input("Workflow Title", value="Hybrid Retrieval Benchmark Study", key="new_wf_title")
        new_topic = st.text_input("Research Topic", value="Academic Literature QA", key="new_wf_topic")
        new_backend = st.selectbox("Backend", ["LOCAL", "DOCKER", "REMOTE"], key="new_wf_backend")
        actor_id = st.text_input("Actor ID", value="researcher", key="new_wf_actor")

        if st.button("🚀 Start Guided Workflow (Stages 1-6)", key="btn_start_wf"):
            with st.spinner("Executing Stages 1–6 (Literature -> Evidence -> Gaps -> Questions -> Hypotheses -> Plan)..."):
                wf, err = orchestrator.execute_guided_workflow_to_approval(
                    workspace_id=ws_id,
                    title=new_title,
                    topic=new_topic,
                    actor_id=actor_id,
                    execution_backend=new_backend,
                )
                if err:
                    st.error(f"Error during workflow initialization: {err}")
                else:
                    st.success(f"✅ Workflow '{wf.title}' initialized! Ready at approval gate.")
                    st.rerun()

    with col_w1:
        st.markdown("#### 📂 Active Workflows")
        if not workflows:
            st.info("No research workflows recorded yet in this workspace. Initialize one on the right.")
            return

        wf_map = {f"{w.title} ({w.workflow_id[:8]}) - [{w.status}]": w.workflow_id for w in workflows}
        selected_wf_label = st.selectbox("Select Workflow", list(wf_map.keys()), key="sel_wf_item")
        active_wf_id = wf_map[selected_wf_label]
        active_wf = store.get_workflow(ws_id, active_wf_id)

    if not active_wf:
        return

    st.markdown("---")

    # Workflow Status Bar & Stage Indicator
    st.subheader(f"Workflow: {active_wf.title}")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Current Stage", active_wf.current_stage)
    col_s2.metric("Workflow Status", active_wf.status)
    col_s3.metric("Approval Status", active_wf.approval_status)
    col_s4.metric("Approved By", active_wf.approved_by or "None")

    # Visual pipeline progression bar
    stages_display = [
        "1. Discovery", "2. Evidence", "3. Gaps", "4. Questions", "5. Hypothesis",
        "6. Plan", "7. Approval", "8. Matrix", "9. Execution", "10. Evaluation",
        "11. Comparability", "12. Finding", "13. Review",
    ]
    st.caption("Pipeline Sequence: " + " ➔ ".join(stages_display))

    tabs = st.tabs([
        "📚 1-3. Literature & Gaps",
        "❓ 4-5. Questions & Hypotheses",
        "📋 6. Experiment Plan",
        "🛡️ 7. Approval Gate",
        "⚙️ 8-9. Execution & Matrix",
        "📊 10-11. Evaluation & Comparability",
        "🏆 12-13. Findings & Review",
        "📜 Provenance & Audit",
    ])

    # Tab 1: Literature & Gaps
    with tabs[0]:
        st.markdown("### 📚 Literature Discovery & Validated Gaps")
        lit_data = active_wf.stage_data.get("literature", {})
        ev_data = active_wf.stage_data.get("evidence", {})
        gap_data = active_wf.stage_data.get("gaps", {})

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.write(f"**Discovered Papers ({lit_data.get('paper_count', 0)}):**")
            for p in lit_data.get("papers", []):
                st.write(f"- 📄 **{p.get('title', 'Unknown')}** ({p.get('year', 'N/A')})")

            st.write(f"**Evidence Excerpts ({ev_data.get('evidence_count', 0)}):**")
            for ev in ev_data.get("evidence_items", [])[:3]:
                st.caption(f"\"{ev.get('source_text', '')[:120]}...\" (Page: {ev.get('page')})")

        with col_l2:
            st.write(f"**Validated Research Gaps ({gap_data.get('gap_count', 0)}):**")
            for g in gap_data.get("validated_gaps", []):
                st.markdown(f"- 💡 `{g.get('status')}`: {g.get('description')}")

    # Tab 2: Questions & Hypotheses
    with tabs[1]:
        st.markdown("### ❓ Research Questions & Testable Hypotheses")
        q_data = active_wf.stage_data.get("questions", {})
        h_data = active_wf.stage_data.get("hypothesis", {})

        st.markdown("#### Primary Research Question")
        pq = q_data.get("primary_question", {})
        st.info(pq.get("text", "No question formulated."))

        st.markdown("#### Proposed Testable Hypothesis")
        prop = h_data.get("proposal", {})
        st.success(f"**Hypothesis:** {h_data.get('statement', 'No hypothesis formulated.')}")

        if prop:
            with st.expander("🔍 View Explicit Assumptions & Missing Evidence", expanded=True):
                st.markdown("**Explicit Assumptions:**")
                for asm in prop.get("explicit_assumptions", []):
                    st.write(f"- 📌 {asm}")

                st.markdown("**Missing Literature Evidence:**")
                for me in prop.get("missing_evidence", []):
                    st.write(f"- ⚠️ {me}")

                st.markdown("**Measurable Evaluation Criteria:**")
                for k, v in prop.get("measurable_criteria", {}).items():
                    st.write(f"- **{k}**: `{v}`")

                st.warning(prop.get("scientific_integrity_notice", ""))

    # Tab 3: Experiment Plan
    with tabs[2]:
        st.markdown("### 📋 Structured Experiment Plan")
        plan_data = active_wf.stage_data.get("experiment_plan", {})
        if plan_data:
            st.write(f"**Plan Name:** {plan_data.get('name')}")
            st.write(f"**Dataset:** `{plan_data.get('dataset')}`")
            st.write(f"**Model Architecture:** `{plan_data.get('model')}`")
            st.write(f"**Baseline Model:** `{plan_data.get('baseline')}`")
            st.write(f"**Evaluation Metrics:** `{', '.join(plan_data.get('metrics', []))}`")
            st.write(f"**Execution Backend:** `{plan_data.get('backend')}`")
            st.write(f"**Status:** `{plan_data.get('status')}`")
        else:
            st.info("No experiment plan recorded.")

    # Tab 4: Approval Gate
    with tabs[3]:
        st.markdown("### 🛡️ Mandatory Researcher Approval Gate")
        st.markdown(
            "Consequential experiment execution, matrix sweeps, and compute resource allocation "
            "strictly require explicit confirmation by an authorized researcher."
        )

        st.write(f"**Current Approval Status:** `{active_wf.approval_status}`")
        if active_wf.approval_status == "APPROVED":
            st.success(f"✅ Plan approved by `{active_wf.approved_by}` on `{active_wf.approved_at}`.")
            if active_wf.approval_notes:
                st.caption(f"Approval Notes: {active_wf.approval_notes}")
        elif active_wf.approval_status == "REJECTED":
            st.error(f"❌ Plan rejected. Notes: {active_wf.approval_notes}")
        else:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                approver = st.text_input("Reviewer / Researcher ID", value="m_owner", key="gate_approver_id")
                notes = st.text_area("Review Notes", value="Verified hypothesis formulation and benchmark limits.", key="gate_notes")
                if st.button("✅ Approve Plan for Execution", key="btn_gate_approve"):
                    _, err = orchestrator.approve_workflow_plan(
                        workflow_id=active_wf.workflow_id,
                        workspace_id=ws_id,
                        approver_id=approver,
                        approval_notes=notes,
                    )
                    if err:
                        st.error(f"Approval failed: {err}")
                    else:
                        st.success("Plan successfully approved!")
                        st.rerun()

            with col_a2:
                if st.button("❌ Reject / Require Revision", key="btn_gate_reject"):
                    _, err = orchestrator.reject_workflow_plan(
                        workflow_id=active_wf.workflow_id,
                        workspace_id=ws_id,
                        reviewer_id=approver,
                        notes=notes,
                    )
                    if err:
                        st.error(f"Rejection failed: {err}")
                    else:
                        st.warning("Plan marked as rejected.")
                        st.rerun()

    # Tab 5: Execution & Matrix
    with tabs[4]:
        st.markdown("### ⚙️ Bounded Matrix & Controlled Execution")
        if active_wf.approval_status != "APPROVED":
            st.warning("🔒 Execution pipeline is locked. Human researcher approval required in tab **🛡️ 7. Approval Gate**.")
        else:
            if st.button("🚀 Launch Execution Pipeline (Stages 8–12)", key="btn_launch_exec"):
                with st.spinner("Executing bounded matrix, container/local run, evaluation, and comparability..."):
                    wf, err = orchestrator.execute_post_approval_pipeline(
                        workflow_id=active_wf.workflow_id,
                        workspace_id=ws_id,
                    )
                    if err:
                        st.error(f"Pipeline error: {err}")
                    else:
                        st.success("✅ Execution pipeline completed through Stage 12!")
                        st.rerun()

            mat_data = active_wf.stage_data.get("matrix", {})
            exec_data = active_wf.stage_data.get("execution", {})

            if mat_data:
                st.markdown("#### Bounded Experiment Matrix")
                st.write(f"- Planned Runs: `{mat_data.get('planned_run_count')}` (Limit: $\\le 20$)")
                st.write(f"- Models: `{mat_data.get('models')}`")
                st.write(f"- Datasets: `{mat_data.get('datasets')}`")

            if exec_data:
                st.markdown("#### Execution Output")
                st.write(f"- Run ID: `{exec_data.get('run_id')}`")
                st.write(f"- Status: `{exec_data.get('status')}`")
                st.write(f"- Duration: `{exec_data.get('duration')}s`")
                if exec_data.get("mlflow_run_id"):
                    st.write(f"- MLflow Run ID: `{exec_data.get('mlflow_run_id')}`")

    # Tab 6: Evaluation & Comparability
    with tabs[5]:
        st.markdown("### 📊 Evaluation Results & Scientific Comparability")
        eval_data = active_wf.stage_data.get("evaluation", {})
        comp_data = active_wf.stage_data.get("comparability", {})

        if eval_data:
            st.markdown("#### Observed Empirical Metrics")
            st.json(eval_data.get("metrics", {}))

        if comp_data:
            st.markdown("#### Scientific Comparability Analysis")
            st.write(f"- Comparability Status: `{comp_data.get('comparability_status')}`")
            st.write(f"- Baseline Model: `{comp_data.get('baseline_model')}`")
            comps = comp_data.get("comparisons", [])
            if comps:
                st.dataframe(comps)
            st.info(comp_data.get("significance_notice", ""))

    # Tab 7: Findings & Review
    with tabs[6]:
        st.markdown("### 🏆 Finding Candidate & Formal Human Review")
        fc_data = active_wf.stage_data.get("finding_candidate", {})
        if not fc_data:
            st.info("No finding candidate generated yet. Run Stages 8–12.")
        else:
            st.markdown(f"#### {fc_data.get('title')}")
            st.write(f"**Statement:** {fc_data.get('statement')}")
            st.metric("Finding Status", fc_data.get("status"))

            st.write("**Observed Metrics:**", fc_data.get("observed_metrics", {}))
            if fc_data.get("baseline_comparison"):
                st.write("**Baseline Comparisons:**", fc_data.get("baseline_comparison"))

            st.markdown("---")
            st.markdown("#### 🔬 Formal Researcher Review Decision")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                reviewer_id = st.text_input("Reviewer ID", value="m_owner", key="fnd_rev_id")
                review_decision = st.selectbox(
                    "Review Decision",
                    ["CONFIRMED", "REJECTED", "REQUIRES_REVISION"],
                    key="fnd_rev_dec",
                )
                rev_notes = st.text_area("Review Critique / Rationale", value="Empirical results verified on PubMedQA.", key="fnd_rev_notes")
                if st.button("Submit Review Decision", key="btn_fnd_submit"):
                    _, err = orchestrator.review_workflow_finding(
                        workflow_id=active_wf.workflow_id,
                        workspace_id=ws_id,
                        reviewer_id=reviewer_id,
                        decision=review_decision,
                        review_notes=rev_notes,
                    )
                    if err:
                        st.error(f"Review failed: {err}")
                    else:
                        st.success(f"Review recorded as {review_decision}!")
                        st.rerun()

            with col_r2:
                rev_data = active_wf.stage_data.get("human_review", {})
                if rev_data:
                    st.success(f"Decision: **{rev_data.get('decision')}**")
                    st.write(f"Reviewed By: `{rev_data.get('reviewer_id')}`")
                    st.write(f"Notes: {rev_data.get('notes')}")

    # Tab 8: Provenance & Audit
    with tabs[7]:
        st.markdown("### 📜 Workflow Provenance & Audit Trail")
        st.markdown("#### Stage Transitions History")
        st.dataframe(active_wf.stage_history)

        st.markdown("#### Workspace Audit Events")
        audits = store.list_audit(ws_id)
        wf_audits = [
            {"action": a.action, "actor": a.actor_id, "timestamp": a.timestamp, "metadata": a.metadata}
            for a in audits if a.target_id == active_wf.workflow_id or a.target_type == "WORKFLOW"
        ]
        if wf_audits:
            st.dataframe(wf_audits)
        else:
            st.caption("No audit events recorded for this workflow.")
