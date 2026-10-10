"""Streamlit view component for ResearchLens Research Workspaces, Collaboration & Experiment Intelligence (Phase 11)."""

from typing import Optional

import streamlit as st

from researchlens.utils.logging import get_logger
from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.planner import (
    ResearchPlanner,
    calculate_direction_priority_score,
    infer_risks_from_limitations,
)
from researchlens.workspace.schema import (
    DiscussionMessage,
    DiscussionThread,
    EvidenceBookmark,
    ExperimentRun,
    PaperCollection,
    ResearchAnnotation,
    ResearchDirection,
    ResearchHypothesis,
    ResearchMilestone,
    ResearchQuestion,
    ResearchReview,
    SavedPaper,
    WorkspaceMember,
    generate_id,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("workspace_view")


def render_workspace_view(store: Optional[WorkspaceStore] = None) -> None:
    """Render the Research Workspace, Collaboration & Experiment tab."""
    st.markdown("## 🧪 Collaborative Research Workspace & Experiment Lab")
    st.markdown(
        "Collaborate across research teams, organize shared literature collections, bookmark evidentiary quotes, "
        "formulate testable hypotheses, ingest external experiment results, compare against baselines, and maintain audit trails."
    )

    if store is None:
        store = WorkspaceStore()

    workspaces = store.list_workspaces()

    # Workspace Selector / Creator
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        if workspaces:
            ws_options = {f"{w.name} ({w.workspace_id})": w for w in workspaces}
            selected_ws_label = st.selectbox("Select Research Workspace", list(ws_options.keys()))
            current_ws = ws_options[selected_ws_label]
        else:
            current_ws = None
            st.info("No research workspaces found. Create one below to begin organizing your research project.")

    with col_btn:
        st.write("")
        st.write("")
        with st.popover("➕ New Workspace", use_container_width=True):
            with st.form("create_workspace_form"):
                new_ws_name = st.text_input("Workspace Name", placeholder="e.g., Multimodal Recommendation")
                new_ws_desc = st.text_area("Description", placeholder="Objectives and scope...")
                submitted = st.form_submit_button("Create Workspace", use_container_width=True)
                if submitted and new_ws_name.strip():
                    created = store.create_workspace(new_ws_name.strip(), new_ws_desc.strip())
                    store.log_activity(created.workspace_id, "user", "WORKSPACE_CREATED", "WORKSPACE", created.workspace_id, f"Workspace '{created.name}' created.")
                    st.success(f"Workspace '{created.name}' created!")
                    st.rerun()

    if not current_ws:
        return

    ws_id = current_ws.workspace_id

    # Workspace Sub-Tabs
    tabs = st.tabs([
        "📊 Overview",
        "👥 Team & Activity",
        "📚 Papers & Collections",
        "🔖 Evidence & Annotations",
        "💬 Discussions",
        "❓ Questions & Hypotheses",
        "🎯 Research Directions",
        "📋 Research Plan",
        "🔬 Experiment Lab",
        "💡 Findings & Reviews",
        "⚠️ Risks & Timeline",
        "💾 Export & Reports",
    ])

    # 1. OVERVIEW
    with tabs[0]:
        papers = store.list_papers(ws_id)
        collections = store.list_collections(ws_id)
        evidence = store.list_evidence(ws_id)
        annotations = store.list_annotations(ws_id)
        questions = store.list_questions(ws_id)
        hypotheses = store.list_hypotheses(ws_id)
        directions = store.list_directions(ws_id)
        experiments = store.list_experiments(ws_id)
        runs = store.list_runs(ws_id)
        results = store.list_results(ws_id)
        findings = store.list_findings(ws_id)
        members = store.list_members(ws_id)
        risks = store.list_risks(ws_id)

        st.markdown(f"### Project: **{current_ws.name}**")
        if current_ws.description:
            st.caption(current_ws.description)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Team Members", len(members) or 1, help="Collaborators in workspace")
        m_col2.metric("Saved Papers", len(papers), help="Literature saved to workspace")
        m_col3.metric("Collections", len(collections), help="Thematic paper folders")
        m_col4.metric("Evidence Bookmarks", len(evidence), help="Extracted quotes & citations")

        m_col5, m_col6, m_col7, m_col8 = st.columns(4)
        m_col5.metric("Research Questions", len(questions), help="Investigative questions")
        m_col6.metric("Hypotheses", len(hypotheses), help="Testable hypotheses")
        m_col7.metric("Ingested Results", len(results), help="Empirical evaluation metrics")
        m_col8.metric("Research Findings", len(findings), help="Evidence-grounded findings")

        _ = (annotations, directions, experiments, runs, risks)

        st.markdown("---")
        st.markdown("#### Workspace Lifecycle")
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if current_ws.status == "ACTIVE":
                if st.button("Archive Workspace", key="arch_btn"):
                    current_ws.status = "ARCHIVED"
                    store.save_workspace(current_ws)
                    store.log_audit(ws_id, "user", "ARCHIVE", "WORKSPACE", ws_id)
                    st.rerun()
            else:
                if st.button("Unarchive Workspace", key="unarch_btn"):
                    current_ws.status = "ACTIVE"
                    store.save_workspace(current_ws)
                    store.log_audit(ws_id, "user", "UNARCHIVE", "WORKSPACE", ws_id)
                    st.rerun()
        with act_c2:
            if st.button("🗑️ Delete Workspace", key="del_ws_btn", type="secondary"):
                store.delete_workspace(ws_id)
                st.warning("Workspace deleted.")
                st.rerun()

    # 2. TEAM & ACTIVITY
    with tabs[1]:
        tm_col1, tm_col2 = st.columns([1, 1])
        with tm_col1:
            st.markdown("#### 👥 Workspace Members & Roles")
            with st.expander("➕ Invite Team Member"):
                with st.form("invite_member_form"):
                    mem_name = st.text_input("Name", placeholder="Dr. Alice Smith")
                    mem_email = st.text_input("Email", placeholder="alice@lab.org")
                    mem_role = st.selectbox("Role", ["OWNER", "RESEARCHER", "REVIEWER", "VIEWER"], index=1)
                    if st.form_submit_button("Add Member", use_container_width=True) and mem_name.strip():
                        new_mem = WorkspaceMember(
                            member_id=generate_id("mem"),
                            workspace_id=ws_id,
                            display_name=mem_name.strip(),
                            email=mem_email.strip(),
                            role=mem_role,
                        )
                        store.save_member(new_mem)
                        store.log_activity(ws_id, "user", "MEMBER_ADDED", "MEMBER", new_mem.member_id, f"Added {new_mem.display_name} as {new_mem.role}.")
                        store.log_audit(ws_id, "user", "CREATE", "MEMBER", new_mem.member_id)
                        st.success(f"Added {mem_name} to workspace!")
                        st.rerun()

            members = store.list_members(ws_id)
            if not members:
                st.info("No explicit members configured yet. Current user acts as Owner.")
            else:
                for m in members:
                    with st.container(border=True):
                        st.markdown(f"**{m.display_name}** (`{m.role}`)")
                        st.caption(f"Email: {m.email} | Status: `{m.status}` | Joined: {m.joined_at[:10]}")
                        if st.button("Remove Member", key=f"del_mem_{m.member_id}"):
                            store.delete_member(ws_id, m.member_id)
                            store.log_audit(ws_id, "user", "DELETE", "MEMBER", m.member_id)
                            st.rerun()

        with tm_col2:
            st.markdown("#### 📜 Recent Team Activity")
            activities = store.list_activity(ws_id)
            if not activities:
                st.info("No activity logged yet in this workspace.")
            else:
                for a in activities[:15]:
                    with st.container(border=True):
                        st.markdown(f"**{a.actor_id}** — {a.summary}")
                        st.caption(f"Action: `{a.action}` | Target: `{a.target_type}` | {a.timestamp[:19]}")

    # 3. PAPERS & COLLECTIONS
    with tabs[2]:
        p_sub1, p_sub2 = st.columns([2, 1])
        with p_sub2:
            with st.expander("➕ Save New Paper", expanded=False):
                with st.form("save_paper_form"):
                    p_title = st.text_input("Title")
                    p_authors = st.text_input("Authors (comma separated)")
                    p_year = st.number_input("Year", min_value=1950, max_value=2030, value=2026)
                    p_doi = st.text_input("DOI (optional)")
                    p_priority = st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH"], index=1)
                    p_notes = st.text_area("Initial Notes")
                    p_tags = st.text_input("Tags (comma separated)", placeholder="baseline, method, transformer")
                    if st.form_submit_button("Save Paper", use_container_width=True) and p_title.strip():
                        p_id = f"doi:{p_doi.strip()}" if p_doi.strip() else generate_id("paper")
                        new_paper = SavedPaper(
                            paper_id=p_id,
                            workspace_id=ws_id,
                            title=p_title.strip(),
                            authors=[a.strip() for a in p_authors.split(",") if a.strip()],
                            year=int(p_year),
                            doi=p_doi.strip() or None,
                            priority=p_priority,
                            notes=p_notes.strip(),
                            tags=[t.strip() for t in p_tags.split(",") if t.strip()],
                        )
                        store.save_paper(new_paper)
                        store.log_activity(ws_id, "user", "PAPER_ADDED", "PAPER", p_id, f"Saved paper '{p_title}'.")
                        store.log_audit(ws_id, "user", "CREATE", "PAPER", p_id)
                        st.success(f"Saved paper '{p_title}'!")
                        st.rerun()

            with st.expander("📁 Manage Collections", expanded=False):
                with st.form("create_col_form"):
                    col_name = st.text_input("New Collection Name")
                    col_desc = st.text_area("Collection Description")
                    if st.form_submit_button("Create Collection", use_container_width=True) and col_name.strip():
                        new_col = PaperCollection(
                            collection_id=generate_id("col"),
                            workspace_id=ws_id,
                            name=col_name.strip(),
                            description=col_desc.strip(),
                        )
                        store.save_collection(new_col)
                        st.success(f"Created collection '{col_name}'!")
                        st.rerun()

        with p_sub1:
            st.markdown("#### Saved Papers")
            saved_papers = store.list_papers(ws_id)
            if not saved_papers:
                st.info("No papers saved in this workspace yet.")
            else:
                for sp in saved_papers:
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**{sp.title}** ({sp.year or 'N/A'})")
                            if sp.authors:
                                st.caption("Authors: " + ", ".join(sp.authors[:3]) + (" et al." if len(sp.authors) > 3 else ""))
                            if sp.tags:
                                st.markdown(" ".join([f"`{t}`" for t in sp.tags]))
                            if sp.notes:
                                st.markdown(f"> *{sp.notes}*")
                        with c2:
                            st.caption(f"Priority: `{sp.priority}` | Status: `{sp.status}`")
                            if st.button("Remove", key=f"del_p_{sp.paper_id}"):
                                store.delete_paper(ws_id, sp.paper_id)
                                st.rerun()

            st.markdown("#### Collections")
            cols = store.list_collections(ws_id)
            if cols:
                for c in cols:
                    with st.expander(f"📁 {c.name} ({len(c.paper_ids)} papers)"):
                        st.caption(c.description or "No description")
                        if c.paper_ids:
                            st.write("Included Paper IDs:", c.paper_ids)
                        if st.button("Delete Collection", key=f"del_col_{c.collection_id}"):
                            store.delete_collection(ws_id, c.collection_id)
                            st.rerun()

    # 4. EVIDENCE & ANNOTATIONS
    with tabs[3]:
        ev_c1, ev_c2 = st.columns([1, 1])
        with ev_c1:
            st.markdown("#### 🔖 Evidence Bookmarks")
            with st.expander("➕ Add Evidence Bookmark"):
                with st.form("add_evidence_form"):
                    b_paper = st.text_input("Paper Title / ID")
                    b_sec = st.text_input("Section Name (e.g. Experiments)")
                    b_page = st.number_input("Page Number", min_value=1, max_value=200, value=1)
                    b_quote = st.text_area("Evidence Quote (exact excerpt)")
                    b_note = st.text_input("Researcher Note")
                    if st.form_submit_button("Save Bookmark", use_container_width=True) and b_quote.strip():
                        b_obj = EvidenceBookmark(
                            bookmark_id=generate_id("bm"),
                            workspace_id=ws_id,
                            paper_id=b_paper.strip(),
                            section_name=b_sec.strip() or None,
                            page_number=int(b_page),
                            source_text=b_quote.strip(),
                            note=b_note.strip(),
                        )
                        store.save_evidence(b_obj)
                        store.log_activity(ws_id, "user", "EVIDENCE_BOOKMARKED", "EVIDENCE", b_obj.bookmark_id, f"Bookmarked quote from {b_paper}.")
                        st.success("Evidence bookmark saved!")
                        st.rerun()

            bms = store.list_evidence(ws_id)
            if not bms:
                st.info("No evidence bookmarks saved yet.")
            else:
                for bm in bms:
                    with st.container(border=True):
                        st.markdown(f"> *\"{bm.source_text}\"*")
                        st.caption(f"Source: **{bm.paper_id}** | Page {bm.page_number or 'N/A'} | Section: {bm.section_name or 'N/A'}")
                        if bm.note:
                            st.markdown(f"📝 Note: {bm.note}")
                        if st.button("Delete", key=f"del_bm_{bm.bookmark_id}"):
                            store.delete_evidence(ws_id, bm.bookmark_id)
                            st.rerun()

        with ev_c2:
            st.markdown("#### 📝 Research Annotations")
            with st.expander("➕ Add Annotation"):
                with st.form("add_ann_form"):
                    a_paper = st.text_input("Paper ID")
                    a_type = st.selectbox("Type", ["IMPORTANT", "QUESTION", "LIMITATION", "METHOD", "RESULT", "IDEA", "FOLLOW_UP"])
                    a_content = st.text_area("Annotation Content")
                    a_quote = st.text_area("Anchor Passage (optional)")
                    if st.form_submit_button("Save Annotation", use_container_width=True) and a_content.strip():
                        a_obj = ResearchAnnotation(
                            annotation_id=generate_id("ann"),
                            workspace_id=ws_id,
                            paper_id=a_paper.strip(),
                            annotation_type=a_type,
                            content=a_content.strip(),
                            source_text=a_quote.strip(),
                        )
                        store.save_annotation(a_obj)
                        st.success("Annotation saved!")
                        st.rerun()

            anns = store.list_annotations(ws_id)
            if not anns:
                st.info("No annotations created yet.")
            else:
                for ann in anns:
                    with st.container(border=True):
                        st.markdown(f"`{ann.annotation_type}` — {ann.content}")
                        if ann.source_text:
                            st.caption(f"Referencing: \"{ann.source_text[:100]}...\"")
                        if st.button("Delete", key=f"del_ann_{ann.annotation_id}"):
                            store.delete_annotation(ws_id, ann.annotation_id)
                            st.rerun()

    # 5. DISCUSSIONS
    with tabs[4]:
        d_c1, d_c2 = st.columns([1, 1])
        with d_c1:
            st.markdown("#### 💬 Discussion Threads")
            with st.expander("➕ Start New Discussion Thread"):
                with st.form("create_thread_form"):
                    th_title = st.text_input("Thread Title", placeholder="Discrepancy in SQuAD baseline reproduction")
                    th_type = st.selectbox("Target Type", ["PAPER", "EVIDENCE", "QUESTION", "HYPOTHESIS", "EXPERIMENT", "RESULT"])
                    th_id = st.text_input("Target ID", placeholder="exp_01 or paper_doi")
                    th_msg = st.text_area("Initial Message")
                    if st.form_submit_button("Open Thread", use_container_width=True) and th_title.strip():
                        thread_obj = DiscussionThread(
                            thread_id=generate_id("thd"),
                            workspace_id=ws_id,
                            target_type=th_type,
                            target_id=th_id.strip(),
                            title=th_title.strip(),
                        )
                        if th_msg.strip():
                            thread_obj.messages.append(DiscussionMessage(generate_id("msg"), thread_obj.thread_id, "user", th_msg.strip()))
                        store.save_thread(thread_obj)
                        store.log_activity(ws_id, "user", "THREAD_CREATED", "THREAD", thread_obj.thread_id, f"Opened discussion '{thread_obj.title}'.")
                        st.success("Thread opened!")
                        st.rerun()

            threads = store.list_threads(ws_id)
            if not threads:
                st.info("No open discussion threads.")
            else:
                for t in threads:
                    with st.container(border=True):
                        st.markdown(f"**{t.title}** (`{t.status}`)")
                        st.caption(f"Target: `{t.target_type}` ({t.target_id}) | Messages: {len(t.messages)}")
                        for m in t.messages:
                            st.markdown(f"> **{m.author_id}**: {m.content}")
                        with st.form(f"reply_form_{t.thread_id}"):
                            reply_text = st.text_input("Reply to thread", key=f"reply_input_{t.thread_id}")
                            if st.form_submit_button("Send Reply") and reply_text.strip():
                                new_msg = DiscussionMessage(generate_id("msg"), t.thread_id, "user", reply_text.strip())
                                store.add_message_to_thread(ws_id, t.thread_id, new_msg)
                                st.rerun()

        with d_c2:
            st.markdown("#### 📝 Entity Comments")
            comments = store.list_comments(ws_id)
            if not comments:
                st.info("No inline comments logged yet.")
            else:
                for c in comments:
                    with st.container(border=True):
                        st.markdown(f"**{c.author_id}** on `{c.target_type}` (`{c.target_id}`):")
                        st.markdown(f"> *{c.content}*")
                        st.caption(f"Status: `{c.status}` | {c.created_at[:19]}")

    # 6. QUESTIONS & HYPOTHESES
    with tabs[5]:
        qh_c1, qh_c2 = st.columns([1, 1])
        with qh_c1:
            st.markdown("#### ❓ Research Questions")
            with st.expander("➕ Create Research Question"):
                with st.form("create_rq_form"):
                    rq_text = st.text_input("Question Text", placeholder="Does multimodal pretraining improve generalization?")
                    rq_type = st.selectbox("Type", ["PRIMARY", "SECONDARY", "EXPLORATORY"])
                    rq_prio = st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH"], index=1)
                    if st.form_submit_button("Add Question", use_container_width=True) and rq_text.strip():
                        rq_obj = ResearchQuestion(
                            question_id=generate_id("rq"),
                            workspace_id=ws_id,
                            text=rq_text.strip(),
                            type=rq_type,
                            priority=rq_prio,
                        )
                        store.save_question(rq_obj)
                        store.log_activity(ws_id, "user", "QUESTION_CREATED", "QUESTION", rq_obj.question_id, f"Formulated question: '{rq_obj.text}'.")
                        st.success("Research question created!")
                        st.rerun()

            r_questions = store.list_questions(ws_id)
            if not r_questions:
                st.info("No research questions formulated yet.")
            else:
                for q in r_questions:
                    with st.container(border=True):
                        st.markdown(f"**{q.text}**")
                        st.caption(f"Type: `{q.type}` | Priority: `{q.priority}` | Status: `{q.status}`")
                        if st.button("Delete Question", key=f"del_rq_{q.question_id}"):
                            store.delete_question(ws_id, q.question_id)
                            st.rerun()

        with qh_c2:
            st.markdown("#### 💡 Scientific Hypotheses")
            with st.expander("➕ Formulate Hypothesis"):
                with st.form("create_hyp_form"):
                    h_stmt = st.text_area("Hypothesis Statement", placeholder="Increasing multimodal dataset diversity will improve out-of-domain F1 score.")
                    h_ind = st.text_input("Independent Variables (comma separated)", placeholder="modalities, training dataset size")
                    h_dep = st.text_input("Dependent Variables (comma separated)", placeholder="F1 score, transfer accuracy")
                    h_rel = st.text_input("Expected Relationship", placeholder="Positive correlation")
                    h_status = st.selectbox("Status", ["DRAFT", "PROPOSED", "TESTED", "SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"], index=1)
                    if st.form_submit_button("Save Hypothesis", use_container_width=True) and h_stmt.strip():
                        hyp_obj = ResearchHypothesis(
                            hypothesis_id=generate_id("hyp"),
                            workspace_id=ws_id,
                            statement=h_stmt.strip(),
                            independent_variables=[v.strip() for v in h_ind.split(",") if v.strip()],
                            dependent_variables=[v.strip() for v in h_dep.split(",") if v.strip()],
                            expected_relationship=h_rel.strip(),
                            status=h_status,
                        )
                        store.save_hypothesis(hyp_obj)
                        store.log_activity(ws_id, "user", "HYPOTHESIS_CREATED", "HYPOTHESIS", hyp_obj.hypothesis_id, f"Formulated hypothesis: '{hyp_obj.statement[:50]}...'.")
                        st.success("Hypothesis recorded!")
                        st.rerun()

            hyps = store.list_hypotheses(ws_id)
            if not hyps:
                st.info("No hypotheses formulated yet.")
            else:
                for h in hyps:
                    with st.container(border=True):
                        st.markdown(f"**{h.statement}**")
                        st.caption(f"Status: `{h.status}` | Independent: {h.independent_variables} | Dependent: {h.dependent_variables}")
                        if h.expected_relationship:
                            st.caption(f"Expected Relationship: {h.expected_relationship}")
                        if st.button("Delete Hypothesis", key=f"del_hyp_{h.hypothesis_id}"):
                            store.delete_hypothesis(ws_id, h.hypothesis_id)
                            st.rerun()

    # 7. RESEARCH DIRECTIONS
    with tabs[6]:
        st.markdown("#### 🎯 Research Directions & Prioritization")
        with st.expander("➕ Formulate Research Direction"):
            with st.form("create_rd_form"):
                rd_title = st.text_input("Direction Title")
                rd_desc = st.text_area("Description & Proposed Scope")
                s1, s2, s3 = st.columns(3)
                ev_str = s1.slider("Evidence Strength", 0.0, 1.0, 0.7)
                gap_str = s2.slider("Research Gap Strength", 0.0, 1.0, 0.8)
                feas = s3.slider("Feasibility", 0.0, 1.0, 0.6)
                s4, s5, s6 = st.columns(3)
                imp = s4.slider("Impact Potential", 0.0, 1.0, 0.85)
                data_av = s5.slider("Data Availability", 0.0, 1.0, 0.7)
                eval_f = s6.slider("Evaluation Feasibility", 0.0, 1.0, 0.75)

                if st.form_submit_button("Save Direction", use_container_width=True) and rd_title.strip():
                    p_score, breakdown = calculate_direction_priority_score(
                        ev_str, gap_str, feas, imp, data_av, eval_f
                    )
                    dir_obj = ResearchDirection(
                        direction_id=generate_id("rd"),
                        workspace_id=ws_id,
                        title=rd_title.strip(),
                        description=rd_desc.strip(),
                        priority_score=p_score,
                        score_breakdown=breakdown,
                    )
                    store.save_direction(dir_obj)
                    st.success(f"Direction saved with Priority Score: {p_score:.4f}")
                    st.rerun()

        dirs = store.list_directions(ws_id)
        if not dirs:
            st.info("No research directions formulated yet.")
        else:
            for d in dirs:
                with st.container(border=True):
                    dc1, dc2 = st.columns([3, 1])
                    with dc1:
                        st.markdown(f"**{d.title}**")
                        st.caption(d.description or "No description provided.")
                    with dc2:
                        st.metric("Priority Score", f"{d.priority_score:.2f}")
                        st.caption(f"Status: `{d.status}`")
                    if st.button("Delete Direction", key=f"del_rd_{d.direction_id}"):
                        store.delete_direction(ws_id, d.direction_id)
                        st.rerun()

    # 8. RESEARCH PLAN
    with tabs[7]:
        st.markdown("#### 📋 Research Plan Synthesis & Versioning")
        plan_col1, plan_col2 = st.columns([2, 1])
        with plan_col2:
            with st.container(border=True):
                st.markdown("##### Plan Generator")
                p_prob = st.text_area("Problem Statement", placeholder="State the overarching scientific challenge...")
                if st.button("Generate Draft Plan", use_container_width=True):
                    qs = store.list_questions(ws_id)
                    hyps = store.list_hypotheses(ws_id)
                    dirs = store.list_directions(ws_id)
                    bms = store.list_evidence(ws_id)
                    risks = store.list_risks(ws_id)

                    candidate_ds = [{"name": "Literature-Derived Dataset", "source_paper": "Saved literature", "provenance": "Observed in benchmark papers"}]
                    candidate_base = [{"name": "Baseline Model", "source_paper": "Saved literature", "provenance": "Reported in related work"}]

                    plan = ResearchPlanner.generate_draft_plan(
                        workspace_id=ws_id,
                        title=f"{current_ws.name} — Research Plan",
                        problem_statement=p_prob.strip() or f"Investigating scientific questions in {current_ws.name}.",
                        questions=qs,
                        hypotheses=hyps,
                        directions=dirs,
                        bookmarks=bms,
                        candidate_datasets=candidate_ds,
                        candidate_baselines=candidate_base,
                        inferred_risks=risks,
                    )
                    store.save_plan(plan)
                    store.log_activity(ws_id, "user", "PLAN_GENERATED", "RESEARCH_PLAN", plan.plan_id, f"Generated research plan '{plan.title}'.")
                    st.success("Draft Research Plan synthesized successfully!")
                    st.rerun()

        with plan_col1:
            plans = store.list_plans(ws_id)
            if not plans:
                st.info("No research plans generated yet.")
            else:
                for plan in plans:
                    val = ResearchPlanner.validate_plan(plan)
                    with st.container(border=True):
                        st.markdown(f"### {plan.title} (v{plan.version})")
                        st.caption(f"Created: {plan.created_at}")

                        if val["status"] == "COMPLETE":
                            st.success("✅ Plan Readiness: **Complete & Grounded**")
                        else:
                            st.warning(f"⚠️ Plan Readiness: **Incomplete** ({', '.join(val['missing_components'])})")

                        with st.expander("View Full Structured Plan", expanded=True):
                            st.markdown(ResearchPlanner.export_plan_to_markdown(plan))

                        if st.button("Delete Plan", key=f"del_plan_{plan.plan_id}"):
                            store.delete_plan(ws_id, plan.plan_id)
                            st.rerun()

    # 9. EXPERIMENT LAB
    with tabs[8]:
        st.markdown("#### 🔬 Experiment Ingestion & Baseline Comparison")
        exp_t1, exp_t2, exp_t3 = st.tabs(["📊 Ingest Results", "⚖️ Baseline Comparison", "⚙️ Experiment Runs"])

        with exp_t1:
            st.markdown("##### Import External Experiment Results (CSV / JSON)")
            st.caption("Upload or paste benchmark evaluation logs. Values are validated and normalized automatically.")

            with st.form("ingest_results_form"):
                raw_content = st.text_area(
                    "Result Data (CSV or JSON format)",
                    placeholder="model,dataset,accuracy,f1,latency\nBERT,SQuAD,0.91,0.895,14.2\nRoBERTa,SQuAD,0.93,0.915,18.5",
                    height=140,
                )
                fmt = st.selectbox("File Format", ["csv", "json"])
                run_tag = st.text_input("Associated Run ID", value="run_default_01")
                st.form_submit_button("Preview & Validate Data")

            if raw_content.strip():
                preview = ExperimentService.preview_results(raw_content.strip(), file_format=fmt)
                if "error" in preview:
                    st.error(preview["error"])
                else:
                    st.success(f"Preview: {preview['valid_rows_count']} valid rows detected, {preview['invalid_rows_count']} invalid rows.")
                    st.caption(f"Detected Models: {', '.join(preview['detected_models'])} | Metrics: {', '.join(preview['detected_metrics'])}")

                    if preview["validation_issues"]:
                        with st.expander("⚠️ Validation Warnings & Issues"):
                            for issue in preview["validation_issues"]:
                                st.write(issue)

                    if st.button("Confirm and Ingest Results"):
                        ingested, errs = ExperimentService.ingest_results(raw_content.strip(), run_tag.strip(), file_format=fmt)
                        store.save_results_batch(ingested, ws_id)
                        store.log_activity(ws_id, "user", "RESULT_IMPORTED", "RESULT", run_tag.strip(), f"Ingested {len(ingested)} empirical results.")
                        store.log_audit(ws_id, "user", "IMPORT", "RESULT", run_tag.strip(), {"count": len(ingested)})
                        st.success(f"Successfully ingested {len(ingested)} empirical results!")
                        st.rerun()

            st.markdown("##### Ingested Results")
            results = store.list_results(ws_id)
            if not results:
                st.info("No empirical results ingested yet.")
            else:
                st.dataframe(
                    [{"Model": r.model, "Dataset": r.dataset, "Metric": r.metric, "Value": r.value, "Split": r.split, "Seed": r.seed, "Status": r.status} for r in results[:50]],
                    use_container_width=True,
                )

        with exp_t2:
            st.markdown("##### Model vs Baseline Comparison")
            results = store.list_results(ws_id)
            if len(results) < 2:
                st.info("Ingest at least 2 models on the same dataset/metric to perform baseline comparison.")
            else:
                models_available = sorted(list({r.model for r in results}))
                c_m1, c_m2 = st.columns(2)
                prop_m = c_m1.selectbox("Proposed Model", models_available, index=0)
                base_m = c_m2.selectbox("Baseline Model", models_available, index=min(1, len(models_available) - 1))

                prop_res = [r for r in results if r.model == prop_m]
                base_res = [r for r in results if r.model == base_m]

                comps = ExperimentService.compare_baseline(prop_res, base_res)
                if comps:
                    st.dataframe(
                        [
                            {
                                "Dataset": c["dataset"],
                                "Metric": c["metric"],
                                "Proposed": c["proposed_value"],
                                "Baseline": c["baseline_value"],
                                "Absolute Diff": c["absolute_diff"],
                                "Relative Impr (%)": f"{c['relative_improvement_pct']:+.2f}%" if c["relative_improvement_pct"] is not None else "N/A",
                                "Comparability": c["comparability"],
                            }
                            for c in comps
                        ],
                        use_container_width=True,
                    )

        with exp_t3:
            st.markdown("##### Experiment Execution Runs & Reproducibility")
            with st.expander("➕ Register Execution Run"):
                with st.form("create_run_form"):
                    r_id = st.text_input("Run ID", placeholder="run_v1_seed42")
                    r_exp = st.text_input("Experiment Plan ID", placeholder="exp_01")
                    r_env = st.text_input("Execution Environment", placeholder="Ubuntu 22.04 LTS, Python 3.11")
                    r_hw = st.text_input("Hardware Specification", placeholder="NVIDIA A100 80GB")
                    r_seed = st.number_input("Random Seed", value=42)
                    if st.form_submit_button("Register Run", use_container_width=True) and r_id.strip():
                        run_obj = ExperimentRun(
                            run_id=r_id.strip(),
                            experiment_id=r_exp.strip(),
                            environment=r_env.strip(),
                            hardware=r_hw.strip(),
                            seed=int(r_seed),
                        )
                        store.save_run(run_obj, ws_id)
                        rep = ExperimentService.assess_reproducibility(run_obj)
                        store.save_reproducibility_report(rep, ws_id)
                        st.success(f"Run registered! Reproducibility: **{rep.completeness_status}**")
                        st.rerun()

            runs = store.list_runs(ws_id)
            if not runs:
                st.info("No execution runs registered yet.")
            else:
                for run in runs:
                    with st.container(border=True):
                        st.markdown(f"**{run.run_id}** (`{run.status}`)")
                        st.caption(f"Environment: {run.environment or 'N/A'} | Hardware: {run.hardware or 'N/A'} | Seed: {run.seed}")
                        rep = ExperimentService.assess_reproducibility(run)
                        st.markdown(f"Reproducibility Completeness: `{rep.completeness_status}`")

    # 10. FINDINGS & REVIEWS
    with tabs[9]:
        st.markdown("#### 💡 Empirical Findings & Peer Review Workflow")
        f_c1, f_c2 = st.columns([1, 1])

        with f_c1:
            st.markdown("##### Formalize Research Finding")
            with st.form("create_finding_form"):
                f_title = st.text_input("Finding Title", placeholder="RoBERTa Outperforms BERT on SQuAD")
                f_stmt = st.text_area("Finding Statement", placeholder="Observed +2.23% relative F1 gain on identical test split.")
                f_type = st.selectbox("Finding Type", ["RESULT", "COMPARISON", "OBSERVATION", "LIMITATION", "NEGATIVE_RESULT", "REPRODUCIBILITY_ISSUE"])
                f_hyp = st.text_input("Related Hypothesis ID (optional)")
                if st.form_submit_button("Create Finding", use_container_width=True) and f_title.strip():
                    finding_obj = ExperimentService.generate_research_finding(
                        workspace_id=ws_id,
                        title=f_title.strip(),
                        statement=f_stmt.strip(),
                        finding_type=f_type,
                        related_hypothesis_id=f_hyp.strip() or None,
                    )
                    store.save_finding(finding_obj)
                    store.log_activity(ws_id, "user", "FINDING_CREATED", "FINDING", finding_obj.finding_id, f"Formalized finding '{finding_obj.title}'.")
                    st.success("Finding recorded! Status: RESEARCHER_REVIEW_REQUIRED")
                    st.rerun()

            findings = store.list_findings(ws_id)
            if not findings:
                st.info("No research findings recorded yet.")
            else:
                for f in findings:
                    with st.container(border=True):
                        st.markdown(f"**{f.title}** (`{f.finding_type}`)")
                        st.markdown(f"> *{f.statement}*")
                        st.caption(f"Status: `{f.status}` | Confidence: `{f.confidence_level}`")

        with f_c2:
            st.markdown("##### Peer Review Workflow")
            with st.form("submit_review_form"):
                rev_type = st.selectbox("Target Type", ["FINDING", "HYPOTHESIS", "EXPERIMENT", "RESEARCH_PLAN"])
                rev_id = st.text_input("Target ID", placeholder="fnd_01 or hyp_01")
                rev_dec = st.selectbox("Decision", ["APPROVED", "REQUIRES_REVISION", "REJECTED"])
                rev_com = st.text_area("Review Comments")
                if st.form_submit_button("Submit Peer Review", use_container_width=True) and rev_id.strip():
                    rev_obj = ResearchReview(
                        review_id=generate_id("rev"),
                        target_type=rev_type,
                        target_id=rev_id.strip(),
                        reviewer_id="user",
                        decision=rev_dec,
                        comments=rev_com.strip(),
                    )
                    store.save_review(rev_obj, ws_id)
                    store.log_activity(ws_id, "user", "REVIEW_COMPLETED", rev_type, rev_id.strip(), f"Review decision '{rev_dec}' submitted.")
                    st.success("Review decision saved!")
                    st.rerun()

            reviews = store.list_reviews(ws_id)
            if not reviews:
                st.info("No formal reviews logged.")
            else:
                for r in reviews:
                    with st.container(border=True):
                        st.markdown(f"**{r.decision}** on `{r.target_type}` ({r.target_id}) by `{r.reviewer_id}`")
                        if r.comments:
                            st.caption(f"Comments: *{r.comments}*")

    # 11. RISKS & TIMELINE
    with tabs[10]:
        rk_col1, rk_col2 = st.columns([1, 1])
        with rk_col1:
            st.markdown("#### ⚠️ Research Risk Register")
            if st.button("🔍 Infer Risks from Evidence", use_container_width=True):
                bms = store.list_evidence(ws_id)
                lims = [{"text": b.source_text, "paper_title": b.paper_id} for b in bms if "limit" in b.label.lower() or "limit" in b.note.lower()]
                if not lims:
                    lims = [{"text": "limited single dataset evaluation", "paper_title": "Literature"}]
                inferred = infer_risks_from_limitations(ws_id, lims)
                for r in inferred:
                    store.save_risk(r)
                st.success(f"Inferred {len(inferred)} evidence-grounded risks!")
                st.rerun()

            risks = store.list_risks(ws_id)
            if not risks:
                st.info("No research risks recorded yet.")
            else:
                for r in risks:
                    with st.container(border=True):
                        st.markdown(f"**{r.category} Risk** (`{r.severity}`) — {r.description}")
                        if r.mitigation:
                            st.caption(f"Mitigation: {r.mitigation}")
                        if st.button("Delete Risk", key=f"del_rk_{r.risk_id}"):
                            store.delete_risk(ws_id, r.risk_id)
                            st.rerun()

        with rk_col2:
            st.markdown("#### 🗓️ Project Milestones")
            with st.expander("➕ Add Milestone"):
                with st.form("create_ms_form"):
                    m_title = st.text_input("Milestone Title")
                    m_desc = st.text_area("Milestone Description")
                    m_date = st.text_input("Target Date (YYYY-MM-DD)")
                    m_stat = st.selectbox("Status", ["NOT_STARTED", "IN_PROGRESS", "COMPLETED", "BLOCKED"])
                    if st.form_submit_button("Add Milestone", use_container_width=True) and m_title.strip():
                        ms_obj = ResearchMilestone(
                            milestone_id=generate_id("ms"),
                            workspace_id=ws_id,
                            title=m_title.strip(),
                            description=m_desc.strip(),
                            target_date=m_date.strip() or None,
                            status=m_stat,
                        )
                        store.save_milestone(ms_obj)
                        st.success("Milestone added!")
                        st.rerun()

            milestones = store.list_milestones(ws_id)
            if not milestones:
                st.info("No milestones configured yet.")
            else:
                for ms in milestones:
                    with st.container(border=True):
                        st.markdown(f"**{ms.title}** (`{ms.status}`)")
                        if ms.target_date:
                            st.caption(f"Target: {ms.target_date}")
                        if ms.description:
                            st.caption(ms.description)
                        if st.button("Delete Milestone", key=f"del_ms_{ms.milestone_id}"):
                            store.delete_milestone(ws_id, ms.milestone_id)
                            st.rerun()

    # 12. EXPORT & REPORTS
    with tabs[11]:
        st.markdown("#### 💾 Export Research Workspace & Reports")
        plans = store.list_plans(ws_id)
        exps = store.list_experiments(ws_id)
        results = store.list_results(ws_id)
        findings = store.list_findings(ws_id)
        members = store.list_members(ws_id)

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            if plans:
                md_plan = ResearchPlanner.export_plan_to_markdown(plans[0])
                st.download_button(
                    "Download Research Plan (Markdown)",
                    data=md_plan,
                    file_name=f"{current_ws.name}_research_plan.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.button("Download Research Plan (Markdown)", disabled=True, use_container_width=True)

        with exp_c2:
            collab_rep = ResearchPlanner.generate_collaborative_research_report(
                current_ws.name,
                current_ws.description,
                plans,
                store.list_questions(ws_id),
                store.list_hypotheses(ws_id),
                results,
                findings,
                members,
            )
            st.download_button(
                "Download Collaborative Report (Markdown)",
                data=collab_rep,
                file_name=f"{current_ws.name}_collaborative_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        exp_c3, exp_c4 = st.columns(2)
        with exp_c3:
            if exps:
                csv_exp = ResearchPlanner.export_experiments_to_csv(exps)
                st.download_button(
                    "Download Experiments Protocol (CSV)",
                    data=csv_exp,
                    file_name=f"{current_ws.name}_experiments.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button("Download Experiments Protocol (CSV)", disabled=True, use_container_width=True)

        with exp_c4:
            bundle = store.export_workspace_bundle(ws_id)
            import json
            json_bundle = json.dumps(bundle, indent=2, ensure_ascii=False)
            st.download_button(
                "Download Full Workspace & Audit (JSON)",
                data=json_bundle,
                file_name=f"{current_ws.name}_workspace_bundle.json",
                mime="application/json",
                use_container_width=True,
            )
