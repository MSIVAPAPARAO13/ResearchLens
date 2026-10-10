"""Unit tests for ResearchGapReport aggregation and serialization."""

from researchlens.analysis.research_gap_report import generate_research_gap_report
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


def test_generate_research_gap_report_full():
    lim = ExtractedItem(
        value="A primary limitation is high memory consumption during training.",
        page_number=5,
        section_name="discussion",
        chunk_id="c_5",
    )
    fw = ExtractedItem(
        value="Future work will evaluate cross-domain adaptation on PubMed.",
        page_number=8,
        section_name="conclusion",
        chunk_id="c_8",
    )
    profile = ResearchProfile(
        document_id="p1",
        title="Attention Mechanism Paper",
        limitations=[lim],
        future_work=[fw],
    )

    report = generate_research_gap_report(profile, ablation_mode="full")
    assert report.paper_title == "Attention Mechanism Paper"
    assert len(report.top_gaps) >= 2
    assert len(report.potential_hypotheses) >= 1

    d = report.to_dict()
    assert d["paper_title"] == "Attention Mechanism Paper"
    assert "top_gaps" in d
    assert "potential_hypotheses" in d

    md = report.to_markdown()
    assert "# 💡 Research Gap Intelligence Report" in md
    assert "Executive Summary" in md
    assert "Top Identified Research Gaps" in md


def test_generate_research_gap_report_ablation_modes():
    lim = ExtractedItem(value="Memory is limited.", page_number=2, section_name="discussion")
    fw = ExtractedItem(value="Future work on multilingual text.", page_number=3, section_name="conclusion")
    profile = ResearchProfile(document_id="p2", title="Ablation Test", limitations=[lim], future_work=[fw])

    # Mode A: limitation_only
    rep_a = generate_research_gap_report(profile, ablation_mode="limitation_only")
    assert all(g.gap_type == "EXPLICIT_LIMITATION" for g in rep_a.top_gaps)

    # Mode B: limitation_future
    rep_b = generate_research_gap_report(profile, ablation_mode="limitation_future")
    g_types = [g.gap_type for g in rep_b.top_gaps]
    assert "EXPLICIT_LIMITATION" in g_types
    assert "EXPLICIT_FUTURE_WORK" in g_types
