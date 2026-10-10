"""Unit tests for MonitoringReport generation and Markdown/JSON/CSV exports."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.gap_evolution import ResearchGapEvolutionRecord
from researchlens.analysis.monitoring_report import generate_monitoring_report
from researchlens.analysis.research_topic import MonitoringRun, ResearchTopic
from researchlens.analysis.trend_analysis import EmergingEntity


def test_monitoring_report_generation_and_exports():
    topic = ResearchTopic(
        topic_id="top_test_rep",
        name="Clinical NLP Benchmark Tracking",
        query="clinical nlp electronic health records",
    )
    p_new = ExternalPaper(
        paper_id="p_clin_1",
        title="ClinicalBioBERT on MIMIC-IV",
        authors=["Alice Clinical", "Bob Health"],
        year=2025,
        provider="PubMed",
        doi="10.1000/clinbio",
        citation_count=8,
    )
    run = MonitoringRun(
        run_id="run_001",
        topic_id="top_test_rep",
        started_at="2026-10-01T00:00:00Z",
        completed_at="2026-10-01T00:01:00Z",
        papers=[p_new],
        new_paper_ids=["p_clin_1"],
    )
    topic.runs.append(run)

    gaps = {
        "g1": ResearchGapEvolutionRecord(
            gap_id="g1",
            gap_statement="Domain shift across ICU electronic health records",
            category="Clinical Domain",
            first_detected="2026-10-01T00:00:00Z",
            last_detected="2026-10-01T00:00:00Z",
            status="NEW",
            supporting_papers=["ClinicalBioBERT on MIMIC-IV"],
        )
    }

    methods = [
        EmergingEntity("method", "ClinicalBioBERT", "2026-10-01", ["ClinicalBioBERT on MIMIC-IV"], "Extracted model")
    ]

    report = generate_monitoring_report(
        topic,
        latest_run=run,
        gap_history=gaps,
        emerging_methods=methods,
    )

    # 1. Object properties
    assert report.topic_name == "Clinical NLP Benchmark Tracking"
    assert len(report.new_papers) == 1
    assert len(report.new_methods) == 1
    assert len(report.gap_evolution) == 1

    # 2. Markdown output
    md = report.to_markdown()
    assert "# Literature Monitoring Report: Clinical NLP Benchmark Tracking" in md
    assert "ClinicalBioBERT on MIMIC-IV" in md
    assert "Domain shift across ICU electronic health records" in md
    assert "Among the literature retrieved by ResearchLens" in md

    # 3. JSON export
    js = report.to_json()
    assert '"topic_name": "Clinical NLP Benchmark Tracking"' in js

    # 4. CSV export
    csv_str = report.to_csv()
    assert "paper_id,title,authors" in csv_str
    assert "ClinicalBioBERT on MIMIC-IV" in csv_str
