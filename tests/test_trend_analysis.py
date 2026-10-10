from researchlens.analysis.trend_analysis import TrendAnalyzer
from researchlens.extraction.schema import ResearchProfile


def test_detect_emerging_entities():
    analyzer = TrendAnalyzer()

    # Historical profile: BERT on SQuAD evaluating F1
    hist_prof = ResearchProfile(
        title="BERT Paper",
        models=["BERT", "RoBERTa"],
        datasets=["SQuAD v2.0"],
        metrics=["F1", "EM"],
    )

    # Current profile introducing DeBERTa on BioASQ evaluating ROUGE-L
    curr_prof = ResearchProfile(
        title="DeBERTa Paper",
        models=["DeBERTa", "BERT"],
        datasets=["SQuAD v2.0", "BioASQ"],
        metrics=["F1", "ROUGE-L"],
    )

    emerging = analyzer.detect_emerging_entities(
        historical_profiles=[hist_prof],
        current_profile=curr_prof,
        paper_title="DeBERTa Paper",
    )

    # Emerging methods
    method_names = [m.name.lower() for m in emerging["methods"]]
    assert "deberta" in method_names
    assert "bert" not in method_names

    # Emerging datasets
    dataset_names = [d.name.lower() for d in emerging["datasets"]]
    assert "bioasq" in dataset_names
    assert "squad v2.0" not in dataset_names

    # Emerging metrics
    metric_names = [met.name.lower() for met in emerging["metrics"]]
    assert "rouge-l" in metric_names
    assert "f1" not in metric_names


def test_build_trend_report():
    analyzer = TrendAnalyzer()

    p1 = ResearchProfile(title="P1", models=["Transformer"], datasets=["ImageNet"], metrics=["Top-1"])
    p2 = ResearchProfile(title="P2", models=["Transformer", "CNN"], datasets=["ImageNet"], metrics=["Top-1"])

    report = analyzer.build_trend_report(
        topic_name="Vision Transformers",
        historical_runs_papers=[[], []],
        all_profiles=[p1, p2],
    )

    assert report.topic_name == "Vision Transformers"
    assert report.method_frequency["Transformer"] == 2
    assert report.method_frequency["CNN"] == 1
    assert report.dataset_frequency["ImageNet"] == 2
    assert "Among the literature retrieved by ResearchLens" in report.summary_narrative
