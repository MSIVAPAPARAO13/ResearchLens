"""Unit tests for experiment result preview and ingestion from CSV and JSON."""

import json

from researchlens.workspace.experiment_service import ExperimentService


def test_csv_result_preview_and_ingestion():
    csv_content = """model,dataset,accuracy,f1,latency
BERT,SQuAD,0.91,0.895,14.2
RoBERTa,SQuAD,0.93,0.915,18.5
DeBERTa,SQuAD,0.945,0.932,22.1
BadModel,SQuAD,1.5,invalid_f1,10.0
"""
    # 1. Preview
    preview = ExperimentService.preview_results(csv_content, file_format="csv")
    assert preview["total_rows_detected"] == 4
    assert preview["valid_rows_count"] == 3
    assert preview["invalid_rows_count"] == 1
    assert "accuracy" in preview["detected_metrics"]
    assert "f1" in preview["detected_metrics"]
    assert len(preview["validation_issues"]) >= 1

    # 2. Ingest
    results, errors = ExperimentService.ingest_results(
        content=csv_content,
        run_id="run_01",
        file_format="csv",
        source_name="squad_results.csv",
    )
    # Valid models have 3 metrics each = 9 results
    assert len(results) == 9
    assert len(errors) >= 1  # BadModel errors captured
    assert all(r.status == "UNVERIFIED_IMPORT" for r in results)
    assert all(r.source == "squad_results.csv" for r in results)


def test_json_result_preview_and_ingestion():
    json_data = [
        {"model": "GPT-4", "dataset": "GSM8K", "metric": "accuracy", "value": 0.92, "split": "test"},
        {"model": "Claude-3", "dataset": "GSM8K", "metric": "accuracy", "value": 0.95, "split": "test"},
    ]
    json_content = json.dumps(json_data)

    preview = ExperimentService.preview_results(json_content, file_format="json")
    assert preview["valid_rows_count"] == 2
    assert "accuracy" in preview["detected_metrics"]

    results, errors = ExperimentService.ingest_results(
        content=json_content,
        run_id="run_02",
        file_format="json",
        source_name="reasoning.json",
    )
    assert len(results) == 2
    assert len(errors) == 0
    assert results[0].value == 0.92
    assert results[1].value == 0.95
