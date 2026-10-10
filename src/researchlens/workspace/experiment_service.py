"""Experiment ingestion, metric validation, baseline comparison, reproducibility checks, and findings engine."""

import csv
import io
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExperimentResult,
    ExperimentRun,
    ReproducibilityReport,
    ResearchFinding,
    generate_id,
)

logger = get_logger("experiment_service")

# Metric validation rules: (min_val, max_val, can_be_percentage)
METRIC_CONSTRAINTS = {
    "accuracy": (0.0, 1.0, True),
    "precision": (0.0, 1.0, True),
    "recall": (0.0, 1.0, True),
    "f1": (0.0, 1.0, True),
    "f1_score": (0.0, 1.0, True),
    "f1-score": (0.0, 1.0, True),
    "roc_auc": (0.0, 1.0, True),
    "roc-auc": (0.0, 1.0, True),
    "auc": (0.0, 1.0, True),
    "rmse": (0.0, math.inf, False),
    "mae": (0.0, math.inf, False),
    "bleu": (0.0, math.inf, False),
    "rouge": (0.0, math.inf, False),
    "ndcg": (0.0, 1.0, True),
    "mrr": (0.0, 1.0, True),
    "map": (0.0, 1.0, True),
    "loss": (0.0, math.inf, False),
    "latency": (0.0, math.inf, False),
}


def normalize_and_validate_metric(metric_name: str, raw_value: Any) -> Tuple[bool, float, str]:
    """Validate and normalize metric value.

    Returns:
        (is_valid, normalized_float_value, error_message)
    """
    try:
        val = float(raw_value)
    except (ValueError, TypeError):
        return False, 0.0, f"Non-numeric metric value: '{raw_value}'"

    clean_name = metric_name.strip().lower()

    if clean_name in METRIC_CONSTRAINTS:
        min_v, max_v, can_pct = METRIC_CONSTRAINTS[clean_name]
        # Handle percentage (e.g. 91.5% -> 0.915)
        if can_pct and val > 1.0 and val <= 100.0:
            val = val / 100.0

        if val < min_v or val > max_v:
            return False, val, f"Metric '{metric_name}' value {val} out of valid bounds [{min_v}, {max_v}]"

    return True, round(val, 6), ""


class ExperimentService:
    """Core engine for ingesting and evaluating empirical research results."""

    @staticmethod
    def preview_results(
        content: str,
        file_format: str = "csv",
    ) -> Dict[str, Any]:
        """Preview and validate incoming experiment result file without persisting."""
        rows_data: List[Dict[str, Any]] = []
        parse_errors: List[str] = []

        if file_format.lower() == "csv":
            try:
                reader = csv.DictReader(io.StringIO(content))
                if not reader.fieldnames:
                    return {"error": "Empty or headerless CSV file."}
                for idx, row in enumerate(reader, 1):
                    row["_row_number"] = idx
                    rows_data.append(row)
            except Exception as e:
                return {"error": f"Malformed CSV file: {e}"}

        elif file_format.lower() == "json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for idx, row in enumerate(data, 1):
                        if isinstance(row, dict):
                            row["_row_number"] = idx
                            rows_data.append(row)
                elif isinstance(data, dict) and "results" in data:
                    for idx, row in enumerate(data["results"], 1):
                        if isinstance(row, dict):
                            row["_row_number"] = idx
                            rows_data.append(row)
                else:
                    return {"error": "JSON must be a list of records or contain a 'results' array."}
            except Exception as e:
                return {"error": f"Malformed JSON file: {e}"}
        else:
            return {"error": f"Unsupported format '{file_format}'. Must be 'csv' or 'json'."}

        if not rows_data:
            return {"error": "No records found in content."}

        # Analyze columns and validate rows
        models = set()
        datasets = set()
        metrics = set()
        valid_rows = 0
        invalid_rows = 0
        validation_issues: List[Dict[str, Any]] = []

        required_cols = {"model", "dataset"}

        for row in rows_data:
            row_idx = row.get("_row_number", 0)
            clean_keys = {str(k).strip().lower(): k for k in row.keys() if k}
            missing = [c for c in required_cols if c not in clean_keys]
            if missing:
                invalid_rows += 1
                validation_issues.append({"row": row_idx, "error": f"Missing required columns: {missing}"})
                continue

            # Identify model and dataset
            m_val = row.get(clean_keys.get("model", "model"), "")
            d_val = row.get(clean_keys.get("dataset", "dataset"), "")
            if not m_val or not d_val:
                invalid_rows += 1
                validation_issues.append({"row": row_idx, "error": "Empty model or dataset value."})
                continue

            models.add(str(m_val).strip())
            datasets.add(str(d_val).strip())

            # Check format: narrow (metric + value) vs wide (columns are metrics)
            if "metric" in clean_keys and "value" in clean_keys:
                metric_name = str(row[clean_keys["metric"]]).strip()
                raw_val = row[clean_keys["value"]]
                is_valid, norm_val, err_msg = normalize_and_validate_metric(metric_name, raw_val)
                if is_valid:
                    metrics.add(metric_name.lower())
                    valid_rows += 1
                else:
                    invalid_rows += 1
                    validation_issues.append({"row": row_idx, "metric": metric_name, "error": err_msg})
            else:
                row_has_valid_metric = False
                row_has_error = False
                for k_orig, raw_val in row.items():
                    col_lower = str(k_orig).strip().lower()
                    if col_lower in ["model", "dataset", "run_id", "split", "seed", "unit", "_row_number"]:
                        continue
                    if raw_val is None or str(raw_val).strip() == "":
                        continue

                    is_valid, norm_val, err_msg = normalize_and_validate_metric(col_lower, raw_val)
                    if is_valid:
                        metrics.add(col_lower)
                        row_has_valid_metric = True
                    else:
                        row_has_error = True
                        validation_issues.append({"row": row_idx, "metric": col_lower, "error": err_msg})

                if row_has_valid_metric and not row_has_error:
                    valid_rows += 1
                else:
                    invalid_rows += 1

        return {
            "total_rows_detected": len(rows_data),
            "valid_rows_count": valid_rows,
            "invalid_rows_count": invalid_rows,
            "detected_models": sorted(list(models)),
            "detected_datasets": sorted(list(datasets)),
            "detected_metrics": sorted(list(metrics)),
            "validation_issues": validation_issues[:20],  # preview top 20
            "parse_errors": parse_errors,
        }

    @staticmethod
    def ingest_results(
        content: str,
        run_id: str,
        file_format: str = "csv",
        source_name: str = "imported_results.csv",
    ) -> Tuple[List[ExperimentResult], List[str]]:
        """Parse, validate, and convert raw file content into ExperimentResult records."""
        results: List[ExperimentResult] = []
        errors: List[str] = []

        rows_data: List[Dict[str, Any]] = []
        if file_format.lower() == "csv":
            try:
                reader = csv.DictReader(io.StringIO(content))
                rows_data = list(reader)
            except Exception as e:
                return [], [f"Failed to parse CSV: {e}"]
        elif file_format.lower() == "json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    rows_data = data
                elif isinstance(data, dict) and "results" in data:
                    rows_data = data["results"]
            except Exception as e:
                return [], [f"Failed to parse JSON: {e}"]

        for row_idx, row in enumerate(rows_data, 1):
            # Key normalization
            clean_row = {k.strip().lower(): v for k, v in row.items() if k}
            m_val = clean_row.get("model", "")
            d_val = clean_row.get("dataset", "")
            split_val = clean_row.get("split", "test")
            seed_val = clean_row.get("seed")
            seed_int = int(seed_val) if seed_val is not None and str(seed_val).isdigit() else None

            if not m_val or not d_val:
                errors.append(f"Row {row_idx}: Missing 'model' or 'dataset'")
                continue

            # Standard structured columns vs key-value metric columns
            if "metric" in clean_row and "value" in clean_row:
                metric_name = str(clean_row["metric"]).strip()
                val_raw = clean_row["value"]
                is_valid, norm_val, err_msg = normalize_and_validate_metric(metric_name, val_raw)
                if is_valid:
                    results.append(
                        ExperimentResult(
                            result_id=generate_id("res"),
                            run_id=run_id,
                            model=str(m_val).strip(),
                            dataset=str(d_val).strip(),
                            metric=metric_name,
                            value=norm_val,
                            split=split_val,
                            seed=seed_int,
                            source=source_name,
                        )
                    )
                else:
                    errors.append(f"Row {row_idx}: {err_msg}")
            else:
                # Wide format: each additional column is a metric
                row_candidates: List[ExperimentResult] = []
                row_has_error = False
                for col_name, raw_val in clean_row.items():
                    if col_name in ["model", "dataset", "split", "seed", "unit", "run_id"]:
                        continue
                    if raw_val is None or str(raw_val).strip() == "":
                        continue

                    is_valid, norm_val, err_msg = normalize_and_validate_metric(col_name, raw_val)
                    if is_valid:
                        row_candidates.append(
                            ExperimentResult(
                                result_id=generate_id("res"),
                                run_id=run_id,
                                model=str(m_val).strip(),
                                dataset=str(d_val).strip(),
                                metric=col_name,
                                value=norm_val,
                                split=split_val,
                                seed=seed_int,
                                source=source_name,
                            )
                        )
                    else:
                        row_has_error = True
                        errors.append(f"Row {row_idx} [{col_name}]: {err_msg}")

                if not row_has_error and row_candidates:
                    results.extend(row_candidates)

        return results, errors

    @staticmethod
    def compare_baseline(
        proposed_results: List[ExperimentResult],
        baseline_results: List[ExperimentResult],
    ) -> List[Dict[str, Any]]:
        """Compare proposed model metrics against baseline model metrics.

        Enforces Phase 6 Scientific Comparability:
        Only metrics sharing identical dataset and metric name are directly compared.
        """
        comparisons: List[Dict[str, Any]] = []

        # Index baseline by (dataset, metric)
        baseline_map: Dict[Tuple[str, str], List[ExperimentResult]] = {}
        for b in baseline_results:
            key = (b.dataset.lower(), b.metric.lower())
            baseline_map.setdefault(key, []).append(b)

        for p in proposed_results:
            key = (p.dataset.lower(), p.metric.lower())
            matching_baselines = baseline_map.get(key, [])

            if not matching_baselines:
                comparisons.append({
                    "dataset": p.dataset,
                    "metric": p.metric,
                    "proposed_model": p.model,
                    "proposed_value": p.value,
                    "baseline_model": "None (No matching baseline)",
                    "baseline_value": None,
                    "absolute_diff": None,
                    "relative_improvement_pct": None,
                    "comparability": "NO_BASELINE_MATCH",
                })
                continue

            for b in matching_baselines:
                diff = round(p.value - b.value, 6)
                pct = round((diff / abs(b.value)) * 100.0, 2) if abs(b.value) > 1e-9 else 0.0

                comparisons.append({
                    "dataset": p.dataset,
                    "metric": p.metric,
                    "proposed_model": p.model,
                    "proposed_value": p.value,
                    "baseline_model": b.model,
                    "baseline_value": b.value,
                    "absolute_diff": diff,
                    "relative_improvement_pct": pct,
                    "comparability": "DIRECTLY_COMPARABLE",
                    "provenance": {
                        "proposed_run": p.run_id,
                        "baseline_run": b.run_id,
                    },
                })

        return comparisons

    @staticmethod
    def assess_reproducibility(run: ExperimentRun) -> ReproducibilityReport:
        """Audit reproducibility metadata for an experiment run."""
        missing = []
        if not run.environment:
            missing.append("execution_environment")
        if not run.hardware:
            missing.append("hardware_specification")
        if not run.software_versions:
            missing.append("software_library_versions")
        if run.seed is None:
            missing.append("random_seed")

        if len(missing) == 0:
            status = "COMPLETE"
        elif len(missing) <= 2:
            status = "PARTIAL"
        else:
            status = "INSUFFICIENT"

        summary = {
            "environment": run.environment or "NOT PROVIDED",
            "hardware": run.hardware or "NOT PROVIDED",
            "seed": run.seed if run.seed is not None else "NOT PROVIDED",
            "software_versions_count": len(run.software_versions),
        }

        return ReproducibilityReport(
            report_id=generate_id("rep"),
            run_id=run.run_id,
            completeness_status=status,
            missing_fields=missing,
            metadata_summary=summary,
        )

    @staticmethod
    def generate_research_finding(
        workspace_id: str,
        title: str,
        statement: str,
        finding_type: str = "RESULT",
        related_hypothesis_id: Optional[str] = None,
        experiment_results: Optional[List[ExperimentResult]] = None,
        evidence_ids: Optional[List[str]] = None,
    ) -> ResearchFinding:
        """Formulate a researcher-reviewed finding preserving complete provenance."""
        result_ids = [r.result_id for r in (experiment_results or [])]
        ev_ids = evidence_ids or []

        # If no empirical or literature evidence exists, mark insufficient
        confidence = "HIGH" if (result_ids and ev_ids) else ("MEDIUM" if (result_ids or ev_ids) else "LOW")
        status = "RESEARCHER_REVIEW_REQUIRED" if (result_ids or ev_ids) else "INSUFFICIENT_EVIDENCE"

        return ResearchFinding(
            finding_id=generate_id("fnd"),
            workspace_id=workspace_id,
            title=title,
            statement=statement,
            finding_type=finding_type,
            evidence_ids=ev_ids,
            experiment_result_ids=result_ids,
            related_hypothesis_id=related_hypothesis_id,
            confidence_level=confidence,
            status=status,
        )

    @staticmethod
    def calculate_run_statistics(results: List[ExperimentResult]) -> Dict[str, Dict[str, Any]]:
        """Compute empirical summary statistics (mean, std, min, max) across multiple runs or seeds.

        Note: Strictly calculates statistics only when multiple observations exist.
        Does not fabricate statistical significance or p-values without repeated data.
        """
        by_metric: Dict[str, List[float]] = {}
        metric_datasets: Dict[str, str] = {}
        for r in results:
            k = r.metric.lower()
            by_metric.setdefault(k, []).append(r.value)
            if getattr(r, "dataset", None) and k not in metric_datasets:
                metric_datasets[k] = r.dataset

        stats: Dict[str, Dict[str, Any]] = {}
        for metric, vals in by_metric.items():
            n = len(vals)
            if n == 0:
                continue
            m = round(sum(vals) / n, 4)
            mn = round(min(vals), 4)
            mx = round(max(vals), 4)
            if n >= 2:
                variance = sum((x - m) ** 2 for x in vals) / (n - 1)
                std = round(math.sqrt(variance), 4)
            else:
                std = 0.0

            stats[metric] = {
                "count": n,
                "mean": m,
                "std": std,
                "min": mn,
                "max": mx,
                "has_repeated_runs": n >= 2,
                "dataset": metric_datasets.get(metric, "All Datasets"),
            }
        return stats
