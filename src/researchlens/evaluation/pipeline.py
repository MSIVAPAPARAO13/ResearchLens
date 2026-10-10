"""Automated Evaluation Pipeline and NLP Experiment Pipeline Templates (Phase 14)."""

import json
import os
from typing import Any, Dict, List, Optional

from researchlens.utils.logging import get_logger
from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import (
    EvaluationContract,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("evaluation_pipeline")


class NLPEvaluationTemplates:
    """Pre-configured experiment pipeline templates for standardized NLP benchmarks."""

    @staticmethod
    def get_rag_template(
        dataset: str = "PubMedQA",
        model: str = "MiniLM-L6-v2 + CrossEncoder",
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Template for RAG Evaluation Pipeline (chunking, embedding, retrieval, reranker, answerability)."""
        default_params = {
            "chunk_size": 256,
            "chunk_overlap": 32,
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 5,
            "threshold": 0.40,
            "reranker": "ms-marco-MiniLM-L6-v2",
        }
        if parameters:
            default_params.update(parameters)

        return {
            "pipeline_type": "RAG_EVALUATION",
            "dataset": dataset,
            "model": model,
            "parameters": default_params,
            "seed": seed,
            "expected_metrics": [
                "precision",
                "recall",
                "MRR",
                "citation_accuracy",
                "answerability",
            ],
            "entrypoint": "scripts/experiments/deterministic_demo.py",
        }

    @staticmethod
    def get_summarization_template(
        dataset: str = "arXiv-Summarization",
        model: str = "BART-Large-CNN",
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Template for Summarization Evaluation Pipeline (ROUGE-1, ROUGE-2, ROUGE-L)."""
        default_params = {
            "model": model,
            "max_length": 250,
            "min_length": 50,
            "beam_count": 4,
        }
        if parameters:
            default_params.update(parameters)

        return {
            "pipeline_type": "SUMMARIZATION_EVALUATION",
            "dataset": dataset,
            "model": model,
            "parameters": default_params,
            "seed": seed,
            "expected_metrics": [
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
            ],
            "entrypoint": "scripts/experiments/deterministic_demo.py",
        }

    @staticmethod
    def get_classification_template(
        dataset: str = "SciCite",
        model: str = "SciBERT-Classifier",
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Template for Classification Pipeline (accuracy, precision, recall, F1, ROC-AUC)."""
        default_params = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "epochs": 3,
            "learning_rate": 0.0001,
            "batch_size": 16,
        }
        if parameters:
            default_params.update(parameters)

        return {
            "pipeline_type": "CLASSIFICATION_EVALUATION",
            "dataset": dataset,
            "model": model,
            "parameters": default_params,
            "seed": seed,
            "expected_metrics": [
                "accuracy",
                "precision",
                "recall",
                "F1",
                "ROC-AUC",
            ],
            "entrypoint": "scripts/experiments/deterministic_demo.py",
        }

    @staticmethod
    def get_information_extraction_template(
        dataset: str = "CoNLL-2003",
        model: str = "NER-Transformer",
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Template for Information Extraction Pipeline."""
        default_params = {
            "entity_types": ["METHOD", "DATASET", "METRIC"],
            "confidence_threshold": 0.70,
        }
        if parameters:
            default_params.update(parameters)

        return {
            "pipeline_type": "INFORMATION_EXTRACTION_EVALUATION",
            "dataset": dataset,
            "model": model,
            "parameters": default_params,
            "seed": seed,
            "expected_metrics": ["precision", "recall", "f1"],
            "entrypoint": "scripts/experiments/deterministic_demo.py",
        }

    @staticmethod
    def get_embedding_retrieval_template(
        dataset: str = "SciFact",
        model: str = "all-MiniLM-L6-v2",
        parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Template for Embedding Retrieval Pipeline."""
        default_params = {
            "distance_metric": "cosine",
            "top_k": 10,
        }
        if parameters:
            default_params.update(parameters)

        return {
            "pipeline_type": "EMBEDDING_RETRIEVAL_EVALUATION",
            "dataset": dataset,
            "model": model,
            "parameters": default_params,
            "seed": seed,
            "expected_metrics": ["HitRate@1", "HitRate@5", "MRR", "NDCG@5"],
            "entrypoint": "scripts/experiments/deterministic_demo.py",
        }


class AutomatedEvaluationPipeline:
    """Automated Evaluation Pipeline: execute -> validate -> evaluate -> compare -> track -> report.

    Guarantees:
    - Never automatically produce a scientific conclusion.
    - Exports evaluation.json contract.
    """

    def __init__(self, store: WorkspaceStore):
        self.store = store

    def run_pipeline(
        self,
        workspace_id: str,
        experiment_id: str,
        run_fn: Any,  # function that calls executor
        actor_id: str = "researcher",
    ) -> Dict[str, Any]:
        """Execute full six-stage evaluation pipeline."""
        warnings: List[str] = []

        # 1. Execute
        experiment = self.store.get_experiment(workspace_id, experiment_id)
        if not experiment:
            return {"error": f"Experiment '{experiment_id}' not found.", "status": "FAILED"}

        run, exec_err = run_fn(workspace_id, experiment_id)
        if not run or run.status != "COMPLETED":
            return {
                "status": "FAILED",
                "error": exec_err or "Execution failed or did not complete.",
                "run": run.to_dict() if run else None,
            }

        # 2. Validate (Output Contract)
        run_dir = self.store.get_run_dir(workspace_id, run.run_id)
        results_path = os.path.join(run_dir, "artifacts", "results.json")
        if not os.path.exists(results_path):
            results_path = os.path.join(run_dir, "results.json")

        if not os.path.exists(results_path):
            warnings.append("Output contract validation warning: results.json not found in artifacts directory.")
            results_payload = {"metrics": {}}
        else:
            with open(results_path, "r", encoding="utf-8") as f:
                results_payload = json.load(f)

        raw_metrics = results_payload.get("metrics", {})

        # 3. Evaluate (metrics collected)
        ingested_results = self.store.get_results_for_run(workspace_id, run.run_id)
        metrics_map = {r.metric: r.value for r in ingested_results}
        if not metrics_map and raw_metrics:
            metrics_map = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}

        # 4. Compare (with baseline if available)
        baseline_info: Dict[str, Any] = {}
        comparability_info: Dict[str, Any] = {}
        if experiment.baseline:
            all_res = self.store.list_results(workspace_id)
            base_res = [r for r in all_res if r.model == experiment.baseline and r.run_id != run.run_id]
            if base_res:
                comparisons = ExperimentService.compare_baseline(
                    proposed_results=ingested_results,
                    baseline_results=base_res,
                )
                if comparisons:
                    comparability_info = comparisons[0]
                    baseline_info = {
                        "model": experiment.baseline,
                        "results": {c["metric"]: c["baseline_value"] for c in comparisons},
                    }
            else:
                warnings.append(f"Baseline model '{experiment.baseline}' has no prior evaluated results in workspace.")

        # 5. Track (Persist evaluation.json contract)
        eval_contract = EvaluationContract(
            metrics=metrics_map,
            baseline=baseline_info,
            comparability=comparability_info,
            warnings=warnings,
        )

        eval_json_path = os.path.join(run_dir, "evaluation.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_contract.to_dict(), f, indent=2)

        # Also copy to reports
        reports_dir = os.path.join(run_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(eval_contract.to_dict(), f, indent=2)

        # 6. Report (Structured summary without automatic scientific conclusion)
        return {
            "status": "COMPLETED",
            "run_id": run.run_id,
            "experiment_id": experiment.experiment_id,
            "evaluation_contract": eval_contract.to_dict(),
            "pipeline_stages": [
                "1. EXECUTE: COMPLETED",
                "2. VALIDATE: COMPLETED",
                "3. EVALUATE: COMPLETED",
                "4. COMPARE: COMPLETED",
                "5. TRACK: COMPLETED",
                "6. REPORT: COMPLETED",
            ],
            "scientific_status": "EVALUATED_NOT_CONCLUDED",
            "note": "Evaluation completed. Empirical results require human researcher review before scientific conclusions.",
        }
