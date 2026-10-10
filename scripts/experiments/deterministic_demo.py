"""Deterministic baseline experiment script for Phase 12 validation.

Inputs:
--seed: integer random seed
--dataset: evaluation dataset name
--output-dir: destination directory for results.json and artifacts
"""

import argparse
import hashlib
import json
import os


def run_deterministic_experiment(seed: int, dataset: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Compute deterministic metric values from seed and dataset
    seed_hash = int(hashlib.sha256(f"{dataset}_{seed}".encode("utf-8")).hexdigest(), 16)
    base_accuracy = 0.85 + ((seed_hash % 1000) / 10000.0)  # [0.850, 0.950]
    base_f1 = base_accuracy - 0.015

    results_payload = {
        "metrics": {
            "accuracy": round(base_accuracy, 4),
            "f1": round(base_f1, 4),
            "latency": 18.5,
        },
        "metadata": {
            "dataset": dataset,
            "seed": seed,
            "model": "DeterministicBaseline",
            "eval_split": "test",
        },
        "artifacts": [
            "predictions.json",
        ],
    }

    # Write output contract results.json
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Write sample artifact
    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump({"sample_count": 100, "correct": int(100 * base_accuracy)}, f, indent=2)

    print(f"Experiment completed successfully. Results written to {results_path}")
    print(f"Metrics: Accuracy={base_accuracy:.4f}, F1={base_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Deterministic Experiment Demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="SQuAD_v2", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")

    args, _ = parser.parse_known_args()
    run_deterministic_experiment(seed=args.seed, dataset=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
