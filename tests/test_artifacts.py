"""Unit tests for artifact discovery and categorization (Phase 12)."""

import os
import shutil
import tempfile

from researchlens.execution.artifacts import discover_artifacts


def test_artifact_discovery_and_categorization():
    tmp = tempfile.mkdtemp(prefix="rl_test_art_")
    try:
        # Create sample artifacts
        with open(os.path.join(tmp, "model.pt"), "w") as f:
            f.write("mock_model_weights")
        with open(os.path.join(tmp, "metrics.json"), "w") as f:
            f.write('{"accuracy": 0.95}')
        with open(os.path.join(tmp, "roc_curve.png"), "w") as f:
            f.write("mock_png")
        with open(os.path.join(tmp, "evaluation.csv"), "w") as f:
            f.write("model,dataset\nA,B")
        with open(os.path.join(tmp, "config.yaml"), "w") as f:
            f.write("lr: 0.001")

        artifacts = discover_artifacts(tmp, run_id="run_art_01")
        assert len(artifacts) == 5

        type_map = {a.name: a.type for a in artifacts}
        assert type_map["model.pt"] == "MODEL"
        assert type_map["metrics.json"] == "METRICS"
        assert type_map["roc_curve.png"] == "PLOT"
        assert type_map["evaluation.csv"] == "TABLE"
        assert type_map["config.yaml"] == "CONFIG"

        # Ensure sizes and hashes populated
        for a in artifacts:
            assert a.size_bytes > 0
            assert len(a.sha256) == 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
