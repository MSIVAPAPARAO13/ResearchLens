"""Artifact discovery, SHA-256 calculation, and integrity auditing (Phase 12)."""

import hashlib
import os
from typing import List

from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import ExperimentArtifact, generate_id

logger = get_logger("execution_artifacts")

# Extensions mapping to artifact types
ARTIFACT_EXTENSIONS = {
    ".pt": "MODEL",
    ".pth": "MODEL",
    ".bin": "MODEL",
    ".onnx": "MODEL",
    ".safetensors": "MODEL",
    ".json": "METRICS",
    ".csv": "TABLE",
    ".tsv": "TABLE",
    ".parquet": "TABLE",
    ".png": "PLOT",
    ".jpg": "PLOT",
    ".jpeg": "PLOT",
    ".pdf": "REPORT",
    ".md": "REPORT",
    ".log": "LOG",
    ".txt": "LOG",
    ".yaml": "CONFIG",
    ".yml": "CONFIG",
}


def calculate_sha256(filepath: str) -> str:
    """Calculate the cryptographic SHA-256 checksum of a file."""
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return ""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate SHA-256 for '{filepath}': {e}")
        return ""


def verify_artifact_hash(filepath: str, expected_sha256: str) -> bool:
    """Verify that an artifact matches its recorded SHA-256 hash."""
    if not expected_sha256:
        return False
    actual = calculate_sha256(filepath)
    return actual.lower() == expected_sha256.lower()


def discover_artifacts(artifacts_dir: str, run_id: str) -> List[ExperimentArtifact]:
    """Discover, hash, and register all generated experiment output artifacts.

    Strictly restricts discovery within artifacts_dir to guard against path traversal.
    """
    artifacts: List[ExperimentArtifact] = []
    if not os.path.exists(artifacts_dir) or not os.path.isdir(artifacts_dir):
        return artifacts

    base_abs = os.path.abspath(artifacts_dir)

    for root, _, files in os.walk(base_abs):
        for fname in files:
            full_path = os.path.join(root, fname)

            # Security check: must reside inside base_abs
            is_safe, _ = SecurityValidator.validate_safe_path(full_path, base_abs)
            if not is_safe:
                logger.warning(f"Skipping artifact outside designated directory: {full_path}")
                continue

            rel_path = os.path.relpath(full_path, base_abs).replace("\\", "/")
            _, ext = os.path.splitext(fname.lower())
            art_type = ARTIFACT_EXTENSIONS.get(ext, "METRICS")

            try:
                size = os.path.getsize(full_path)
                sha = calculate_sha256(full_path)
                art = ExperimentArtifact(
                    artifact_id=generate_id("art"),
                    run_id=run_id,
                    name=fname,
                    path=rel_path,
                    type=art_type,
                    size_bytes=size,
                    sha256=sha,
                )
                artifacts.append(art)
            except Exception as e:
                logger.error(f"Error registering artifact '{fname}': {e}")

    return artifacts
