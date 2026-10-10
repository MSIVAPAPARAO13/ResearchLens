"""Unit tests for artifact SHA-256 calculation and integrity verification (Phase 12)."""

import os
import shutil
import tempfile

from researchlens.execution.artifacts import calculate_sha256, verify_artifact_hash


def test_artifact_sha256_and_tamper_detection():
    tmp = tempfile.mkdtemp(prefix="rl_test_hash_")
    try:
        sample_file = os.path.join(tmp, "model.bin")
        with open(sample_file, "wb") as f:
            f.write(b"deterministic_original_data_bytes_12345")

        # 1. Compute hash
        original_hash = calculate_sha256(sample_file)
        assert len(original_hash) == 64
        assert verify_artifact_hash(sample_file, original_hash) is True

        # 2. Modify artifact (tampering simulation)
        with open(sample_file, "ab") as f:
            f.write(b"_tampered_byte")

        # 3. Verify mismatch is detected
        assert verify_artifact_hash(sample_file, original_hash) is False
        new_hash = calculate_sha256(sample_file)
        assert new_hash != original_hash
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
