"""Tests for Dataset Mount Isolation & Security Boundaries (Phase 14)."""

import os
import tempfile

from researchlens.execution.dataset_mount import DatasetMountValidator
from researchlens.workspace.schema import DatasetMount


def test_dataset_mount_defaults():
    """Verify default read-only and hash attributes."""
    mount = DatasetMount(dataset_id="ds_123", mount_path="/workspace/data")
    assert mount.read_only is True
    assert mount.hash == "NOT_PROVIDED"


def test_prohibited_host_root_and_user_mounts():
    """Verify prohibited root mounts are rejected."""
    # Root paths
    assert DatasetMountValidator.validate_mount_safety("/")[0] is False
    assert DatasetMountValidator.validate_mount_safety("C:\\")[0] is False
    assert DatasetMountValidator.validate_mount_safety("C:/")[0] is False

    # Docker socket
    assert DatasetMountValidator.validate_mount_safety("/var/run/docker.sock")[0] is False

    # Arbitrary user root
    assert DatasetMountValidator.validate_mount_safety("/home/user")[0] is False


def test_valid_dataset_mount_creation():
    """Verify safe dataset path validates and hashes cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = os.path.join(tmp_dir, "dataset.csv")
        with open(sample_file, "w") as f:
            f.write("text,label\nhello,1\n")

        mount, err = DatasetMountValidator.prepare_dataset_mount(
            dataset_id="ds_valid",
            host_dataset_path=sample_file,
            mount_target="/workspace/data",
            read_only=True,
        )
        assert mount is not None
        assert mount.read_only is True
        assert mount.hash != "NOT_PROVIDED"
        assert len(mount.hash) == 64  # SHA-256
