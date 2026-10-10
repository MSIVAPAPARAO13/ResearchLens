"""Tests for Artifact Store Abstraction (Local & Mocked S3) (Phase 14)."""

import os
import tempfile

from researchlens.execution.artifact_store import (
    LocalArtifactStore,
    S3CompatibleArtifactStore,
    get_artifact_store,
)


def test_local_artifact_store_lifecycle():
    """Verify local store put, get, exists, hash, and delete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_dir = os.path.join(tmp_dir, "store")
        store = LocalArtifactStore(base_dir=store_dir)

        src_file = os.path.join(tmp_dir, "weights.pt")
        with open(src_file, "w") as f:
            f.write("model_binary_content_123")

        # 1. Put
        assert store.put("checkpoints/weights.pt", src_file) is True
        assert store.exists("checkpoints/weights.pt") is True

        # 2. Hash
        h = store.hash("checkpoints/weights.pt")
        assert len(h) == 64

        # 3. Get
        dest_file = os.path.join(tmp_dir, "downloaded.pt")
        assert store.get("checkpoints/weights.pt", dest_file) is True
        assert os.path.exists(dest_file)
        with open(dest_file, "r") as f:
            assert f.read() == "model_binary_content_123"

        # 4. Missing artifact
        assert store.exists("non_existent.bin") is False
        assert store.get("non_existent.bin", os.path.join(tmp_dir, "out.bin")) is False

        # 5. Delete
        assert store.delete("checkpoints/weights.pt") is True
        assert store.exists("checkpoints/weights.pt") is False


def test_s3_compatible_store_mocked(monkeypatch):
    """Verify S3 compatible store operations with mocked boto3 client."""
    s3_store = S3CompatibleArtifactStore(bucket="my-bucket")

    class MockBotoClient:
        def __init__(self):
            self.uploaded = {}

        def upload_file(self, src, bucket, key):
            self.uploaded[key] = src

        def download_file(self, bucket, key, dest):
            if key not in self.uploaded:
                raise Exception("NoSuchKey")
            with open(dest, "w") as f:
                f.write("mocked_s3_data")

        def head_object(self, Bucket, Key):
            if Key not in self.uploaded:
                raise Exception("404")
            return {"ETag": '"abc123etag"'}

        def delete_object(self, Bucket, Key):
            self.uploaded.pop(Key, None)

    mock_client = MockBotoClient()
    monkeypatch.setattr(s3_store, "_get_client", lambda: mock_client)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = os.path.join(tmp_dir, "data.csv")
        with open(sample, "w") as f:
            f.write("a,b\n1,2\n")

        # Put
        assert s3_store.put("datasets/data.csv", sample) is True
        assert s3_store.exists("datasets/data.csv") is True

        # Hash (ETag)
        assert s3_store.hash("datasets/data.csv") == "abc123etag"

        # Get
        dest = os.path.join(tmp_dir, "out.csv")
        assert s3_store.get("datasets/data.csv", dest) is True

        # Delete
        assert s3_store.delete("datasets/data.csv") is True
        assert s3_store.exists("datasets/data.csv") is False


def test_artifact_store_factory(monkeypatch):
    """Verify artifact store factory selects based on environment variable."""
    monkeypatch.setenv("ARTIFACT_STORE", "LOCAL")
    assert isinstance(get_artifact_store(), LocalArtifactStore)

    monkeypatch.setenv("ARTIFACT_STORE", "S3")
    assert isinstance(get_artifact_store(), S3CompatibleArtifactStore)
