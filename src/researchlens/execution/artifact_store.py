"""Artifact storage abstraction supporting Local and S3-compatible stores (Phase 14)."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional

from researchlens.execution.artifacts import calculate_sha256
from researchlens.utils.logging import get_logger

logger = get_logger("artifact_store")


class BaseArtifactStore(ABC):
    """Abstract interface for artifact storage."""

    @abstractmethod
    def put(self, name: str, src_path: str) -> bool:
        """Upload or store an artifact file."""
        pass

    @abstractmethod
    def get(self, name: str, dest_path: str) -> bool:
        """Download or retrieve an artifact file."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if an artifact exists in storage."""
        pass

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete an artifact from storage."""
        pass

    @abstractmethod
    def hash(self, name: str) -> str:
        """Return the SHA-256 hash of an artifact."""
        pass


class LocalArtifactStore(BaseArtifactStore):
    """Filesystem-based artifact store (default)."""

    def __init__(self, base_dir: str = "./data/artifacts"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _resolve(self, name: str) -> str:
        clean = name.replace("\\", "/").lstrip("/")
        return os.path.join(self.base_dir, clean)

    def put(self, name: str, src_path: str) -> bool:
        if not os.path.exists(src_path):
            return False
        dest = self._resolve(name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src_path, dest)
        return True

    def get(self, name: str, dest_path: str) -> bool:
        src = self._resolve(name)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        shutil.copy2(src, dest_path)
        return True

    def exists(self, name: str) -> bool:
        return os.path.exists(self._resolve(name))

    def delete(self, name: str) -> bool:
        path = self._resolve(name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def hash(self, name: str) -> str:
        path = self._resolve(name)
        return calculate_sha256(path)


class S3CompatibleArtifactStore(BaseArtifactStore):
    """Optional S3-compatible remote object store (MinIO, Ceph, AWS S3).

    Does not require AWS credentials for initialization.
    Credentials must originate from environment variables, never workspace files or logs.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.bucket = bucket or os.getenv("S3_BUCKET", "researchlens-artifacts")
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT")
        self.region = region or os.getenv("S3_REGION", "us-east-1")
        self._s3_client = None

    def _get_client(self):
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3

            # Credentials retrieved strictly from secure environment variables
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            client_kwargs = {
                "region_name": self.region,
            }
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if aws_access_key and aws_secret_key:
                client_kwargs["aws_access_key_id"] = aws_access_key
                client_kwargs["aws_secret_access_key"] = aws_secret_key

            self._s3_client = boto3.client("s3", **client_kwargs)
            return self._s3_client
        except Exception as e:
            logger.debug(f"S3 client initialization deferred or unavailable: {e}")
            return None

    def put(self, name: str, src_path: str) -> bool:
        client = self._get_client()
        if not client or not os.path.exists(src_path):
            return False
        try:
            client.upload_file(src_path, self.bucket, name)
            return True
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return False

    def get(self, name: str, dest_path: str) -> bool:
        client = self._get_client()
        if not client:
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
            client.download_file(self.bucket, name, dest_path)
            return True
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return False

    def exists(self, name: str) -> bool:
        client = self._get_client()
        if not client:
            return False
        try:
            client.head_object(Bucket=self.bucket, Key=name)
            return True
        except Exception:
            return False

    def delete(self, name: str) -> bool:
        client = self._get_client()
        if not client:
            return False
        try:
            client.delete_object(Bucket=self.bucket, Key=name)
            return True
        except Exception as e:
            logger.error(f"S3 delete error: {e}")
            return False

    def hash(self, name: str) -> str:
        """Compute hash if object exists via ETag or head."""
        client = self._get_client()
        if not client:
            return "NOT_AVAILABLE"
        try:
            head = client.head_object(Bucket=self.bucket, Key=name)
            etag = head.get("ETag", "").strip('"')
            return etag or "NOT_AVAILABLE"
        except Exception:
            return "NOT_AVAILABLE"


def get_artifact_store() -> BaseArtifactStore:
    """Factory selecting the configured artifact store from environment."""
    store_type = os.getenv("ARTIFACT_STORE", "LOCAL").strip().upper()
    if store_type in ["S3", "S3_COMPATIBLE"]:
        return S3CompatibleArtifactStore()
    return LocalArtifactStore()
