# ResearchLens Artifact Storage Architecture

## 1. Overview

ResearchLens supports dual-backend artifact persistence:
1. `LOCAL`: Filesystem-based storage located within `./data/artifacts` or run directory structures.
2. `S3_COMPATIBLE`: Remote object storage supporting AWS S3, MinIO, Ceph, and Cloudflare R2 via optional `boto3`.

## 2. ArtifactStore Interface

Defined in `src/researchlens/execution/artifact_store.py`:

```python
class BaseArtifactStore(ABC):
    def put(self, name: str, src_path: str) -> bool: ...
    def get(self, name: str, dest_path: str) -> bool: ...
    def exists(self, name: str) -> bool: ...
    def delete(self, name: str) -> bool: ...
    def hash(self, name: str) -> str: ...
```

## 3. Configuration & Security

Configured via environment variables:
- `ARTIFACT_STORE`: `LOCAL` or `S3_COMPATIBLE` (Default: `LOCAL`)
- `S3_ENDPOINT`: Optional endpoint URL (e.g. MinIO `http://localhost:9000`)
- `S3_BUCKET`: Target S3 bucket name
- `S3_REGION`: Region identifier

**Security Rule**: AWS Access Keys and Secrets are strictly read from environment variables or IAM instance profiles. They are never written to workspace JSON, MLflow tags, logs, or audit metadata.
