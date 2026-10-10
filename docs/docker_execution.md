# Docker Execution Backend & Sandboxing

## 1. Overview

The Docker Execution Backend provides containerized isolation for scientific experiments. It prevents unintentional host file modifications and resource exhaustion while producing reproducible container runs.

## 2. Security Guardrails

1. **Image Allowlist**:
   - Only images registered in `ExecutionImageRegistry` are permitted.
   - Default approved images: `researchlens/nlp-experiment:1.0`, `researchlens/cpu-baseline:1.0`.
   - Unregistered images are rejected with `BLOCKED`.
2. **Mount Restrictions**:
   - Strictly mounts run output to `/workspace/output`.
   - Never mounts host root (`/`, `C:\`), user home directories (`~`, `/home`, `C:\Users`), or `/var/run/docker.sock`.
   - Attempts to mount forbidden directories are blocked by `DatasetMountValidator`.
3. **Execution Privilege**:
   - Runs as non-root user (`--user 1000:1000`).
   - Drops all Linux capabilities (`--cap-drop ALL`).
   - Enables `--security-opt no-new-privileges:true`.
4. **Network Policy**:
   - Default: `--network none` (`DISABLED`).
   - Network is only enabled (`ENABLED`) if the researcher explicitly requests external dataset download.
5. **Resource Limits**:
   - CPU quota (`--cpus <N>`).
   - Memory ceiling (`--memory <N>`).
   - Process timeout enforcement.

## 3. Dockerfile Templates

Templates reside in `docker/experiments/`:
- `docker/experiments/Dockerfile.cpu`: Lean CPU baseline runtime based on `python:3.11-slim`.
- `docker/experiments/Dockerfile.nlp`: NLP/RAG evaluation runtime with PyTorch, transformers, sentence-transformers, faiss-cpu, and rouge-score.

Images are not built automatically at startup; researchers build explicitly:
```bash
docker build -f docker/experiments/Dockerfile.nlp -t researchlens/nlp-experiment:1.0 .
```
