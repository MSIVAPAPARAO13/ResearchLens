# Remote Execution Contract Architecture

## 1. Design & Purpose

Phase 14 defines a clean, extensible architectural contract for future remote execution orchestrators without introducing premature dependencies on cloud providers (AWS Batch, Azure ML, GCP Vertex, Slurm, Ray, Kubernetes).

```
ResearchLens Run
      ↓
RemoteExecutionBackend
      ↓
[Future Cloud / Slurm Provider]
```

## 2. Interface Contract

Defined in `src/researchlens/execution/backends/remote.py`:

```python
class RemoteExecutionBackend(BaseExecutionBackend):
    def submit(self, job: ExecutionJob) -> Dict[str, Any]: ...
    def status(self, remote_job_id: str) -> Dict[str, Any]: ...
    def logs(self, remote_job_id: str) -> Dict[str, str]: ...
    def cancel(self, remote_job_id: str) -> bool: ...
    def artifacts(self, remote_job_id: str, dest_dir: str) -> List[str]: ...
```

## 3. Initial Status

Because no external cloud provider credentials or clusters are assumed in local developer workstations, the backend returns:
- Status: `NOT_CONFIGURED`
- Error: `Remote execution backend is not configured in this environment.`

This cleanly prepares the codebase for Phase 15 distributed computing integrations.
