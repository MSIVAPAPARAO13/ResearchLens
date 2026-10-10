# ResearchLens Execution Backend Architecture

## 1. Overview & Abstraction

Phase 14 introduces the Execution Backend Abstraction to decouple experiment specification from execution environments without modifying any scientific, analytical, or algorithmic logic.

```
ResearchLens
      ↓
Experiment Registry
      ↓
Execution Specification (ExecutionJob)
      ↓
Execution Backend (BaseExecutionBackend)
 ┌────┼──────────┐
 ↓    ↓          ↓
LOCAL DOCKER   REMOTE
      ↓
Experiment Run
      ↓
MLflow Tracking
      ↓
Artifacts & Findings
```

## 2. Base Backend Contract

Defined in `src/researchlens/execution/backends/base.py`:

```python
class BaseExecutionBackend(ABC):
    @property
    def backend_type(self) -> str: ...
    def validate(self, job: ExecutionJob, experiment: Optional[ExperimentPlan] = None) -> Tuple[bool, str]: ...
    def prepare(self, job: ExecutionJob, run_dir: str) -> Tuple[bool, str]: ...
    def start(self, job: ExecutionJob, run_dir: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]: ...
    def cancel(self, run_id: str) -> bool: ...
    def collect_logs(self, job: ExecutionJob, run_dir: str) -> Dict[str, str]: ...
    def collect_artifacts(self, job: ExecutionJob, run_dir: str) -> List[ExperimentArtifact]: ...
    def cleanup(self, job: ExecutionJob, run_dir: str) -> None: ...
```

## 3. Supported Backends

| Backend | Identifier | Enforced Security / Isolation | Default Status |
| :--- | :--- | :--- | :--- |
| **Local Process** | `LOCAL` | Sandboxed subprocess, timeout killer, directory isolation, safe argument parsing (`shell=False`) | `AVAILABLE` (Default) |
| **Docker Container** | `DOCKER` | Non-root container (`1000:1000`), `--cap-drop ALL`, `no-new-privileges`, CPU/Memory limits, network disablement, image allowlist | `AVAILABLE` (if Docker daemon running) |
| **Remote Orchestration** | `REMOTE` | Interface contract for AWS/Slurm/Kubernetes/Ray | `NOT_CONFIGURED` |

## 4. Enforcement Transparency

If the host execution platform or operating system cannot strictly enforce a requested constraint (e.g. cgroups CPU quota on Windows native processes):
- The backend reports `LIMIT_NOT_ENFORCED`.
- ResearchLens never fabricates or claims enforcement without OS confirmation.
