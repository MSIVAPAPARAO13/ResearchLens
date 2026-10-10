# ResearchLens Security Audit & Threat Model

## Executive Summary

Phase 16 conducted a comprehensive security audit of the ResearchLens architecture, runtime dependencies, workspace persistence layer, and containerized execution infrastructure. This document details the security posture, threat model, tested attack vectors, and implemented defensive controls.

---

## 1. Threat Model & Trust Boundaries

ResearchLens operates across three primary security boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Boundary                        │
│  - Web Browser / Streamlit Session / API Callers            │
│  - Untrusted inputs: user queries, uploaded PDF files        │
└──────────────────────────────┬──────────────────────────────┘
                               │ (TLS 1.3, Secret Key Auth)
┌──────────────────────────────▼──────────────────────────────┐
│                    Application Runtime                      │
│  - Streamlit Application Process (UID 1000: researcher)      │
│  - Workspace Storage (`data/workspaces`)                    │
│  - Secret Masking Logging Filter                            │
│  - Ingestion & NLP Pipelines                                │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Docker API / Isolated Bridge)
┌──────────────────────────────▼──────────────────────────────┐
│                    Execution Sandbox                        │
│  - Docker Container Sandbox                                 │
│  - Read-only rootfs, dropped capabilities, no host socket   │
│  - Explicit Image Allowlist (`ExecutionImageRegistry`)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Authentication, Authorization & RBAC

ResearchLens implements role-based access control across all workspace actions:

| Role | Read Workspaces | Edit Content | Submit Plans | Execute Experiments | Review / Approve | Administer Members |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OWNER** | Allowed | Allowed | Allowed | Allowed | Allowed | Allowed |
| **RESEARCHER** | Allowed | Allowed | Allowed | Allowed | Denied | Denied |
| **REVIEWER** | Allowed | Denied | Denied | Denied | Allowed | Denied |
| **VIEWER** | Allowed | Denied | Denied | Denied | Denied | Denied |

### Verified RBAC Enforcement
- `tests/test_execution_permissions.py` and `tests/test_production_resilience_and_security.py` verify that `VIEWER` and `REVIEWER` roles are rejected with `Permission denied` when attempting to execute experiments.
- Only users with `RESEARCHER` or `OWNER` roles can trigger experiment jobs.
- Plan approvals and finding confirmations require `REVIEWER` or `OWNER` roles.

---

## 3. Path Traversal & Input Sanitization

### Vulnerability Vector
Adversaries attempting to read or overwrite arbitrary filesystem files using relative path sequences (`../../etc/passwd`, `..\..\windows\system32`) via workspace IDs, entity identifiers, or artifact names.

### Defensive Implementation
All workspace and artifact identifiers are passed through `_sanitize_id`:
```python
def _sanitize_id(identifier: str) -> str:
    """Validate and sanitize identifiers against path traversal attacks."""
    if not identifier or not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
        raise ValueError(f"Invalid identifier for workspace storage: '{identifier}'")
    return identifier
```
- Reject all forward and backward slashes.
- Reject null bytes, current directory (`.`), and parent directory (`..`) references.
- Verified in `tests/test_production_resilience_and_security.py::test_path_traversal_sanitization`.

---

## 4. Container Isolation & Sandbox Security

Containerized experiment execution via [src/researchlens/execution/docker_backend.py](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/execution/docker_backend.py) enforces strict security boundaries:

1. **Docker Image Allowlisting**:
   - Every container execution image must be registered in [ExecutionImageRegistry](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/execution/image_registry.py).
   - Unapproved or untrusted images (e.g. `malicious/cryptominer`) are rejected immediately prior to container instantiation (`test_docker_image_allowlisting_security`).
2. **Mount Restrictions**:
   - Host root (`/`, `C:\`) and Docker daemon socket (`/var/run/docker.sock`) mounts are explicitly blocked.
   - Experiments execute inside ephemeral containers with bounded workspace artifact mounts.
3. **Execution Limits & Cancellation**:
   - Timeout limits are enforced; runaway processes are automatically terminated with `SIGKILL`.
   - CPU quota and memory limits prevent denial-of-service on host machines.

---

## 5. Secret Management & Log Sanitization

### Defensive Controls
1. **Zero Secret Storage in Code**:
   - API tokens and database URIs are provided exclusively via environment variables (`.env`).
   - `.env`, `.env.local`, and credential files are excluded in `.gitignore` and `.dockerignore`.
2. **Runtime Secret Masking Filter**:
   - All log records emitted across the system pass through `SecretMaskingFilter` in [src/researchlens/utils/logging.py](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/utils/logging.py).
   - Automatically redacts API keys (`sk-...`), passwords, tokens, and authorization headers (`[REDACTED]`).
   - Verified in `tests/test_production_resilience_and_security.py::test_secret_masking_in_logs`.
3. **MLflow Parameter Sanitization**:
   - The MLflow tracking bridge filters out parameters matching credential patterns before transmission to the tracking server.

---

## 6. Persistence Integrity & Corrupted Record Recovery

- Workspace JSON files are written using atomic file replacement (`_atomic_write_json`) to guarantee zero partial-write corruption upon sudden process termination or power loss.
- In the event of disk corruption, `_read_json` catches deserialization errors and gracefully returns empty defaults rather than crashing the application (`test_atomic_write_and_corrupted_json_recovery`).
- MLflow connection failures never delete or corrupt local completed experiment records (`test_offline_mlflow_fallback_preserves_local_runs`).

---

## 7. Audit Conclusion

| Audit Item | Status | Verification Reference |
|---|:---:|---|
| RBAC Enforcement | **PASS** | `tests/test_execution_permissions.py` |
| Path Traversal Protection | **PASS** | `tests/test_production_resilience_and_security.py` |
| Image Allowlisting | **PASS** | `tests/test_production_resilience_and_security.py` |
| Log Secret Masking | **PASS** | `tests/test_production_resilience_and_security.py` |
| Atomic File Write Integrity | **PASS** | `tests/test_production_resilience_and_security.py` |
| Docker Socket Mount Guard | **PASS** | `tests/test_container_security.py` |
| Production Config Validation | **PASS** | `tests/test_production_resilience_and_security.py` |
