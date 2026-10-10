# Execution Security & Sandboxing Architecture

## 1. Security Philosophy

ResearchLens follows a zero-trust model for experiment execution. It enables automated research workflows while strictly defending against malicious script execution, arbitrary command injection, and data exfiltration.

---

## 2. Security Boundaries & Enforcements

### 2.1 No Arbitrary Code Execution
- **User Uploads**: Users cannot upload and execute arbitrary `.py` scripts, shell scripts, or binary files.
- **Forbidden Extensions**: Files ending in `.exe`, `.bat`, `.cmd`, `.ps1`, `.sh`, `.vbs`, or `.ipynb` are rejected immediately.
- **Jupyter Notebooks**: Arbitrary notebook execution is explicitly prohibited.

### 2.2 Shell Injection Prevention
- All subprocesses are launched via Python's `subprocess.run()` or `subprocess.Popen()` with `shell=False`.
- Command arguments are strictly passed as typed lists of strings (`List[str]`).
- Arguments are validated using `SecurityValidator.validate_arguments()`, rejecting characters commonly used for command chaining and shell substitution (`;`, `&&`, `||`, `|`, `` ` ``, `$`).

### 2.3 Strict Directory Containment & Path Traversal Guards
- All entrypoints must reside within pre-approved repository directories:
  - `scripts/experiments/`
  - `experiments/`
  - `src/researchlens/experiments/`
- Traversal components (`..`, `/..`, `\..`) are rejected by `SecurityValidator.validate_entrypoint()`.
- Run workspaces are created with sanitized identifiers under `data/workspaces/<workspace_id>/runs/<run_id>/`.

### 2.4 Credential & Secret Masking
Environment variables and configuration dictionaries captured in pre-flight snapshots or logs are scrubbed against credential keywords:
- Patterns masked: `SECRET`, `TOKEN`, `KEY`, `PASSWORD`, `CREDENTIAL`, `AUTH`, `API_KEY`.
- Substituted with: `[MASKED]`.
- Plaintext secrets are never persisted to disk or included in manifests.

### 2.5 Role-Based Execution Permissions
- `OWNER` & `RESEARCHER`: Authorized to launch runs, re-run experiments, and cancel active runs.
- `REVIEWER` & `VIEWER`: Restricted to read-only viewing of logs, results, manifests, and artifacts.
