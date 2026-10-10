"""Environment snapshot capture and hardware detection (Phase 12)."""

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any, Dict

from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger

logger = get_logger("execution_environment")


def get_git_commit_sha(repo_dir: str = ".") -> str:
    """Capture HEAD git commit SHA without fabricating hashes."""
    prov = get_git_provenance(repo_dir)
    return prov.get("commit_sha", "NOT_AVAILABLE")


def get_git_provenance(repo_dir: str = ".") -> Dict[str, Any]:
    """Capture repository_url, branch, commit_sha, and dirty_worktree status (Phase 14)."""
    result = {
        "repository_url": "NOT_AVAILABLE",
        "branch": "NOT_AVAILABLE",
        "commit_sha": "NOT_AVAILABLE",
        "dirty_worktree": False,
    }

    # 1. Commit SHA
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            result["commit_sha"] = res.stdout.strip()
    except Exception as e:
        logger.debug(f"Git commit retrieval failed: {e}")

    # 2. Branch name
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            result["branch"] = res.stdout.strip()
    except Exception as e:
        logger.debug(f"Git branch retrieval failed: {e}")

    # 3. Repository URL
    try:
        res = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            result["repository_url"] = res.stdout.strip()
    except Exception as e:
        logger.debug(f"Git remote url retrieval failed: {e}")

    # 4. Dirty worktree check
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=5,
        )
        if res.returncode == 0:
            result["dirty_worktree"] = bool(res.stdout.strip())
    except Exception as e:
        logger.debug(f"Git dirty check failed: {e}")

    return result


def capture_source_snapshot(repo_dir: str = ".") -> Any:
    """Create a SourceSnapshot representing Git code lineage (Phase 14)."""
    from researchlens.workspace.schema import SourceSnapshot

    prov = get_git_provenance(repo_dir)
    return SourceSnapshot(
        repository=prov["repository_url"],
        branch=prov["branch"],
        commit=prov["commit_sha"],
        dirty=prov["dirty_worktree"],
        snapshot_hash=prov["commit_sha"],
    )


def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU hardware and CUDA version if available, without requiring GPU."""
    gpu_info = {
        "available": False,
        "name": "None",
        "vram_gb": 0.0,
        "cuda_version": "NOT_AVAILABLE",
    }
    # 1. Try PyTorch if installed
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda or "AVAILABLE"
            props = torch.cuda.get_device_properties(0)
            gpu_info["vram_gb"] = round(props.total_memory / (1024**3), 2)
            return gpu_info
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyTorch GPU check returned: {e}")

    # 2. Try nvidia-smi fallback
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_info["available"] = True
            gpu_info["name"] = parts[0]
            if len(parts) > 1 and parts[1].replace(".", "").isdigit():
                gpu_info["vram_gb"] = round(float(parts[1]) / 1024.0, 2)
            gpu_info["cuda_version"] = "DETECTED_VIA_NVIDIA_SMI"
            return gpu_info
    except Exception:
        pass

    return gpu_info


def get_memory_info() -> Dict[str, Any]:
    """Capture host RAM specifications."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
        }
    except Exception:
        return {"total_gb": "NOT_AVAILABLE", "available_gb": "NOT_AVAILABLE"}


def get_installed_packages() -> Dict[str, str]:
    """Collect installed Python package versions."""
    packages = {}
    try:
        for dist in importlib.metadata.distributions():
            packages[dist.metadata["Name"].lower()] = dist.version
    except Exception as e:
        logger.debug(f"Failed to inspect distributions: {e}")
    return packages


def capture_environment_snapshot(working_dir: str = ".") -> Dict[str, Any]:
    """Capture complete reproducible environment snapshot with masked secrets."""
    sanitized_env = SecurityValidator.sanitize_env_vars(dict(os.environ))

    git_prov = get_git_provenance(working_dir)

    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "git_commit": git_prov.get("commit_sha", "NOT_AVAILABLE"),
        "git_provenance": git_prov,
        "source_snapshot": capture_source_snapshot(working_dir).to_dict(),
        "working_directory": os.path.abspath(working_dir),
        "package_versions": get_installed_packages(),
        "environment_variables": sanitized_env,
    }
