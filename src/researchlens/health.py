"""System health verification and runtime diagnostic checks for ResearchLens (Phase 16)."""

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from researchlens.config.settings import get_settings
from researchlens.mlops.client import check_mlflow_health
from researchlens.utils.logging import get_logger

logger = get_logger("health")


def check_docker_availability() -> Dict[str, Any]:
    """Check whether Docker daemon is installed and responding."""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip()
            return {"available": True, "server_version": version, "message": "Docker daemon connected."}
        return {"available": False, "server_version": None, "message": f"Docker exited with {proc.returncode}."}
    except FileNotFoundError:
        return {"available": False, "server_version": None, "message": "Docker CLI binary not found in PATH."}
    except subprocess.TimeoutExpired:
        return {"available": False, "server_version": None, "message": "Docker daemon connection timed out."}
    except Exception as e:
        return {"available": False, "server_version": None, "message": f"Docker error: {str(e)}"}


def check_workspace_storage(base_dir: str) -> Dict[str, Any]:
    """Verify writeability and capacity of the workspace storage directory."""
    try:
        os.makedirs(base_dir, exist_ok=True)
        test_file = os.path.join(base_dir, ".health_check_probe.tmp")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("probe")
        if os.path.exists(test_file):
            os.remove(test_file)

        usage = shutil.disk_usage(base_dir)
        free_gb = round(usage.free / (1024**3), 2)
        total_gb = round(usage.total / (1024**3), 2)

        return {
            "accessible": True,
            "path": os.path.abspath(base_dir),
            "free_space_gb": free_gb,
            "total_space_gb": total_gb,
            "writeable": True,
        }
    except Exception as e:
        return {
            "accessible": False,
            "path": os.path.abspath(base_dir),
            "error": str(e),
            "writeable": False,
        }


def check_nlp_components() -> Dict[str, Any]:
    """Check availability of core NLP runtime libraries."""
    components: Dict[str, bool] = {}
    for mod_name in ["fitz", "torch", "transformers", "sentence_transformers", "faiss", "rank_bm25", "rouge_score", "spacy", "nltk"]:
        try:
            __import__(mod_name)
            components[mod_name] = True
        except ImportError:
            components[mod_name] = False
    return components


def check_system_health() -> Dict[str, Any]:
    """Execute complete health audit and return structured diagnostic status."""
    settings = get_settings()

    config_issues = settings.validate_production_configuration()
    storage_health = check_workspace_storage(settings.workspaces_dir)
    mlflow_status = check_mlflow_health()
    docker_status = check_docker_availability()
    nlp_status = check_nlp_components()

    # Determine PyTorch device
    device_info = {
        "configured_device": settings.device,
        "cuda_available": False,
        "device_count": 0,
    }
    try:
        import torch
        device_info["cuda_available"] = torch.cuda.is_available()
        device_info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        pass

    # Determine overall status: HEALTHY, DEGRADED, UNHEALTHY
    unhealthy_reasons: List[str] = []
    degraded_reasons: List[str] = []

    if not storage_health.get("accessible"):
        unhealthy_reasons.append("Workspace storage directory is not accessible or not writeable.")
    if not all([nlp_status.get("fitz"), nlp_status.get("torch"), nlp_status.get("transformers")]):
        unhealthy_reasons.append("Critical NLP libraries (PyMuPDF, PyTorch, Transformers) are missing.")
    if config_issues:
        degraded_reasons.extend(config_issues)
    if mlflow_status.get("status") in ("UNAVAILABLE", "MISCONFIGURED"):
        degraded_reasons.append(f"MLflow service status: {mlflow_status.get('status')}")
    if not docker_status.get("available"):
        degraded_reasons.append("Docker daemon is not available (local execution fallback active).")

    if unhealthy_reasons:
        overall_status = "UNHEALTHY"
    elif degraded_reasons:
        overall_status = "DEGRADED"
    else:
        overall_status = "HEALTHY"

    return {
        "status": overall_status,
        "environment": settings.env,
        "debug": settings.debug,
        "platform": {
            "os": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "architecture": platform.machine(),
        },
        "compute": device_info,
        "storage": storage_health,
        "mlflow": mlflow_status,
        "docker": docker_status,
        "nlp_dependencies": nlp_status,
        "configuration_issues": config_issues,
        "unhealthy_reasons": unhealthy_reasons,
        "degraded_reasons": degraded_reasons,
    }


def main() -> int:
    """CLI diagnostic entrypoint for containers and monitoring."""
    health = check_system_health()
    print(json.dumps(health, indent=2))
    return 0 if health["status"] in ("HEALTHY", "DEGRADED") else 1


if __name__ == "__main__":
    sys.exit(main())
