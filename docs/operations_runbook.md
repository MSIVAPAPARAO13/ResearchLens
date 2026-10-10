# ResearchLens Operations & Maintenance Runbook

## Overview

This operational runbook provides sysadmins, DevOps engineers, and platform operators with standard operating procedures (SOPs), diagnostic playbooks, reverse proxy administration, backup/restore routines, and disaster recovery workflows for the ResearchLens platform.

---

## 1. Routine Service Management

### Starting Services (Docker Compose)
```bash
docker-compose up -d
```

### Stopping Services
```bash
docker-compose down
```

### Viewing Logs
```bash
# ResearchLens Application Logs (Structured JSON)
docker-compose logs -f --tail=100 researchlens

# MLflow Tracking Server Logs
docker-compose logs -f --tail=100 mlflow

# Reverse Proxy & Access Logs
docker-compose logs -f --tail=100 reverse-proxy
```

---

## 2. Reverse Proxy & Security Administration

### TLS Certificate Renewal (Certbot)
When using Let's Encrypt certificates on a production domain:
```bash
certbot renew
# Copy updated certificates to nginx volume
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./nginx/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./nginx/ssl/key.pem

# Gracefully reload Nginx without dropping active connections
docker-compose exec reverse-proxy nginx -s reload
```

### Local / Staging Certificate Regeneration
```bash
python scripts/generate_tls_certs.py
docker-compose restart reverse-proxy
```

### Managing MLflow Interface Access (`.htpasswd`)
The MLflow interface is gated behind HTTP Basic Authentication:
```bash
# Add a new operator account or update password
htpasswd -B ./nginx/.htpasswd <username>

# Gracefully reload Nginx to apply changes
docker-compose exec reverse-proxy nginx -s reload
```

---

## 3. Health Checks & Continuous Monitoring

ResearchLens exposes a built-in health diagnostic module: [src/researchlens/health.py](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/health.py).

### Running Health Diagnostic
```bash
python -m researchlens.health
```

### Interpreting Health Status Codes
- **Exit Code `0` (`HEALTHY`)**: All subsystems (storage, NLP libraries, MLflow, Docker backend) are fully operational.
- **Exit Code `1` (`UNHEALTHY` / `DEGRADED`)**: Storage directory is unwritable, free disk space is under 1 GB, or critical ML runtime components are missing.

### Monitoring Key Indicators
| Metric / Check | Threshold | Action |
|---|---|---|
| Disk Free Space | `< 2 GB` | Expand disk volume or purge historical scratch artifacts. |
| Ingestion Memory | `> 4 GB` | Adjust batch size in PDF ingestion pipeline. |
| MLflow Sync Errors | `> 5 consecutive failures` | Verify MLflow network connectivity; check SQLite database lock. |
| 429 Rate Limit Spikes | `> 50 / minute` | Inspect client IP for scraping or DoS activity. |

---

## 4. Backup & Recovery Procedures

### Workspaces Storage Backup (`data/workspaces`)
The workspace persistence layer is filesystem-based, facilitating simple snapshotting:

```bash
# Create timestamped tarball
tar -czvf /backups/workspaces_$(date +%Y%m%d_%H%M%S).tar.gz ./data/workspaces
```

### MLflow Tracking Database Backup
```bash
# Backup SQLite tracking database safely
sqlite3 /mlflow/mlflow.db ".backup '/backups/mlflow_$(date +%Y%m%d_%H%M%S).db'"
```

### Restoration Workflow
1. Stop the application: `docker-compose down`
2. Restore workspace data:
   ```bash
   rm -rf ./data/workspaces/*
   tar -xzvf /backups/workspaces_<backup_date>.tar.gz -C ./
   ```
3. Verify directory permissions:
   ```bash
   chown -R 1000:1000 ./data/workspaces
   ```
4. Restart application: `docker-compose up -d`
5. Execute health check: `docker-compose exec researchlens python -m researchlens.health`

---

## 5. Incident Response & Troubleshooting Playbook

### Incident A: "Permission denied" during experiment execution
- **Cause**: User role is `VIEWER` or `REVIEWER` instead of `RESEARCHER` or `OWNER`.
- **Resolution**: Check user role in `data/workspaces/<workspace_id>/members.json`. Promote member role using `store.save_member(member)` with `role="RESEARCHER"`.

### Incident B: MLflow Tracking Server Unreachable
- **Symptoms**: Logs show `Warning: Connection error to MLflow server`.
- **Impact**: Zero data loss. ResearchLens persists experiment runs locally in `data/workspaces/<workspace_id>/runs.json` and continues normal operations.
- **Resolution**:
  1. Check MLflow container status: `docker-compose ps mlflow`
  2. Restart MLflow service: `docker-compose restart mlflow`
  3. Run sync service to push pending runs once connection is restored.

### Incident C: 401 Unauthorized on `/mlflow/`
- **Symptoms**: User cannot access MLflow dashboard.
- **Resolution**: Verify user credentials against `nginx/.htpasswd`. Reset operator credentials using `htpasswd -B ./nginx/.htpasswd <username>`.

### Incident D: Corrupted JSON File in Workspace
- **Symptoms**: Warning in logs: `Error reading JSON from <path>: Expecting value`.
- **Impact**: ResearchLens safely isolates corrupt files by returning empty defaults, preventing application crashes.
- **Resolution**:
  1. Inspect the damaged file in `data/workspaces/<workspace_id>/`.
  2. If a corresponding `.tmp_ws_*` file exists, inspect if it contains valid recent JSON.
  3. Restore the corrupted file from the latest daily backup snapshot.

### Incident E: Docker Execution Sandbox Unavailable
- **Symptoms**: Health check reports `docker_execution.status = "NOT_CONFIGURED"`.
- **Resolution**:
  1. Ensure Docker Desktop or dockerd is running on the host.
  2. Verify user has permissions to connect to `/var/run/docker.sock` or Windows named pipe `npipe:////./pipe/docker_engine`.
  3. In development environments without Docker, configure execution backend to use `local` execution mode.
