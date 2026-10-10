# ResearchLens Production Deployment Guide

## Overview

ResearchLens is designed as an enterprise-grade scientific intelligence and automated research platform. This document outlines standard deployment topologies, production configurations, containerization, security hardening, reverse proxy setup, and monitoring procedures for reliable operations.

---

## 1. Architecture Topology

```
                       [ HTTPS / 443 ]
                              │
                    ┌─────────▼─────────┐
                    │   Reverse Proxy   │ (Nginx: ports 80/443, TLS, CSP, HSTS)
                    │  Security Headers │ (Rate Limiting: 50r/s, Max Body: 100M)
                    └────┬───────────┬──┘
                         │           │
       / (Streamlit/WSS) │           │ /mlflow/ (Basic Auth Protected)
                         │           │
           ┌─────────────▼─┐       ┌─▼─────────────┐
           │ ResearchLens  │       │ MLflow Server │
           │ App (:8501)   │       │ (:5000)       │
           │ (Internal Net)│       │ (Internal Net)│
           └───────┬───────┘       └───────┬───────┘
                   │                       │
              Persistent              Persistent
                Storage                 Storage
           `workspaces_data`       `mlflow_data`
```

---

## 2. Containerized Deployment (Docker)

ResearchLens includes an officially hardened [Dockerfile](file:///c:/Users/msiva/Music/ResearchLens/Dockerfile) following security best practices:
- Base image: `python:3.11-slim`
- Least privilege: Unprivileged runtime user `researcher` (`uid=1000:gid=1000`)
- Built-in health check: `python -m researchlens.health`
- Exposed ports: `8501` (Internal container network only)

### Building the Image

```bash
docker build -t researchlens:latest .
```

---

## 3. Multi-Service Composition (`docker-compose`)

For production public deployments, [docker-compose.yml](file:///c:/Users/msiva/Music/ResearchLens/docker-compose.yml) provides a fully isolated 3-tier architecture:

```yaml
version: '3.8'

services:
  reverse-proxy:
    image: nginx:alpine
    container_name: researchlens-reverse-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/.htpasswd:/etc/nginx/.htpasswd:ro
    depends_on:
      - researchlens
      - mlflow
    networks:
      - internal-net

  researchlens:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: researchlens-app
    restart: unless-stopped
    expose:
      - "8501"
    ports:
      - "127.0.0.1:8501:8501"  # Bound to loopback only; public access via reverse proxy
    environment:
      - RESEARCHLENS_ENV=production
      - RESEARCHLENS_DEBUG=false
      - RESEARCHLENS_SECRET_KEY=${RESEARCHLENS_SECRET_KEY}
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - RESEARCHLENS_LOG_FORMAT=json
      - RESEARCHLENS_LOG_LEVEL=INFO
      - RESEARCHLENS_MAX_UPLOAD_SIZE_BYTES=104857600
    volumes:
      - workspaces_data:/app/data/workspaces
    depends_on:
      - mlflow
    healthcheck:
      test: ["CMD", "python", "-m", "researchlens.health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    networks:
      - internal-net

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    container_name: researchlens-mlflow
    restart: unless-stopped
    expose:
      - "5000"
    ports:
      - "127.0.0.1:5000:5000"  # Bound to loopback only; never exposed publicly
    command: >
      mlflow server
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    volumes:
      - mlflow_data:/mlflow
    networks:
      - internal-net

networks:
  internal-net:
    driver: bridge

volumes:
  workspaces_data:
    name: researchlens_workspaces
  mlflow_data:
    name: researchlens_mlflow
```

---

## 4. Configuration & Environment Variables

All production configuration is managed via environment variables. See [.env.example](file:///c:/Users/msiva/Music/ResearchLens/.env.example).

| Variable | Type | Default | Description |
|---|---|---|---|
| `RESEARCHLENS_ENV` | string | `development` | Set to `production` in live environments. |
| `RESEARCHLENS_DEBUG` | bool | `false` | Must be `false` in production. If `true` under `production`, startup validation fails. |
| `RESEARCHLENS_SECRET_KEY` | string | None | Random 32+ character secret key for cryptographic tokens and session integrity. |
| `RESEARCHLENS_WORKSPACES_DIR` | path | `data/workspaces` | Persistent directory storing workspace metadata, findings, hypotheses, and runs. |
| `RESEARCHLENS_LOG_FORMAT` | string | `text` | Use `json` in production for centralized ingest (Datadog, CloudWatch, ELK). |
| `RESEARCHLENS_LOG_LEVEL` | string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `MLFLOW_TRACKING_URI` | string | `./mlruns` | Local SQLite, remote HTTP tracking server, or cloud URI. |
| `RESEARCHLENS_MAX_UPLOAD_SIZE_BYTES` | int | `104857600` | Max PDF file size (100 MB). |

---

## 5. Health Check & Observability

ResearchLens provides a comprehensive health diagnostic module: [src/researchlens/health.py](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/health.py).

### Manual CLI Invocation
```bash
python -m researchlens.health
```

Sample output:
```json
{
  "status": "HEALTHY",
  "timestamp": "2026-10-09T07:44:40Z",
  "environment": "development",
  "components": {
    "storage": {
      "status": "HEALTHY",
      "workspaces_dir": "data/workspaces",
      "writable": true,
      "free_space_mb": 269784
    },
    "nlp_runtime": {
      "status": "HEALTHY",
      "torch_available": true,
      "transformers_available": true,
      "device": "cpu"
    },
    "mlflow": {
      "status": "HEALTHY",
      "tracking_uri": "./mlruns",
      "connected": true
    },
    "docker_execution": {
      "status": "HEALTHY",
      "connected": true,
      "version": "29.8.1"
    }
  }
}
```

---

## 6. Reverse Proxy & Security Hardening (Nginx)

The reverse proxy handles TLS termination, request validation, rate limiting, and route-level authentication.

### Security Highlights:
1. **HTTP -> HTTPS Redirection**: Port 80 automatically redirects all traffic to HTTPS (port 443) with HTTP 301.
2. **TLS 1.2 & 1.3**: Weak ciphers and legacy SSL/TLS versions are completely disabled.
3. **Security Headers**:
   - `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
   - `X-Frame-Options: SAMEORIGIN`
   - `X-Content-Type-Options: nosniff`
   - `X-XSS-Protection: 1; mode=block`
   - `Referrer-Policy: strict-origin-when-cross-origin`
   - `Content-Security-Policy: default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;`
4. **Request Size Limit**: `client_max_body_size 100M;` strictly enforces the maximum upload limit at the gateway before hitting application memory.
5. **Rate Limiting**: `limit_req_zone $binary_remote_addr zone=api_limit:10m rate=50r/s;` with `burst=100 nodelay;` mitigates denial-of-service attempts.
6. **MLflow Interface Protection**: The `/mlflow/` path is protected by HTTP Basic Authentication (`auth_basic` via `nginx/.htpasswd`). Public clients cannot view model registries or experiment runs without credentials.
7. **Streamlit WebSocket Upgrade**: Proxy directives include `proxy_http_version 1.1`, `Upgrade $http_upgrade`, and `Connection $connection_upgrade` to ensure real-time bi-directional streaming stability.

### Generating TLS Certificates:
- **Production Public Domain**: Provision via Certbot / Let's Encrypt:
  ```bash
  certbot certonly --standalone -d yourdomain.com
  cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./nginx/ssl/cert.pem
  cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./nginx/ssl/key.pem
  ```
- **Staging / Local Testing**: Generate 2048-bit RSA self-signed certificates:
  ```bash
  python scripts/generate_tls_certs.py
  ```
