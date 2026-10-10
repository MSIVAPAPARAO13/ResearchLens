# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:/app" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install minimal OS dependencies for healthchecks and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 1000 researcher && \
    useradd -u 1000 -g researcher -m -s /bin/bash researcher

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt pyproject.toml /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy source code and necessary asset directories
COPY src/ /app/src/
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY evaluation/baselines/ /app/evaluation/baselines/

# Ensure workspace and log directories exist and are owned by researcher
RUN mkdir -p /app/data/workspaces /app/mlruns /app/runs && \
    chown -R researcher:researcher /app

USER researcher

EXPOSE 8501

# Container healthcheck using ResearchLens health diagnostic CLI
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -m researchlens.health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
