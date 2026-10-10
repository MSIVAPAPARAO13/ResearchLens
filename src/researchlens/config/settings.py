"""Centralized settings and configuration management for ResearchLens."""

import os
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application-wide configuration parameters."""

    # Model & Tokenizer settings
    model_name: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_MODEL_NAME", "facebook/bart-large-cnn"),
        description="Pretrained summarization model identifier from Hugging Face hub",
    )
    max_input_tokens: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_MAX_INPUT_TOKENS", "1024")),
        description="Maximum sequence length accepted by the transformer model",
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_CHUNK_SIZE", "900")),
        description="Target maximum token count per text chunk",
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_CHUNK_OVERLAP", "100")),
        description="Number of overlapping tokens between successive chunks",
    )
    default_summary_length: int = Field(
        default=160,
        description="Default maximum target tokens for generated summaries",
    )
    min_summary_length: int = Field(
        default=60,
        description="Default minimum target tokens for generated summaries",
    )
    device: str = Field(
        default_factory=lambda: (
            os.getenv("RESEARCHLENS_DEVICE")
            if os.getenv("RESEARCHLENS_DEVICE") in ("cpu", "cuda")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        description="Execution device: 'cuda' or 'cpu'",
    )

    # Ingestion & Upload limits
    max_upload_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum upload payload size in bytes (50MB)",
    )
    supported_file_types: List[str] = Field(
        default_factory=lambda: [".pdf"],
        description="Supported academic document file extensions",
    )
    min_section_length: int = Field(
        default=100,
        description="Minimum character length required to retain an extracted section",
    )

    # Retrieval & RAG configuration
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        description="Pretrained sentence embedding model identifier",
    )
    vector_store_type: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_VECTOR_STORE_TYPE", "faiss"),
        description="Vector storage backend type: 'faiss' or 'in_memory'",
    )
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_TOP_K", "5")),
        description="Default number of top relevant chunks to retrieve",
    )
    similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("RESEARCHLENS_SIMILARITY_THRESHOLD", "0.25")),
        description="Minimum cosine similarity score required for relevant evidence chunks",
    )
    max_context_chunks: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_MAX_CONTEXT_CHUNKS", "5")),
        description="Maximum number of context chunks passed to question answerer",
    )

    # Phase 3: Hybrid Retrieval & Cross-Encoder Reranking
    reranker_model_name: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        description="Pretrained cross-encoder reranker model identifier",
    )
    dense_candidate_k: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_DENSE_CANDIDATE_K", "10")),
        description="Candidate pool size for first-stage dense retrieval",
    )
    bm25_candidate_k: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_BM25_CANDIDATE_K", "10")),
        description="Candidate pool size for first-stage BM25 lexical retrieval",
    )
    rerank_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RESEARCHLENS_RERANK_TOP_K", "5")),
        description="Final number of top evidence chunks to return after cross-encoder reranking",
    )
    dense_weight: float = Field(
        default_factory=lambda: float(os.getenv("RESEARCHLENS_DENSE_WEIGHT", "0.6")),
        description="Weight assigned to normalized dense score in hybrid fusion (sum with bm25_weight = 1.0)",
    )
    bm25_weight: float = Field(
        default_factory=lambda: float(os.getenv("RESEARCHLENS_BM25_WEIGHT", "0.4")),
        description="Weight assigned to normalized BM25 score in hybrid fusion (sum with dense_weight = 1.0)",
    )

    # Environment & Deployment Configuration (Phase 16)
    env: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_ENV", "development").lower(),
        description="Application runtime environment: 'development', 'production', or 'test'",
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_DEBUG", "false").lower() in ("true", "1", "yes"),
        description="Debug mode toggle (must be disabled in production)",
    )
    api_secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_SECRET_KEY"),
        description="API secret key for authenticated endpoints (required in production if auth enabled)",
    )
    workspaces_dir: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_WORKSPACES_DIR", "data/workspaces"),
        description="Root storage directory for workspace persistence",
    )
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        description="MLflow tracking URI (local directory or remote HTTP/SQL server)",
    )

    # Logging
    log_level: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_LOG_LEVEL", "INFO"),
        description="Application logging level",
    )
    log_format: str = Field(
        default_factory=lambda: os.getenv("RESEARCHLENS_LOG_FORMAT", "text"),
        description="Logging format: 'text' or 'json'",
    )

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.env == "production"

    def validate_production_configuration(self) -> List[str]:
        """Validate production security and configuration constraints, returning any violations."""
        issues: List[str] = []
        if self.is_production:
            if self.debug:
                issues.append("Debug mode is enabled in production (RESEARCHLENS_DEBUG=true).")
            if self.max_upload_size_bytes > 100 * 1024 * 1024:
                issues.append(f"Upload size limit ({self.max_upload_size_bytes} bytes) exceeds 100MB production safety threshold.")
            if self.chunk_size > 2048:
                issues.append(f"Chunk size ({self.chunk_size}) exceeds recommended transformer context threshold.")
            if self.chunk_size <= self.chunk_overlap:
                issues.append(f"Chunk size ({self.chunk_size}) must be strictly greater than chunk overlap ({self.chunk_overlap}).")
            if self.api_secret_key and len(self.api_secret_key) < 16:
                issues.append("Configured API secret key is too short (< 16 characters).")
            if os.getenv("RESEARCHLENS_AUTH_REQUIRED", "false").lower() in ("true", "1") and not self.api_secret_key:
                issues.append("Authentication is enforced (RESEARCHLENS_AUTH_REQUIRED=true) but RESEARCHLENS_SECRET_KEY is not set.")
        return issues


    # Canonical Academic Section Patterns
    section_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "abstract": r"\babstract\b",
            "introduction": r"\b(?:1[\.\s]*)?introduction\b",
            "literature review": r"\b(?:2[\.\s]*)?(?:literature\s+review|related\s+work|background)\b",
            "methodology": r"\b(?:3[\.\s]*)?(?:methodology|methods|field\s+research\s+procedures|materials\s+and\s+methods|experimental\s+setup|system\s+architecture)\b",
            "results": r"\b(?:4[\.\s]*)?(?:results|findings|presentation\s+of\s+research\s+results|experiments\s+and\s+results)\b",
            "discussion": r"\b(?:5[\.\s]*)?(?:discussion|results\s+and\s+discussion)\b",
            "summary": r"\b(?:summary|summary\s+of\s+results)\b",
            "ethics": r"\b(?:ethical\s+considerations|ethics\s+statement)\b",
            "conclusion": r"\b(?:conclusion|conclusions|concluding\s+remarks)\b",
            "references": r"\b(?:references|bibliography)\b",
        }
    )

    # Section-specific summary length configurations
    section_summary_config: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "abstract": {"max": 120, "min": 50},
            "introduction": {"max": 160, "min": 80},
            "literature review": {"max": 200, "min": 100},
            "methodology": {"max": 240, "min": 120},
            "results": {"max": 240, "min": 120},
            "discussion": {"max": 200, "min": 100},
            "summary": {"max": 160, "min": 80},
            "ethics": {"max": 120, "min": 40},
            "conclusion": {"max": 160, "min": 80},
            "references": {"max": 100, "min": 40},
            "unknown": {"max": 160, "min": 60},
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached global settings instance."""
    return Settings()
