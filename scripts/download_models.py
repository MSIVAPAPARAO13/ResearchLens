"""Download and verify pre-trained models required by ResearchLens."""

import sys

from researchlens.config.settings import get_settings
from researchlens.preprocessing.cleaner import get_spacy_model
from researchlens.summarization.model import load_model_and_tokenizer
from researchlens.utils.logging import get_logger

logger = get_logger("download_models")


def main():
    settings = get_settings()
    logger.info(f"Target summarization model: {settings.model_name}")

    try:
        logger.info("Checking / downloading BART model and tokenizer...")
        model, tokenizer, device = load_model_and_tokenizer(settings.model_name)
        logger.info(f"Summarization model verified on {device}")
    except Exception as exc:
        logger.error(f"Failed to load BART model: {exc}")
        sys.exit(1)

    try:
        logger.info("Checking spaCy en_core_web_sm model...")
        spacy_nlp = get_spacy_model()
        if spacy_nlp:
            logger.info("spaCy en_core_web_sm model loaded successfully.")
        else:
            logger.warning("spaCy model unavailable. Install with: python -m spacy download en_core_web_sm")
    except Exception as exc:
        logger.warning(f"spaCy check encountered error: {exc}")

    try:
        logger.info(f"Checking embedding model: {settings.embedding_model_name}...")
        from researchlens.retrieval import get_embedding_model
        emb = get_embedding_model()
        dim = emb.dimension
        logger.info(f"Embedding model verified ({dim} dimensions).")
    except Exception as exc:
        logger.warning(f"Embedding model check encountered error: {exc}")

    logger.info("Model download and verification complete.")


if __name__ == "__main__":
    main()
