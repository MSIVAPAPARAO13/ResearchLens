"""Lazy model management and device resolution for summarization transformers."""

from typing import Optional, Tuple

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from researchlens.config.settings import get_settings
from researchlens.utils.logging import get_logger

logger = get_logger("model")

_TOKENIZER_CACHE: Optional[BartTokenizer] = None
_MODEL_CACHE: Optional[BartForConditionalGeneration] = None


def get_device() -> torch.device:
    """Resolve compute device based on configuration and hardware availability."""
    settings = get_settings()
    if settings.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_tokenizer(model_name: Optional[str] = None) -> BartTokenizer:
    """Lazily load and cache the BART tokenizer."""
    global _TOKENIZER_CACHE
    target_model = model_name or get_settings().model_name
    if _TOKENIZER_CACHE is None or getattr(_TOKENIZER_CACHE, "name_or_path", None) != target_model:
        logger.info(f"Loading tokenizer: {target_model}")
        _TOKENIZER_CACHE = BartTokenizer.from_pretrained(target_model)
    return _TOKENIZER_CACHE


def get_model(model_name: Optional[str] = None) -> BartForConditionalGeneration:
    """Lazily load, configure, and cache the BART conditional generation model."""
    global _MODEL_CACHE
    target_model = model_name or get_settings().model_name
    device = get_device()

    if _MODEL_CACHE is None or getattr(_MODEL_CACHE, "name_or_path", None) != target_model:
        logger.info(f"Loading summarization model: {target_model} onto {device}")
        model = BartForConditionalGeneration.from_pretrained(target_model)
        model.to(device)
        model.eval()
        _MODEL_CACHE = model

    return _MODEL_CACHE


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
) -> Tuple[BartForConditionalGeneration, BartTokenizer, torch.device]:
    """Retrieve model, tokenizer, and target device in a single call."""
    tok = get_tokenizer(model_name)
    mod = get_model(model_name)
    dev = get_device()
    return mod, tok, dev
