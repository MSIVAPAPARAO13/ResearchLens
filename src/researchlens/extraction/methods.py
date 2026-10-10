"""Scientific methods, machine learning architectures, models, and algorithms extraction with provenance."""

import re
from typing import Any, List, Tuple

from researchlens.extraction.schema import ExtractedItem

MODEL_PATTERNS = [
    r"\b(?:resnet(?:-[0-9]+)?|transformer|bert|roberta|bart|t5|gpt(?:-[0-9a-z]+)?|llama(?:-[0-9a-z]+)?|mistral|vgg(?:-[0-9]+)?|alexnet|densenet|efficientnet(?:-[b0-7]+)?)\b",
    r"\b(?:yolo(?:v[0-9]+)?|mask-rcnn|faster-rcnn|unet|u-net)\b",
    r"\b(?:support vector machine|svm|random forest|xgboost|gradient boosting|decision tree)\b",
    r"\b(?:lstm|gru|cnn|rnn|autoencoder|vae|gan|diffusion model)\b",
]

METHOD_PATTERNS = [
    r"\b(?:qualitative analysis|semi-structured interview|thematic analysis|grounded theory)\b",
    r"\b(?:regression analysis|anova|chi-square|structural equation modeling|meta-analysis)\b",
    r"\b(?:fine-tuning|pre-training|prompt tuning|lora|few-shot learning|zero-shot learning|transfer learning)\b",
    r"\b(?:contrastive learning|reinforcement learning|self-supervised learning|federated learning)\b",
]

ALGORITHM_PATTERNS = [
    r"\b(?:adamw|adam optimizer|sgd|stochastic gradient descent|rmsprop|backpropagation)\b",
    r"\b(?:attention mechanism|multi-head attention|self-attention|rotary position embeddings?|flashattention)\b",
    r"\b(?:beam search|greedy decoding|nucleus sampling|top-k sampling)\b",
    r"\b(?:k-means|hierarchical clustering|pca|principal component analysis|t-sne|umap)\b",
]


def _extract_items_by_pattern(
    doc: Any, patterns: List[str], default_category: str
) -> List[ExtractedItem]:
    """Helper to extract matching pattern terms from document chunks with provenance."""
    items: List[ExtractedItem] = []
    seen = set()

    chunks = getattr(doc, "chunks", []) if doc else []
    if not chunks and hasattr(doc, "full_text"):
        # Fallback if doc is a string or doc without chunks
        text = doc.full_text if hasattr(doc, "full_text") else str(doc)
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                val = m.group(0).strip().title()
                if val.lower() not in seen:
                    seen.add(val.lower())
                    items.append(
                        ExtractedItem(
                            value=val,
                            source_text=val,
                            page_number=1,
                            section_name=default_category,
                            confidence=0.85,
                            confidence_level="Medium",
                            extraction_method="pattern",
                        )
                    )
        return items if items else [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low")]

    for chunk in chunks:
        text = getattr(chunk, "text", "")
        for pat in patterns:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                val = m.group(0).strip()
                # Proper title casing preserving known acronyms
                val_clean = val.title() if len(val) > 4 else val.upper()
                if val_clean.lower() not in seen:
                    seen.add(val_clean.lower())

                    # Context sentence
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    ctx = text
                    for s in sentences:
                        if val.lower() in s.lower():
                            ctx = s.strip()
                            break

                    items.append(
                        ExtractedItem(
                            value=val_clean,
                            source_text=ctx,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", default_category),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.90,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )

    if not items:
        return [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ]

    return items[:8]


def extract_models_methods_algorithms(
    doc: Any,
) -> Tuple[List[ExtractedItem], List[ExtractedItem], List[ExtractedItem]]:
    """Extract models, methods, and algorithms with exact page, section, and chunk provenance."""
    models = _extract_items_by_pattern(doc, MODEL_PATTERNS, "methodology")
    methods = _extract_items_by_pattern(doc, METHOD_PATTERNS, "methodology")
    algorithms = _extract_items_by_pattern(doc, ALGORITHM_PATTERNS, "methodology")

    return models, methods, algorithms


def extract_methods(text_or_doc: Any) -> List[str]:
    """Backward-compatible string list extractor for methods."""
    if isinstance(text_or_doc, str):
        detected = set()
        for pat in MODEL_PATTERNS + METHOD_PATTERNS + ALGORITHM_PATTERNS:
            matches = re.findall(pat, text_or_doc, flags=re.IGNORECASE)
            for m in matches:
                detected.add(m.strip().title())
        return sorted(list(detected)) if detected else ["Not detected"]

    models, methods, algs = extract_models_methods_algorithms(text_or_doc)
    all_vals = [
        item.value
        for item in (models + methods + algs)
        if item.value != "Not detected"
    ]
    return sorted(list(set(all_vals))) if all_vals else ["Not detected"]
