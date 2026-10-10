"""Academic dataset and benchmark corpus extraction with evidence provenance."""

import re
from typing import Any, List, Union

from researchlens.extraction.schema import ExtractedItem

DATASET_PATTERNS = [
    r"\b(?:imagenet(?:-1k|-22k)?|cifar-(?:10|100)|mnist|fashion-mnist|svhn|coco)\b",
    r"\b(?:glue|superglue|squad(?:\s*v?2\.0)?|triviaqa|natural questions|hotpotqa|scicite|s2orc)\b",
    r"\b(?:cnn/dailymail|xsum|gigaword|wikihow|arxiv|pubmed(?:-[0-9]+k-rct)?|ms marco)\b",
    r"\b(?:wmt(?:\s*1[0-9]|\s*2[0-9])?|iwslt|opus)\b",
    r"\b(?:conll-(?:2003|2000)|ontonotes|ptb|penn treebank)\b",
    r"\b(?:mmlu|gsm8k|humaneval|hellaswag|arc|truthfulqa)\b",
    r"\b(?:mimic-(?:iii|iv)|chexpert|cmapss)\b",
    r"\b([A-Z][a-zA-Z0-9_-]+\s+dataset)\b",
    r"\b([A-Z][a-zA-Z0-9_-]+\s+benchmark)\b",
    r"\b([A-Z][a-zA-Z0-9_-]+\s+corpus)\b",
]


def extract_dataset_items(doc: Any) -> List[ExtractedItem]:
    """Extract academic datasets and benchmark corpora with page, section, and chunk provenance."""
    items: List[ExtractedItem] = []
    seen = set()

    chunks = getattr(doc, "chunks", []) if doc else []
    if not chunks:
        text = doc.full_text if hasattr(doc, "full_text") else str(doc or "")
        if not text.strip():
            return [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

        for pat in DATASET_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                val = m.group(0).strip()
                if len(val) > 2 and val.lower() not in seen:
                    seen.add(val.lower())
                    items.append(
                        ExtractedItem(
                            value=val,
                            source_text=val,
                            page_number=1,
                            section_name="experiments",
                            confidence=0.85,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )
        return items if items else [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    for chunk in chunks:
        sec = getattr(chunk, "section_name", "").lower()
        # Prefer Experiments, Datasets, Methodology
        text = getattr(chunk, "text", "")
        for pat in DATASET_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                val = m.group(0).strip()
                if len(val) > 2 and val.lower() not in seen:
                    seen.add(val.lower())

                    # Sentential context
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    ctx = text
                    for s in sentences:
                        if val.lower() in s.lower():
                            ctx = s.strip()
                            break

                    conf = 0.92 if sec in ("experiments", "dataset", "datasets", "methodology") else 0.82
                    items.append(
                        ExtractedItem(
                            value=val,
                            source_text=ctx,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "experiments"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=conf,
                            confidence_level="High" if conf > 0.85 else "Medium",
                            extraction_method="pattern",
                        )
                    )

    if not items:
        return [
            ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")
        ]

    return items[:8]


def extract_datasets(text_or_doc: Union[str, Any]) -> List[str]:
    """Backward-compatible string list extractor for datasets."""
    if isinstance(text_or_doc, str):
        detected = set()
        for pat in DATASET_PATTERNS:
            matches = re.finditer(pat, text_or_doc, flags=re.IGNORECASE)
            for m in matches:
                val = m.group(0).strip()
                if len(val) > 2:
                    detected.add(val)
        return sorted(list(detected)) if detected else ["Not detected"]

    items = extract_dataset_items(text_or_doc)
    vals = [item.value for item in items if item.value != "Not detected"]
    return sorted(list(set(vals))) if vals else ["Not detected"]
