"""Metric, quantitative result, and baseline comparative result extraction."""

import re
from typing import Any, Dict, List, Tuple, Union

from researchlens.extraction.schema import (
    BaselineComparison,
    ExtractedItem,
    QuantitativeResultItem,
)

METRIC_NAME_PATTERNS = [
    r"\b(?:accuracy|acc|top-1 accuracy|top-5 accuracy)\b",
    r"\b(?:f1(?:-score)?|macro-f1|micro-f1|weighted-f1)\b",
    r"\b(?:bleu(?:-[1-4])?|rouge-(?:1|2|l)|meteor|cider)\b",
    r"\b(?:precision|recall|map|map@[0-9]+|mrr|ndcg(?:@[0-9]+)?|hit rate(?:@[0-9]+)?)\b",
    r"\b(?:auc|roc-auc|area under curve)\b",
    r"\b(?:rmse|mae|mse|nrmse|perplexity|ppl)\b",
]

QUANTITATIVE_RESULT_PATTERNS = [
    r"\b(accuracy|acc|f1(?:-score)?|macro-f1|micro-f1|precision|recall|bleu|rouge|rmse|mae|mse|mrr|ndcg)\b\s*(?:of|is|was|reached|achieved|:|=)?\s*([0-9]+(?:\.[0-9]+)?%?)",
    r"\b(achieved|outperformed|scored|yielded)\s*(?:a|an)?\s*(?:test|validation|top-1)?\s*(accuracy|f1|rmse|mae|mrr|score)?\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?%?)",
]

BASELINE_COMPARISON_PATTERNS = [
    r"\b(?:outperformed|surpassed|exceeded)\s+([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)?)\s+by\s+([0-9]+(?:\.[0-9]+)?%?)",
    r"\bcompared\s+(?:with|to)\s+([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)?)\b",
]


def extract_metrics_results_baselines(
    doc: Any,
) -> Tuple[List[ExtractedItem], List[QuantitativeResultItem], List[BaselineComparison]]:
    """Extract metrics, quantitative results, and comparative baselines with evidence provenance."""
    metrics: List[ExtractedItem] = []
    results: List[QuantitativeResultItem] = []
    baselines: List[BaselineComparison] = []

    seen_metrics = set()
    seen_results = set()

    chunks = getattr(doc, "chunks", []) if doc else []

    if not chunks:
        text = doc.full_text if hasattr(doc, "full_text") else str(doc or "")
        if not text.strip():
            return (
                [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low")],
                [],
                [],
            )
        # Process raw text fallback
        for pat in METRIC_NAME_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                val = m.group(0).strip().title()
                if val.lower() not in seen_metrics:
                    seen_metrics.add(val.lower())
                    metrics.append(ExtractedItem(value=val, source_text=val, page_number=1, section_name="results"))
        return (
            metrics
            if metrics
            else [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]
        ), [], []

    for chunk in chunks:
        text = getattr(chunk, "text", "")

        # 1. Metrics Name Extraction
        for pat in METRIC_NAME_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                val = m.group(0).strip()
                val_norm = val.title() if len(val) > 4 else val.upper()
                if val_norm.lower() not in seen_metrics:
                    seen_metrics.add(val_norm.lower())
                    metrics.append(
                        ExtractedItem(
                            value=val_norm,
                            source_text=val,
                            page_number=getattr(chunk, "page_number", None),
                            section_name=getattr(chunk, "section_name", "results"),
                            chunk_id=getattr(chunk, "chunk_id", None),
                            confidence=0.92,
                            confidence_level="High",
                            extraction_method="pattern",
                        )
                    )

        # 2. Quantitative Result Extraction
        for pat in QUANTITATIVE_RESULT_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                groups = [g for g in m.groups() if g is not None]
                if len(groups) >= 2:
                    m_name = groups[0].strip().title()
                    m_val = groups[1].strip()

                    res_key = f"{m_name}:{m_val}"
                    if res_key.lower() not in seen_results:
                        seen_results.add(res_key.lower())

                        # Sentential context
                        sentences = re.split(r"(?<=[.!?])\s+", text)
                        ctx = text
                        for s in sentences:
                            if m_val in s:
                                ctx = s.strip()
                                break

                        unit = "%" if "%" in m_val else None
                        results.append(
                            QuantitativeResultItem(
                                metric=m_name,
                                value=m_val,
                                unit=unit,
                                source_text=ctx,
                                page_number=getattr(chunk, "page_number", None),
                                section_name=getattr(chunk, "section_name", "results"),
                                chunk_id=getattr(chunk, "chunk_id", None),
                                confidence=0.94,
                                extraction_method="pattern",
                            )
                        )

        # 3. Baseline Comparisons
        for pat in BASELINE_COMPARISON_PATTERNS:
            matches = re.finditer(pat, text, flags=re.IGNORECASE)
            for m in matches:
                groups = m.groups()
                b_model = groups[0].strip() if groups else "Baseline"
                diff = groups[1].strip() if len(groups) > 1 else None

                sentences = re.split(r"(?<=[.!?])\s+", text)
                ctx = text
                for s in sentences:
                    if b_model.lower() in s.lower():
                        ctx = s.strip()
                        break

                baselines.append(
                    BaselineComparison(
                        baseline_model=b_model,
                        difference=diff,
                        source_text=ctx,
                        page_number=getattr(chunk, "page_number", None),
                        section_name=getattr(chunk, "section_name", "results"),
                        chunk_id=getattr(chunk, "chunk_id", None),
                    )
                )

    if not metrics:
        metrics = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low", extraction_method="fallback")]

    return metrics[:10], results[:10], baselines[:5]


def extract_metrics(text_or_doc: Union[str, Any]) -> Dict[str, List[str]]:
    """Backward-compatible metric extraction dictionary."""
    if isinstance(text_or_doc, str):
        res: Dict[str, List[str]] = {}
        for metric_name, pat in {
            "Accuracy": r"\b(?:accuracy|acc)\b\s*(?:of|is|was|reached|achieved|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?%?)",
            "F1-Score": r"\b(?:f1(?:-score)?|macro-f1|micro-f1)\b\s*(?:of|is|was|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?%?)",
            "BLEU": r"\bbleu(?:-[1-4])?\b\s*(?:of|is|was|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?)",
            "ROUGE": r"\brouge-(?:1|2|l)\b\s*(?:of|is|was|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?)",
            "Precision": r"\bprecision\b\s*(?:of|is|was|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?%?)",
            "Recall": r"\brecall\b\s*(?:of|is|was|:)?\s*([0-9]{1,2}(?:\.[0-9]{1,3})?%?)",
            "RMSE": r"\brmse\b\s*(?:of|is|was|:)?\s*([0-9]+(?:\.[0-9]{1,4})?)",
        }.items():
            matches = re.finditer(pat, text_or_doc, flags=re.IGNORECASE)
            found = [m.group(1).strip() for m in matches if m.group(1)]
            if found:
                res[metric_name] = found[:5]
        return res if res else {"result": ["Not detected"]}

    metrics, results, _ = extract_metrics_results_baselines(text_or_doc)
    res_dict: Dict[str, List[str]] = {}
    for r in results:
        if r.metric not in res_dict:
            res_dict[r.metric] = []
        if r.value not in res_dict[r.metric]:
            res_dict[r.metric].append(r.value)

    if not res_dict:
        m_vals = [m.value for m in metrics if m.value != "Not detected"]
        return {"metrics": m_vals} if m_vals else {"result": ["Not detected"]}

    return res_dict
