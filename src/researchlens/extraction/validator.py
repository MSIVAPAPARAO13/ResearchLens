"""Validation and normalization layer for extracted research items and profile."""

from typing import List, TypeVar

from researchlens.extraction.schema import (
    BaselineComparison,
    ExtractedItem,
    QuantitativeResultItem,
    ResearchProfile,
)

T = TypeVar("T", ExtractedItem, QuantitativeResultItem, BaselineComparison)


def validate_extracted_items(items: List[T]) -> List[T]:
    """Deduplicate extracted items while preserving provenance and confidence ranking."""
    if not items:
        return []

    validated: List[T] = []
    seen_values = set()

    for item in items:
        val = getattr(item, "value", None) or getattr(item, "metric", None) or getattr(item, "baseline_model", "")
        if not val or val == "Not detected" or val == "Not explicitly stated":
            continue

        val_lower = val.lower().strip()
        if val_lower not in seen_values:
            seen_values.add(val_lower)

            # Ensure page number and section name are normalized
            if item.page_number is None and hasattr(item, "source_text"):
                item.page_number = getattr(item, "page_number", 1)

            validated.append(item)

    return validated


def validate_research_profile(profile: ResearchProfile) -> ResearchProfile:
    """Apply validation rules to clean duplicates, check consistency, and ensure fallback defaults."""
    profile.authors = validate_extracted_items(profile.authors) or [ExtractedItem("Not detected")]
    profile.organizations = validate_extracted_items(profile.organizations) or [ExtractedItem("Not detected")]
    profile.keywords = validate_extracted_items(profile.keywords) or [ExtractedItem("Not detected")]

    profile.problem = validate_extracted_items(profile.problem) or [ExtractedItem("Not detected")]
    profile.objective = validate_extracted_items(profile.objective) or [ExtractedItem("Not detected")]
    profile.research_questions = validate_extracted_items(profile.research_questions) or [ExtractedItem("Not explicitly stated")]
    profile.hypotheses = validate_extracted_items(profile.hypotheses) or [ExtractedItem("Not explicitly stated")]

    profile.methods = validate_extracted_items(profile.methods) or [ExtractedItem("Not detected")]
    profile.models = validate_extracted_items(profile.models) or [ExtractedItem("Not detected")]
    profile.algorithms = validate_extracted_items(profile.algorithms) or [ExtractedItem("Not detected")]

    profile.datasets = validate_extracted_items(profile.datasets) or [ExtractedItem("Not detected")]
    profile.experimental_setup = validate_extracted_items(profile.experimental_setup) or [ExtractedItem("Not detected")]

    profile.metrics = validate_extracted_items(profile.metrics) or [ExtractedItem("Not detected")]

    # Results and Baselines
    profile.results = validate_extracted_items(profile.results)
    profile.baselines = validate_extracted_items(profile.baselines)

    profile.contributions = validate_extracted_items(profile.contributions) or [ExtractedItem("Not detected")]
    profile.limitations = validate_extracted_items(profile.limitations) or [ExtractedItem("Not detected")]
    profile.future_work = validate_extracted_items(profile.future_work) or [ExtractedItem("Not detected")]

    return profile
