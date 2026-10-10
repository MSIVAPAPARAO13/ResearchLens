"""Extraction pipeline orchestrator coordinating section-aware scientific extractors."""

from typing import Any

from researchlens.extraction.contributions import extract_contributions
from researchlens.extraction.datasets import extract_dataset_items
from researchlens.extraction.experimental_setup import extract_experimental_setup
from researchlens.extraction.gaps_limitations import extract_limitations_and_future_work
from researchlens.extraction.keywords import extract_keywords
from researchlens.extraction.metadata import (
    extract_authors_and_orgs,
    extract_publication_year,
    extract_title,
)
from researchlens.extraction.methods import extract_models_methods_algorithms
from researchlens.extraction.metrics import extract_metrics_results_baselines
from researchlens.extraction.problem_objective import (
    extract_problem_and_objective,
    extract_questions_and_hypotheses,
)
from researchlens.extraction.schema import ExtractedItem, ResearchProfile
from researchlens.extraction.validator import validate_research_profile
from researchlens.utils.logging import get_logger

logger = get_logger("extraction_pipeline")


class ExtractionPipeline:
    """Orchestrator for scientific information extraction across structured document sections."""

    def extract_profile(self, doc: Any) -> ResearchProfile:
        """Process structured document and return a validated ResearchProfile with provenance."""
        if not doc:
            return ResearchProfile()

        doc_id = getattr(doc, "document_id", "doc_unknown")
        filename = getattr(doc, "filename", "paper.pdf")
        logger.info(f"Executing scientific information extraction pipeline for document '{filename}'")

        # 1. Metadata
        title_item = extract_title(doc)
        authors, orgs = extract_authors_and_orgs(doc)
        year = extract_publication_year(doc)

        # Keywords
        raw_keywords = extract_keywords(getattr(doc, "full_text", ""))
        kw_items = [
            ExtractedItem(
                value=kw,
                source_text=kw,
                page_number=1,
                section_name="keywords",
                confidence=0.85,
                extraction_method="spacy_noun_chunks",
            )
            for kw in raw_keywords
            if kw != "Not detected"
        ]
        if not kw_items:
            kw_items = [ExtractedItem("Not detected", confidence=0.0, confidence_level="Low")]

        # 2. Problem, Objective, Questions, Hypotheses
        problems, objectives = extract_problem_and_objective(doc)
        rqs, hypotheses = extract_questions_and_hypotheses(doc)

        # 3. Methods, Models, Algorithms
        models, methods, algorithms = extract_models_methods_algorithms(doc)

        # 4. Datasets
        datasets = extract_dataset_items(doc)

        # 5. Experimental Setup
        setup_items = extract_experimental_setup(doc)

        # 6. Metrics, Results, Baselines
        metrics, results, baselines = extract_metrics_results_baselines(doc)

        # 7. Contributions
        contributions = extract_contributions(doc)

        # 8. Limitations & Future Work
        limitations, future_work = extract_limitations_and_future_work(doc)

        # Construct unvalidated profile
        profile = ResearchProfile(
            document_id=doc_id,
            title=title_item.value,
            authors=authors,
            organizations=orgs,
            year=year,
            keywords=kw_items,
            problem=problems,
            objective=objectives,
            research_questions=rqs,
            hypotheses=hypotheses,
            methods=methods,
            models=models,
            algorithms=algorithms,
            datasets=datasets,
            experimental_setup=setup_items,
            metrics=metrics,
            results=results,
            baselines=baselines,
            contributions=contributions,
            limitations=limitations,
            future_work=future_work,
        )

        # 9. Validate and normalize profile
        validated_profile = validate_research_profile(profile)
        logger.info(f"Extraction pipeline completed successfully for '{filename}'")
        return validated_profile


def extract_research_profile(doc: Any) -> ResearchProfile:
    """Convenience function to execute extraction pipeline on a document."""
    pipeline = ExtractionPipeline()
    return pipeline.extract_profile(doc)
