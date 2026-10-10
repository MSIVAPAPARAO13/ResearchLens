"""Evidence-driven Research Gap analysis engine operating on structured ResearchProfile."""

import logging
from typing import Any, List, Optional, Tuple

from researchlens.analysis.gap_scorer import calculate_gap_score
from researchlens.analysis.research_gap_schema import ResearchGap, ResearchHypothesis
from researchlens.extraction.schema import ResearchProfile

logger = logging.getLogger("researchlens.gap_engine")


class ResearchGapEngine:
    """Engine analyzing ResearchProfile data to extract and classify evidence-backed research gaps."""

    def __init__(self, ablation_mode: str = "full"):
        """Initialize Gap Engine with optional ablation mode.

        Ablation modes:
            - 'limitation_only': Process explicit limitations only.
            - 'limitation_future': Process explicit limitations + future work.
            - 'limitation_future_methodology': Process limitations + future work + methodology analysis.
            - 'full': Process all evidence sources (limitations, future work, dataset, baseline, evaluation gaps).
        """
        self.ablation_mode = ablation_mode

    def analyze_profile(
        self, profile: ResearchProfile, doc: Optional[Any] = None
    ) -> Tuple[List[ResearchGap], List[ResearchHypothesis]]:
        """Extract evidence-backed research gaps and potential hypotheses from a ResearchProfile."""
        logger.info(f"Analyzing research gaps for profile '{profile.title}' (Ablation mode: {self.ablation_mode})")

        gaps: List[ResearchGap] = []

        # 1. Explicit Limitations
        lim_gaps = self._extract_explicit_limitation_gaps(profile)
        gaps.extend(lim_gaps)

        # 2. Explicit Future Work
        if self.ablation_mode in ("limitation_future", "limitation_future_methodology", "full"):
            fw_gaps = self._extract_explicit_future_work_gaps(profile)
            gaps.extend(fw_gaps)

        # 3. Methodology & Dataset Generalization Gaps
        if self.ablation_mode in ("limitation_future_methodology", "full"):
            ds_gaps = self._extract_dataset_generalization_gaps(profile)
            gaps.extend(ds_gaps)

        # 4. Baseline & Evaluation Gaps
        if self.ablation_mode == "full":
            base_gaps = self._extract_baseline_gaps(profile)
            gaps.extend(base_gaps)

        # Score and Calibrate Confidence for All Gaps
        calibrated_gaps: List[ResearchGap] = []
        for g in gaps:
            score, level = calculate_gap_score(g)
            g.confidence_score = score
            g.confidence_level = level
            calibrated_gaps.append(g)

        # Refusal check if no valid evidence-grounded gap exists
        if not calibrated_gaps or all(g.confidence_score == 0 for g in calibrated_gaps):
            refusal_gap = ResearchGap(
                gap_id="gap_refusal",
                title="Insufficient Evidence",
                category="OTHER",
                gap_type="REFRUSAL",
                statement="Insufficient evidence to identify a reliable research gap.",
                explicit_or_inferred="INFERRED",
                confidence_level="Low",
                confidence_score=0.0,
                rationale="The analyzed paper evidence does not contain sufficient explicit limitations or detectable methodology gaps.",
                potential_direction="Upload a complete research paper with methodology, evaluation, and limitation sections.",
            )
            return [refusal_gap], []

        # Deduplicate and sort gaps by confidence score
        unique_gaps = self._deduplicate_gaps(calibrated_gaps)
        sorted_gaps = sorted(unique_gaps, key=lambda x: x.confidence_score, reverse=True)

        # Generate Hypotheses for Credible Gaps
        hypotheses = self._generate_hypotheses(sorted_gaps, profile)

        return sorted_gaps, hypotheses

    def _extract_explicit_limitation_gaps(self, profile: ResearchProfile) -> List[ResearchGap]:
        gaps = []
        idx = 1
        for item in profile.limitations:
            if item.value in ("Not detected", "Not explicitly stated") or len(item.value.strip()) < 10:
                continue

            category = self._categorize_text(item.value)
            affected_model = profile.models[0].value if profile.models and profile.models[0].value != "Not detected" else ""
            affected_ds = profile.datasets[0].value if profile.datasets and profile.datasets[0].value != "Not detected" else ""

            g = ResearchGap(
                gap_id=f"gap_lim_{idx}",
                title=f"Explicit Limitation: {item.value[:50]}...",
                category=category,
                gap_type="EXPLICIT_LIMITATION",
                statement=f"Explicit limitation identified: {item.value}",
                supporting_evidence=[item],
                source_pages=[item.page_number] if item.page_number else [],
                source_sections=[item.section_name] if item.section_name else ["discussion"],
                source_chunks=[item.chunk_id] if item.chunk_id else [],
                explicit_or_inferred="EXPLICIT",
                rationale=f"The authors explicitly highlight this constraint in section '{item.section_name}'.",
                potential_direction=f"Investigate architectural or algorithmic modifications to overcome {item.value.lower()}.",
                affected_method=affected_model,
                affected_dataset=affected_ds,
                severity="High" if category in ("METHODOLOGY", "GENERALIZATION") else "Medium",
            )
            gaps.append(g)
            idx += 1
        return gaps

    def _extract_explicit_future_work_gaps(self, profile: ResearchProfile) -> List[ResearchGap]:
        gaps = []
        idx = 1
        for item in profile.future_work:
            if item.value in ("Not detected", "Not explicitly stated") or len(item.value.strip()) < 10:
                continue

            category = self._categorize_text(item.value)
            affected_model = profile.models[0].value if profile.models and profile.models[0].value != "Not detected" else ""
            affected_ds = profile.datasets[0].value if profile.datasets and profile.datasets[0].value != "Not detected" else ""

            g = ResearchGap(
                gap_id=f"gap_fw_{idx}",
                title=f"Explicit Future Work: {item.value[:50]}...",
                category=category,
                gap_type="EXPLICIT_FUTURE_WORK",
                statement=f"Unresolved area identified by authors: {item.value}",
                supporting_evidence=[item],
                source_pages=[item.page_number] if item.page_number else [],
                source_sections=[item.section_name] if item.section_name else ["conclusion"],
                source_chunks=[item.chunk_id] if item.chunk_id else [],
                explicit_or_inferred="EXPLICIT",
                rationale=f"Explicitly proposed by the authors in section '{item.section_name}' as an open research direction.",
                potential_direction=f"Extend evaluation and methodology to address {item.value.lower()}.",
                affected_method=affected_model,
                affected_dataset=affected_ds,
                severity="Medium",
            )
            gaps.append(g)
            idx += 1
        return gaps

    def _extract_dataset_generalization_gaps(self, profile: ResearchProfile) -> List[ResearchGap]:
        gaps = []
        valid_datasets = [d for d in profile.datasets if d.value not in ("Not detected", "Not explicitly stated")]

        if len(valid_datasets) == 1:
            ds_item = valid_datasets[0]
            affected_model = profile.models[0].value if profile.models and profile.models[0].value != "Not detected" else "the proposed approach"

            g = ResearchGap(
                gap_id="gap_ds_single",
                title=f"Single-Dataset Evaluation ({ds_item.value})",
                category="GENERALIZATION",
                gap_type="UNDEREXPLORED_DATASET",
                statement=f"Evaluation is limited to a single dataset ({ds_item.value}). Cross-dataset generalization remains insufficiently validated.",
                supporting_evidence=[ds_item],
                source_pages=[ds_item.page_number] if ds_item.page_number else [],
                source_sections=[ds_item.section_name] if ds_item.section_name else ["experiments"],
                source_chunks=[ds_item.chunk_id] if ds_item.chunk_id else [],
                explicit_or_inferred="INFERRED",
                rationale=f"The analyzed evidence details evaluation solely on '{ds_item.value}'. Robustness across distinct domain distributions is unconfirmed.",
                potential_direction=f"Evaluate {affected_model} across multiple independent domain-specific benchmark datasets.",
                affected_method=affected_model,
                affected_dataset=ds_item.value,
                severity="High",
            )
            gaps.append(g)
        return gaps

    def _extract_baseline_gaps(self, profile: ResearchProfile) -> List[ResearchGap]:
        gaps = []
        valid_baselines = [b for b in profile.baselines if b.baseline_model not in ("Not detected", "Not explicitly stated")]

        if not valid_baselines and profile.results:
            first_res = profile.results[0]
            g = ResearchGap(
                gap_id="gap_base_missing",
                title="Limited Comparative Baseline Evaluation",
                category="BASELINE",
                gap_type="MISSING_COMPARISON",
                statement="No explicit baseline model comparisons were identified in the analyzed quantitative evidence.",
                supporting_evidence=[first_res.supporting_item] if hasattr(first_res, "supporting_item") and first_res.supporting_item else [],
                source_pages=[first_res.page_number] if first_res.page_number else [],
                source_sections=[first_res.section_name] if first_res.section_name else ["results"],
                source_chunks=[first_res.chunk_id] if first_res.chunk_id else [],
                explicit_or_inferred="INFERRED",
                rationale="The paper reports quantitative results but does not explicitly document comparative performance against standard baselines.",
                potential_direction="Benchmark the proposed architecture against classical and contemporary baseline models.",
                severity="Medium",
            )
            gaps.append(g)
        return gaps

    def _categorize_text(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ("dataset", "data", "sample size", "corpus", "benchmark")):
            return "DATASET"
        if any(k in t for k in ("generaliz", "domain", "transfer", "out-of-distribution")):
            return "GENERALIZATION"
        if any(k in t for k in ("memory", "gpu", "compute", "time", "speed", "latency", "efficiency", "quadratic")):
            return "EFFICIENCY"
        if any(k in t for k in ("baseline", "compar", "prior work")):
            return "BASELINE"
        if any(k in t for k in ("robust", "noise", "adversarial")):
            return "ROBUSTNESS"
        if any(k in t for k in ("fairness", "bias")):
            return "FAIRNESS"
        if any(k in t for k in ("ablation", "metric", "evaluat")):
            return "EVALUATION"
        return "METHODOLOGY"

    def _deduplicate_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        seen_statements = set()
        deduped = []
        for g in gaps:
            norm = g.statement.lower().strip()
            if norm not in seen_statements:
                seen_statements.add(norm)
                deduped.append(g)
        return deduped

    def _generate_hypotheses(
        self, gaps: List[ResearchGap], profile: ResearchProfile
    ) -> List[ResearchHypothesis]:
        hypotheses = []
        model_name = profile.models[0].value if profile.models and profile.models[0].value != "Not detected" else "the proposed approach"

        for idx, g in enumerate(gaps[:3]):
            if g.confidence_score < 0.55:
                continue

            h = ResearchHypothesis(
                hypothesis_id=f"hyp_{idx+1}",
                title=f"Potential Hypothesis #{idx+1}: {g.category.title()} Enhancement",
                statement=f"Addressing '{g.statement}' by applying {g.potential_direction.lower()} will improve model robustness and generalization.",
                motivation=f"Derived from {g.explicit_or_inferred.lower()} evidence: {g.rationale}",
                supporting_evidence=g.supporting_evidence,
                variables={
                    "independent_variable": g.potential_direction,
                    "dependent_variable": "Model performance / generalization metric",
                },
                suggested_evaluation=f"Conduct cross-validation evaluation comparing {model_name} with and without the suggested modification.",
                confidence_level=g.confidence_level,
                label="Potential Hypothesis",
            )
            hypotheses.append(h)

        return hypotheses
