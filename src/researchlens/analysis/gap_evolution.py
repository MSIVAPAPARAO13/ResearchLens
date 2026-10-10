"""Research gap evolution intelligence across monitoring runs and literature progression."""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.gap_validator import GapValidationResult, LiteratureGapValidator
from researchlens.utils.logging import get_logger

logger = get_logger("gap_evolution")

VALID_EVOLUTION_STATUSES = {
    "NEW",
    "PERSISTENT",
    "STRENGTHENING",
    "WEAKENING",
    "POTENTIALLY_ADDRESSED",
    "UNVERIFIED",
}


@dataclass
class ResearchGapEvolutionRecord:
    """Historical progression track of a specific research gap over time."""

    gap_id: str
    gap_statement: str
    category: str
    first_detected: str
    last_detected: str
    status: str = "NEW"  # NEW, PERSISTENT, STRENGTHENING, WEAKENING, POTENTIALLY_ADDRESSED, UNVERIFIED
    confidence: float = 0.5
    evidence_count: int = 1
    supporting_papers: List[str] = field(default_factory=list)  # Titles or IDs
    evidence_snippets: List[str] = field(default_factory=list)
    addressed_by_papers: List[str] = field(default_factory=list)
    validation_status: Optional[str] = None  # CONFIRMED_OPEN_GAP, POTENTIALLY_ADDRESSED, UNVERIFIED
    validation_rationale: Optional[str] = None
    observation_count: int = 1
    history_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchGapEvolutionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def normalize_gap_statement(statement: str) -> str:
    """Normalize gap statement for semantic match comparison."""
    clean = re.sub(r"[^\w\s]", "", statement.lower())
    words = [w for w in clean.split() if len(w) > 3]
    return " ".join(sorted(set(words)))


def _gap_similarity(stmt1: str, stmt2: str) -> float:
    """Compute lexical Jaccard similarity between two gap statements."""
    w1 = set(normalize_gap_statement(stmt1).split())
    w2 = set(normalize_gap_statement(stmt2).split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def classify_gap_evolution(
    previous_record: Optional[ResearchGapEvolutionRecord],
    current_papers: List[str],
    validation_result: Optional[GapValidationResult] = None,
) -> str:
    """Classify the evolution trajectory of a research gap across iterations.

    Rules:
    - If no previous record exists: NEW
    - If external validation classifies as POTENTIALLY_ADDRESSED with high confidence: POTENTIALLY_ADDRESSED
    - If supporting paper count grew: STRENGTHENING
    - If supporting paper count dropped or no papers re-observed: WEAKENING
    - Otherwise: PERSISTENT
    """
    if previous_record is None:
        return "NEW"

    if validation_result and validation_result.status == "POTENTIALLY_ADDRESSED":
        return "POTENTIALLY_ADDRESSED"

    prev_count = len(previous_record.supporting_papers)
    curr_count = len(set(previous_record.supporting_papers) | set(current_papers))

    if curr_count > prev_count:
        return "STRENGTHENING"
    elif curr_count < prev_count:
        return "WEAKENING"
    else:
        return "PERSISTENT"


def _get_gap_text(gap: Any) -> str:
    return getattr(gap, "statement", getattr(gap, "description", str(gap)))


def _get_gap_evidence(gap: Any) -> List[str]:
    raw = getattr(gap, "supporting_evidence", getattr(gap, "evidence", []))
    res = []
    for item in raw:
        if hasattr(item, "text"):
            res.append(item.text)
        elif hasattr(item, "value"):
            res.append(str(item.value))
        else:
            res.append(str(item))
    return res


def compare_gap_sets(
    previous_gaps: List[ResearchGapEvolutionRecord],
    current_gaps: List[Any],
) -> Dict[str, List[Any]]:
    """Compare previous gap records against newly observed gaps.

    Returns:
        Dict with keys: 'new_gaps', 'persistent_gaps', 'strengthened_gaps'
    """
    new_gaps = []
    persistent_gaps = []
    strengthened_gaps = []

    for curr in current_gaps:
        curr_text = _get_gap_text(curr)
        curr_ev = _get_gap_evidence(curr)
        matched_prev = None
        for prev in previous_gaps:
            if _gap_similarity(curr_text, prev.gap_statement) >= 0.45 or getattr(curr, "gap_id", "") == prev.gap_id:
                matched_prev = prev
                break

        if matched_prev is None:
            new_gaps.append(curr)
        else:
            if len(curr_ev) > len(matched_prev.evidence_snippets):
                strengthened_gaps.append((matched_prev, curr))
            else:
                persistent_gaps.append((matched_prev, curr))

    return {
        "new_gaps": new_gaps,
        "persistent_gaps": persistent_gaps,
        "strengthened_gaps": strengthened_gaps,
    }


class GapEvolutionEngine:
    """Orchestrates research gap lifecycle tracking and external validation over time."""

    def __init__(self, gap_validator: Optional[LiteratureGapValidator] = None):
        self.gap_validator = gap_validator or LiteratureGapValidator(similarity_threshold=0.25)

    def track_gap_history(
        self,
        existing_history: Dict[str, ResearchGapEvolutionRecord],
        new_gaps: List[Any],
        monitored_papers: List[ExternalPaper],
        paper_title: str = "Monitored Literature Run",
    ) -> Dict[str, ResearchGapEvolutionRecord]:
        """Update gap evolution history with newly detected gaps and external validation findings."""
        updated_history = dict(existing_history)
        now_iso = datetime.now(timezone.utc).isoformat()

        for gap in new_gaps:
            gap_text = _get_gap_text(gap)
            gap_ev = _get_gap_evidence(gap)
            gap_id = getattr(gap, "gap_id", "")
            gap_cat = getattr(gap, "category", "Methodological")
            gap_conf = getattr(gap, "confidence_score", 0.7)

            matched_key = None
            for key, record in updated_history.items():
                if _gap_similarity(gap_text, record.gap_statement) >= 0.45 or (gap_id and gap_id == record.gap_id):
                    matched_key = key
                    break

            # Validate against current monitored papers
            validation_res: Optional[GapValidationResult] = None
            if monitored_papers:
                try:
                    validation_res = self.gap_validator.validate_gap(gap_text, monitored_papers)
                except Exception as e:
                    logger.warning(f"Error during gap validation: {e}")

            val_papers = []
            if validation_res:
                raw_p = getattr(validation_res, "relevant_external_papers", getattr(validation_res, "supporting_papers", []))
                val_papers = [p.title for p in raw_p if hasattr(p, "title")]

            if matched_key:
                rec = updated_history[matched_key]
                prev_status = rec.status
                new_supporting = list(set(rec.supporting_papers + [paper_title]))
                new_evidence = list(set(rec.evidence_snippets + gap_ev))

                new_status = classify_gap_evolution(rec, [paper_title], validation_res)

                rec.last_detected = now_iso
                rec.status = new_status
                rec.observation_count += 1
                rec.supporting_papers = new_supporting
                rec.evidence_snippets = new_evidence
                rec.evidence_count = len(new_evidence)
                rec.confidence = max(rec.confidence, gap_conf)

                if validation_res:
                    rec.validation_status = validation_res.status
                    rec.validation_rationale = validation_res.rationale
                    if val_papers:
                        rec.addressed_by_papers = list(set(rec.addressed_by_papers + val_papers))

                rec.history_events.append({
                    "timestamp": now_iso,
                    "previous_status": prev_status,
                    "new_status": new_status,
                    "supporting_count": len(new_supporting),
                })
            else:
                # Brand new gap
                gap_key = gap_id or f"gap_{len(updated_history) + 1}"
                status = "NEW"
                if validation_res and validation_res.status == "POTENTIALLY_ADDRESSED":
                    status = "POTENTIALLY_ADDRESSED"

                rec = ResearchGapEvolutionRecord(
                    gap_id=gap_key,
                    gap_statement=gap_text,
                    category=gap_cat,
                    first_detected=now_iso,
                    last_detected=now_iso,
                    status=status,
                    confidence=gap_conf,
                    evidence_count=len(gap_ev),
                    supporting_papers=[paper_title],
                    evidence_snippets=list(gap_ev),
                    addressed_by_papers=val_papers,
                    validation_status=validation_res.status if validation_res else "UNVERIFIED",
                    validation_rationale=validation_res.rationale if validation_res else "First observed in literature.",
                    observation_count=1,
                    history_events=[{
                        "timestamp": now_iso,
                        "previous_status": "NONE",
                        "new_status": status,
                        "supporting_count": 1,
                    }],
                )
                updated_history[gap_key] = rec

        return updated_history

