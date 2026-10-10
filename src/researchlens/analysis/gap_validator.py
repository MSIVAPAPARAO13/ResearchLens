"""Literature Gap Validator validating candidate research gaps against external scientific literature."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.retrieval.embeddings import get_embedding_model

logger = logging.getLogger("researchlens.gap_validator")


@dataclass
class GapValidationResult:
    """Dataclass capturing external literature validation status for a candidate research gap."""

    gap_statement: str
    status: str  # CONFIRMED_OPEN_GAP, POTENTIALLY_ADDRESSED, UNVERIFIED
    confidence_score: float
    rationale: str
    relevant_external_papers: List[ExternalPaper] = field(default_factory=list)
    suggested_directions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GapValidationResult to a JSON-compatible dictionary."""
        return {
            "gap_statement": self.gap_statement,
            "status": self.status,
            "confidence_score": round(self.confidence_score, 4),
            "rationale": self.rationale,
            "relevant_external_papers": [p.to_dict() for p in self.relevant_external_papers],
            "suggested_directions": self.suggested_directions,
        }


class LiteratureGapValidator:
    """Validates research gaps against external literature by evaluating semantic alignment and addressing claims."""

    def __init__(self, similarity_threshold: float = 0.40):
        self.similarity_threshold = similarity_threshold

    def validate_gap(
        self, gap_statement: str, external_papers: List[ExternalPaper]
    ) -> GapValidationResult:
        """Validate candidate research gap against a set of externally retrieved research papers."""
        if not gap_statement.strip() or not external_papers:
            return GapValidationResult(
                gap_statement=gap_statement,
                status="UNVERIFIED",
                confidence_score=0.0,
                rationale="Insufficient external literature or empty gap statement provided for validation.",
            )

        emb_model = get_embedding_model()
        gap_emb = emb_model.embed_query(gap_statement)
        paper_texts = [f"{p.title}. {p.abstract}" for p in external_papers]
        paper_embs = emb_model.embed_documents(paper_texts)

        gap_vec = np.array(gap_emb, dtype=np.float32)
        sim_scores = []
        for p_emb in paper_embs:
            p_vec = np.array(p_emb, dtype=np.float32)
            sim = float(np.dot(gap_vec, p_vec) / (np.linalg.norm(gap_vec) * np.linalg.norm(p_vec) + 1e-9))
            sim_scores.append(sim)

        # Match relevant papers above threshold
        relevant_matches = [
            (external_papers[i], sim_scores[i])
            for i in range(len(external_papers))
            if sim_scores[i] >= self.similarity_threshold
        ]

        if not relevant_matches:
            return GapValidationResult(
                gap_statement=gap_statement,
                status="UNVERIFIED",
                confidence_score=0.30,
                rationale="No closely aligned external literature papers met the semantic relevance threshold.",
                suggested_directions=["Expand literature search keywords across additional external repositories."],
            )

        # Sort by similarity descending
        relevant_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in relevant_matches[:5]]
        top_sim = relevant_matches[0][1]

        # Scan abstracts for solution keywords vs persistent limitation keywords
        solution_keywords = [
            "we solve", "proposes a solution", "mitigates", "remedies",
            "eliminates the need", "resolves", "efficient alternative",
            "overcomes the limitation", "addresses this limitation",
        ]
        limitation_keywords = [
            "remains a challenge", "still open", "limited by", "bottleneck",
            "unresolved", "ongoing challenge", "insufficiently explored",
            "future work", "critical challenge",
        ]

        has_solution = False
        has_persistent_challenge = False
        competing_titles = []

        for paper, _score in relevant_matches:
            text = f"{paper.title} {paper.abstract}".lower()
            if any(re.search(rf"\b{k}\b", text) for k in solution_keywords):
                has_solution = True
                competing_titles.append(f"'{paper.title}' ({paper.year or 'Recent'})")
            if any(re.search(rf"\b{k}\b", text) for k in limitation_keywords):
                has_persistent_challenge = True

        if has_solution:
            status = "POTENTIALLY_ADDRESSED"
            conf = min(0.95, top_sim + 0.15)
            rat = (
                f"External literature suggests this gap may be actively addressed or mitigated by recent publications, "
                f"including {', '.join(competing_titles[:2])}."
            )
            directions = [
                "Benchmark your proposed approach directly against existing external solutions.",
                "Verify whether external solutions only partially mitigate the constraint in specific settings.",
            ]
        elif has_persistent_challenge or len(relevant_matches) >= 2:
            status = "CONFIRMED_OPEN_GAP"
            conf = min(0.95, top_sim + 0.20)
            rat = (
                f"Confirmed across {len(relevant_matches)} external publications: this constraint continues to be "
                f"cited as an active bottleneck without established consensus solutions."
            )
            directions = [
                "Formulate a novel hypothesis targeting this verified literature gap.",
                "Design experimental evaluations addressing the specific benchmark constraints identified.",
            ]
        else:
            status = "CONFIRMED_OPEN_GAP"
            conf = min(0.85, top_sim)
            rat = f"Aligned with {len(relevant_matches)} external literature studies discussing related problem domains."
            directions = ["Validate generalizability across benchmark datasets."]

        return GapValidationResult(
            gap_statement=gap_statement,
            status=status,
            confidence_score=round(conf, 4),
            rationale=rat,
            relevant_external_papers=top_matches,
            suggested_directions=directions,
        )
