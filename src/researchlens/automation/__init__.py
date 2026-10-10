"""ResearchLens Research Automation Package (Phase 15).

Provides evidence-grounded research workflow orchestration, hypothesis proposal,
evaluation integration, and finding candidate generation.
"""

from researchlens.automation.finding_candidate import (
    FindingCandidateGenerator,
)
from researchlens.automation.proposer import (
    EvidenceGroundedProposer,
    GroundedHypothesisProposal,
)
from researchlens.automation.workflow import (
    ResearchWorkflowOrchestrator,
)

__all__ = [
    "ResearchWorkflowOrchestrator",
    "EvidenceGroundedProposer",
    "GroundedHypothesisProposal",
    "FindingCandidateGenerator",
]
