"""Document analysis, gap detection, hypothesis generation, and multi-paper literature intelligence package."""

from researchlens.analysis.comparability import check_scientific_comparability
from researchlens.analysis.conflict_detector import CrossPaperConflict, detect_cross_paper_conflicts
from researchlens.analysis.cross_paper_gap_engine import (
    CrossPaperResearchGapEngine,
    LiteratureResearchGap,
)
from researchlens.analysis.evidence import EvidenceSnippet
from researchlens.analysis.external_provider import (
    ArxivProvider,
    CrossrefProvider,
    ExternalLiteratureProvider,
    ExternalPaper,
    PubMedProvider,
    SemanticScholarProvider,
    UnifiedLiteratureSearchService,
    convert_external_paper_to_document,
    deduplicate_papers,
    rank_papers_by_relevance,
)
from researchlens.analysis.gap_engine import ResearchGapEngine
from researchlens.analysis.gap_evolution import (
    GapEvolutionEngine,
    ResearchGapEvolutionRecord,
    classify_gap_evolution,
    compare_gap_sets,
)
from researchlens.analysis.gap_scorer import calculate_gap_score
from researchlens.analysis.gap_validator import (
    GapValidationResult,
    LiteratureGapValidator,
)
from researchlens.analysis.literature_collection import (
    LiteratureCollection,
    LiteratureDocument,
    generate_deterministic_document_id,
)
from researchlens.analysis.literature_report import (
    LiteratureIntelligenceReport,
    generate_literature_report,
)
from researchlens.analysis.monitoring_report import (
    MonitoringReport,
    generate_monitoring_report,
)
from researchlens.analysis.monitoring_service import (
    ConsoleNotificationProvider,
    NotificationProvider,
    ResearchAlert,
    ResearchMonitoringService,
    detect_paper_changes,
)
from researchlens.analysis.paper_comparison import (
    PaperComparisonMatrix,
    PaperProfile,
)
from researchlens.analysis.paper_matrix import (
    generate_dataset_comparison_table,
    generate_method_comparison_table,
    generate_paper_matrix,
    generate_result_comparison_table,
)
from researchlens.analysis.research_gap_report import (
    ResearchGapReport,
    generate_research_gap_report,
)
from researchlens.analysis.research_gap_schema import ResearchGap, ResearchHypothesis
from researchlens.analysis.research_gaps import extract_research_gaps
from researchlens.analysis.research_topic import (
    MonitoringConfig,
    MonitoringRun,
    PaperHistoryEntry,
    ResearchTopic,
    TopicStore,
)
from researchlens.analysis.semantic_clustering import (
    LimitationCluster,
    cluster_limitations_and_future_work,
)
from researchlens.analysis.trend_analysis import (
    EmergingEntity,
    LiteratureTrendReport,
    ThemeEvolutionRecord,
    TrendAnalyzer,
)

__all__ = [
    "extract_research_gaps",
    "PaperProfile",
    "PaperComparisonMatrix",
    "EvidenceSnippet",
    "ResearchGap",
    "ResearchHypothesis",
    "ResearchGapEngine",
    "calculate_gap_score",
    "ResearchGapReport",
    "generate_research_gap_report",
    "LiteratureDocument",
    "LiteratureCollection",
    "generate_deterministic_document_id",
    "generate_paper_matrix",
    "generate_method_comparison_table",
    "generate_dataset_comparison_table",
    "generate_result_comparison_table",
    "check_scientific_comparability",
    "CrossPaperConflict",
    "detect_cross_paper_conflicts",
    "LimitationCluster",
    "cluster_limitations_and_future_work",
    "LiteratureResearchGap",
    "CrossPaperResearchGapEngine",
    "ExternalLiteratureProvider",
    "ExternalPaper",
    "ArxivProvider",
    "SemanticScholarProvider",
    "CrossrefProvider",
    "PubMedProvider",
    "UnifiedLiteratureSearchService",
    "deduplicate_papers",
    "rank_papers_by_relevance",
    "convert_external_paper_to_document",
    "GapValidationResult",
    "LiteratureGapValidator",
    "LiteratureIntelligenceReport",
    "generate_literature_report",
    "ResearchTopic",
    "MonitoringConfig",
    "MonitoringRun",
    "PaperHistoryEntry",
    "TopicStore",
    "GapEvolutionEngine",
    "ResearchGapEvolutionRecord",
    "classify_gap_evolution",
    "compare_gap_sets",
    "TrendAnalyzer",
    "EmergingEntity",
    "ThemeEvolutionRecord",
    "LiteratureTrendReport",
    "ResearchAlert",
    "NotificationProvider",
    "ConsoleNotificationProvider",
    "ResearchMonitoringService",
    "detect_paper_changes",
    "MonitoringReport",
    "generate_monitoring_report",
]
