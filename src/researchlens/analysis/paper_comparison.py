"""Cross-paper comparative analysis data model and matrix aggregator."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PaperProfile:
    """Standardized representation of a single paper for cross-paper comparison."""
    paper_title: str
    authors: str = "Not detected"
    year: Optional[int] = None
    problem_statement: str = "Not detected"
    dataset: str = "Not detected"
    method: str = "Not detected"
    model: str = "Not detected"
    metrics: str = "Not detected"
    results: str = "Not detected"
    limitations: str = "Not detected"
    future_work: str = "Not detected"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperComparisonMatrix:
    """Matrix container aggregating multiple paper profiles."""
    profiles: List[PaperProfile] = field(default_factory=list)

    def add_paper(self, profile: PaperProfile) -> None:
        self.profiles.append(profile)

    def to_table_dict(self) -> Dict[str, List[Any]]:
        """Return dictionary representation suitable for tabular display or pandas DataFrame."""
        if not self.profiles:
            return {}

        headers = [
            "Paper Title", "Authors", "Year", "Problem", "Dataset",
            "Method", "Model", "Metrics", "Results", "Limitations", "Future Work"
        ]
        rows: Dict[str, List[Any]] = {h: [] for h in headers}

        for p in self.profiles:
            rows["Paper Title"].append(p.paper_title)
            rows["Authors"].append(p.authors)
            rows["Year"].append(p.year or "N/A")
            rows["Problem"].append(p.problem_statement)
            rows["Dataset"].append(p.dataset)
            rows["Method"].append(p.method)
            rows["Model"].append(p.model)
            rows["Metrics"].append(p.metrics)
            rows["Results"].append(p.results)
            rows["Limitations"].append(p.limitations)
            rows["Future Work"].append(p.future_work)

        return rows
