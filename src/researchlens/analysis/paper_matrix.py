"""Cross-paper ResearchProfile matrix aggregator and comparison table builder."""

from typing import Any, Dict, List

from researchlens.analysis.comparability import check_scientific_comparability
from researchlens.analysis.literature_collection import LiteratureCollection
from researchlens.analysis.paper_comparison import PaperComparisonMatrix, PaperProfile
from researchlens.extraction.schema import ResearchProfile


def build_paper_profile_from_extraction(profile: ResearchProfile, paper_title: str) -> PaperProfile:
    """Convert an extracted ResearchProfile into a flat PaperProfile for matrix rendering."""
    authors_str = ", ".join([a.value for a in profile.authors if a.value != "Not detected"]) or "Not detected"
    prob_str = "; ".join([p.value for p in profile.problem if p.value != "Not detected"])[:200] or "Not detected"
    ds_str = ", ".join([d.value for d in profile.datasets if d.value != "Not detected"]) or "Not detected"
    meth_str = ", ".join([m.value for m in profile.methods if m.value != "Not detected"]) or "Not detected"
    mod_str = ", ".join([m.value for m in profile.models if m.value != "Not detected"]) or "Not detected"
    met_str = ", ".join([m.value for m in profile.metrics if m.value != "Not detected"]) or "Not detected"

    results_snippets = []
    for r in profile.results:
        results_snippets.append(f"{r.metric}={r.value}")
    res_str = ", ".join(results_snippets) if results_snippets else "Not detected"

    lim_str = "; ".join([lim.value for lim in profile.limitations if lim.value != "Not detected"])[:200] or "Not detected"
    fw_str = "; ".join([f.value for f in profile.future_work if f.value != "Not detected"])[:200] or "Not detected"

    return PaperProfile(
        paper_title=profile.title or paper_title,
        authors=authors_str,
        year=profile.year,
        problem_statement=prob_str,
        dataset=ds_str,
        method=meth_str,
        model=mod_str,
        metrics=met_str,
        results=res_str,
        limitations=lim_str,
        future_work=fw_str,
    )


def generate_paper_matrix(collection: LiteratureCollection) -> PaperComparisonMatrix:
    """Generate a PaperComparisonMatrix from a LiteratureCollection."""
    matrix = PaperComparisonMatrix()
    for doc in collection.documents:
        if doc.profile:
            p_prof = build_paper_profile_from_extraction(doc.profile, doc.title)
            matrix.add_paper(p_prof)
    return matrix


def generate_method_comparison_table(collection: LiteratureCollection) -> List[Dict[str, Any]]:
    """Build a method/model comparison table across all papers in the collection."""
    method_map: Dict[str, List[str]] = {}

    for doc in collection.documents:
        if not doc.profile:
            continue
        models_and_methods = doc.profile.models + doc.profile.methods
        for item in models_and_methods:
            val = item.value.strip()
            if val and val not in ("Not detected", "Not explicitly stated"):
                if val not in method_map:
                    method_map[val] = []
                if doc.title not in method_map[val]:
                    method_map[val].append(doc.title)

    table = []
    for method_name, papers in method_map.items():
        table.append({
            "Method / Model": method_name,
            "Paper Count": len(papers),
            "Supporting Papers": ", ".join(papers),
        })

    return sorted(table, key=lambda x: x["Paper Count"], reverse=True)


def generate_dataset_comparison_table(collection: LiteratureCollection) -> List[Dict[str, Any]]:
    """Build a dataset comparison table across all papers in the collection."""
    dataset_map: Dict[str, Dict[str, Any]] = {}

    for doc in collection.documents:
        if not doc.profile:
            continue
        for ds in doc.profile.datasets:
            val = ds.value.strip()
            if val and val not in ("Not detected", "Not explicitly stated"):
                if val not in dataset_map:
                    dataset_map[val] = {
                        "Dataset": val,
                        "Papers": [],
                        "Sections": set(),
                        "Page Provenance": set(),
                    }
                if doc.title not in dataset_map[val]["Papers"]:
                    dataset_map[val]["Papers"].append(doc.title)
                if ds.section_name:
                    dataset_map[val]["Sections"].add(ds.section_name)
                if ds.page_number:
                    dataset_map[val]["Page Provenance"].add(f"p.{ds.page_number}")

    table = []
    for ds_name, info in dataset_map.items():
        table.append({
            "Dataset": ds_name,
            "Paper Count": len(info["Papers"]),
            "Supporting Papers": ", ".join(info["Papers"]),
            "Provenance": ", ".join(sorted(info["Page Provenance"])),
        })

    return sorted(table, key=lambda x: x["Paper Count"], reverse=True)


def generate_result_comparison_table(collection: LiteratureCollection) -> List[Dict[str, Any]]:
    """Build a quantitative result comparison table with scientific comparability checks."""
    table = []

    for i in range(len(collection.documents)):
        doc_a = collection.documents[i]
        if not doc_a.profile:
            continue
        for res_a in doc_a.profile.results:
            for j in range(i + 1, len(collection.documents)):
                doc_b = collection.documents[j]
                if not doc_b.profile:
                    continue
                for res_b in doc_b.profile.results:
                    is_comp, reason = check_scientific_comparability(
                        metric_a=res_a.metric,
                        dataset_a=res_a.dataset or "Unknown",
                        metric_b=res_b.metric,
                        dataset_b=res_b.dataset or "Unknown",
                    )
                    table.append({
                        "Paper A": doc_a.title,
                        "Model A": res_a.model or "Target Model",
                        "Result A": f"{res_a.metric}={res_a.value}",
                        "Paper B": doc_b.title,
                        "Model B": res_b.model or "Target Model",
                        "Result B": f"{res_b.metric}={res_b.value}",
                        "Comparability Status": "Comparable" if is_comp else "Not directly comparable",
                        "Rationale": reason,
                    })

    return table
