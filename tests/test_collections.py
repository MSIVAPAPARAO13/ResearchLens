"""Unit tests for PaperCollection and SavedPaper management."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import PaperCollection, SavedPaper
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_col_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paper_and_collection_management(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Biomedical NLP")
    ws_id = ws.workspace_id

    # 1. Save papers
    p1 = SavedPaper(
        paper_id="doi:10.1001/p1",
        workspace_id=ws_id,
        title="BioBERT: pre-trained biomedical language representation",
        authors=["Lee et al."],
        year=2020,
        tags=["baseline", "biomedical"],
        priority="HIGH",
    )
    p2 = SavedPaper(
        paper_id="doi:10.1001/p2",
        workspace_id=ws_id,
        title="PubMedBERT: domain-specific pretraining",
        authors=["Gu et al."],
        year=2021,
        tags=["core", "biomedical"],
        priority="HIGH",
    )
    store.save_paper(p1)
    store.save_paper(p2)

    papers = store.list_papers(ws_id)
    assert len(papers) == 2

    # 2. Create collection
    col = PaperCollection(
        collection_id="col_baselines",
        workspace_id=ws_id,
        name="Biomedical Baselines",
        description="Standard pre-trained language models",
    )
    store.save_collection(col)
    assert len(store.list_collections(ws_id)) == 1

    # 3. Add papers to collection
    assert store.add_paper_to_collection(ws_id, "col_baselines", "doi:10.1001/p1") is True
    assert store.add_paper_to_collection(ws_id, "col_baselines", "doi:10.1001/p2") is True
    # Duplicate add returns False or does not duplicate
    assert store.add_paper_to_collection(ws_id, "col_baselines", "doi:10.1001/p1") is False

    updated_col = store.list_collections(ws_id)[0]
    assert len(updated_col.paper_ids) == 2

    # 4. Remove paper from collection without deleting paper record
    assert store.remove_paper_from_collection(ws_id, "col_baselines", "doi:10.1001/p1") is True
    remaining_in_col = store.list_collections(ws_id)[0].paper_ids
    assert len(remaining_in_col) == 1
    assert "doi:10.1001/p1" not in remaining_in_col

    # Underlying paper must still exist in workspace!
    assert len(store.list_papers(ws_id)) == 2
