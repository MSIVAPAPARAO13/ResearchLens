"""Unit tests for EvidenceBookmark creation and provenance preservation."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import EvidenceBookmark
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_bm_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evidence_bookmark_provenance(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Vision Transformers")
    ws_id = ws.workspace_id

    bm = EvidenceBookmark(
        bookmark_id="bm_001",
        workspace_id=ws_id,
        paper_id="doi:10.1001/vit2020",
        page_number=5,
        section_name="Empirical Results",
        chunk_id="chunk_sec4_01",
        source_text="ViT attains 88.55% top-1 accuracy on ImageNet-1k when pretrained on JFT-300M.",
        label="ImageNet Benchmark",
        note="Key comparative accuracy result.",
    )
    store.save_evidence(bm)

    bms = store.list_evidence(ws_id)
    assert len(bms) == 1
    retrieved = bms[0]
    assert retrieved.paper_id == "doi:10.1001/vit2020"
    assert retrieved.page_number == 5
    assert retrieved.section_name == "Empirical Results"
    assert retrieved.chunk_id == "chunk_sec4_01"
    assert "88.55%" in retrieved.source_text

    # Test delete
    assert store.delete_evidence(ws_id, "bm_001") is True
    assert len(store.list_evidence(ws_id)) == 0
