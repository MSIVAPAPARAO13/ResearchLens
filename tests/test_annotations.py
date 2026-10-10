"""Unit tests for ResearchAnnotation creation and categorization."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchAnnotation
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_ann_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_annotation_lifecycle(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Graph Neural Networks")
    ws_id = ws.workspace_id

    ann = ResearchAnnotation(
        annotation_id="ann_001",
        workspace_id=ws_id,
        paper_id="doi:10.1001/gnn_over_smoothing",
        page_number=3,
        section_name="Theoretical Analysis",
        chunk_id="chk_sec2_12",
        source_text="Deep GCNs suffer from node representation collapse as depth exceeds 4 layers.",
        annotation_type="LIMITATION",
        content="Need to test whether residual connections mitigate this collapse on molecular graphs.",
    )
    store.save_annotation(ann)

    anns = store.list_annotations(ws_id)
    assert len(anns) == 1
    a = anns[0]
    assert a.annotation_type == "LIMITATION"
    assert "node representation collapse" in a.source_text
    assert "residual connections" in a.content

    # Ensure source text is preserved unchanged
    assert a.source_text == "Deep GCNs suffer from node representation collapse as depth exceeds 4 layers."

    assert store.delete_annotation(ws_id, "ann_001") is True
    assert len(store.list_annotations(ws_id)) == 0
