"""Unit tests for ExperimentPlan versioning and revision history preservation."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ExperimentPlan
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_exp_ver_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_experiment_versioning_preservation(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Speech Codecs")
    ws_id = ws.workspace_id

    # 1. Version 1 of ExperimentPlan
    exp_v1 = ExperimentPlan(
        experiment_id="exp_codec_v1",
        name="Audio Codec Infilling (Initial)",
        model="SoundStream",
        baseline="Opus",
        dataset="LibriSpeech Clean",
        metrics=["PESQ", "STOI"],
        version=1,
        created_by="mem_alice",
    )
    store.save_experiment(exp_v1, ws_id)

    # 2. Version 2 modifying model and adding WER metric
    exp_v2 = ExperimentPlan(
        experiment_id="exp_codec_v2",
        parent_experiment_id="exp_codec_v1",
        name="Audio Codec Infilling (EnCodec Upgrade)",
        model="EnCodec-24kHz",
        baseline="SoundStream",
        dataset="LibriSpeech Clean",
        metrics=["PESQ", "STOI", "WER"],
        version=2,
        created_by="mem_bob",
    )
    store.save_experiment(exp_v2, ws_id)

    # Verify both versions exist simultaneously without overwriting
    experiments = store.list_experiments(ws_id)
    assert len(experiments) == 2

    e1 = next(e for e in experiments if e.experiment_id == "exp_codec_v1")
    e2 = next(e for e in experiments if e.experiment_id == "exp_codec_v2")

    assert e1.version == 1
    assert e1.model == "SoundStream"
    assert e2.version == 2
    assert e2.parent_experiment_id == "exp_codec_v1"
    assert e2.model == "EnCodec-24kHz"
    assert "WER" in e2.metrics
