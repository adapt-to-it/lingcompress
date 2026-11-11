"""Test di src.v2.train: prepare idempotente, run_job completo, rilancio dal risultato salvato."""
from __future__ import annotations

import math

import pytest
import torch

from src.config import ATTRS
from src.v2 import corpus, train
from src.v2.model2 import GPT2
from src.v2.spec import CORPORA, RESULT_COLUMNS_V2, Job

N = 64
EVAL = 16


@pytest.fixture(scope="module")
def persone():
    return corpus.load_persons(N)


@pytest.fixture
def cartelle(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "V2_DATA", tmp_path / "data")
    monkeypatch.setattr(train, "V2_RUNS", tmp_path / "runs")
    return tmp_path


def _job(corpo: str, protocollo: str) -> Job:
    return Job(stage="test", corpus=corpo, protocol=protocollo, exposures=2, size="0.15M",
             lr_mult=1.0, seed=0, n_persons=N, eval_persons=EVAL)


def _uguali(a: dict, b: dict) -> bool:
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x) and isinstance(y, float) and math.isnan(y):
            continue
        if x != y:
            return False
    return list(a) == list(b)


@pytest.mark.parametrize("corpo,protocollo", [
    ("italiano_pieno", "passi"),
    ("italiano_pieno", "esposizioni"),
    ("multi_bridge", "passi"),
])
def test_run_job_riga(cartelle, persone, corpo, protocollo):
    job = _job(corpo, protocollo)
    riga = train.run_job(job, persone)

    assert tuple(riga) == RESULT_COLUMNS_V2
    for a in ATTRS:
        assert 0.0 <= riga[f"acc_{a}"] <= 1.0
    assert 0.0 <= riga["acc_mean"] <= 1.0
    assert riga["train_tokens"] > 0
    assert riga["flops"] == 6 * riga["params_total"] * riga["train_tokens"]
    if CORPORA[corpo].cross:
        assert all(math.isfinite(riga[f"cross_{a}"]) for a in ATTRS)
        assert math.isfinite(riga["cross_mean"])
        assert math.isfinite(riga["cross_lenient_mean"])
    else:
        assert all(math.isnan(riga[f"cross_{a}"]) for a in ATTRS)

    # il rilancio ritorna la riga salvata senza ripetere training ed eval
    assert _uguali(train.run_job(job, persone), riga)


def test_prepare_idempotente(cartelle, persone):
    spec = CORPORA["italiano_pieno"]
    train.prepare(spec, persone, 2)
    file_epoche = sorted((cartelle / "data").rglob("epoch*.npz"))
    assert len(file_epoche) == 2
    istanti = [f.stat().st_mtime_ns for f in file_epoche]

    train.prepare(spec, persone, 2)
    assert [f.stat().st_mtime_ns for f in file_epoche] == istanti


def test_gpt2_loss_uguale_a_forward():
    torch.manual_seed(0)
    model = GPT2(vocab_size=64, ctx_len=16, n_layer=2, n_embd=32, n_head=4)
    idx = torch.randint(0, 64, (3, 10))
    targets = torch.randint(0, 64, (3, 10))
    targets[:, 6:] = -100
    _, atteso = model(idx, targets)
    assert torch.allclose(model.loss(idx, targets), atteso, atol=1e-5)
