"""Test di src.v3.train3: batch vettorizzati, stream come v2, run_job end-to-end."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.v2 import corpus
from src.v2 import train as v2_train
from src.v2.spec import CORPORA, RESULT_COLUMNS_V2, Job
from src.v3 import train3

PAD_TEST = 4097


def _padding_loop(tokens, offsets, idx, T, pad):
    """Riferimento: padding per biografia, come in v2.run_job."""
    righe = np.full((len(idx), T), pad, dtype=np.int64)
    for r, i in enumerate(idx):
        b = tokens[offsets[i]:offsets[i + 1]]
        righe[r, :len(b)] = b
    return righe


def test_build_batch_uguale_al_loop_v2():
    rng = np.random.default_rng(0)
    lunghezze = rng.integers(1, 30, size=80)
    offsets = np.zeros(len(lunghezze) + 1, dtype=np.int64)
    np.cumsum(lunghezze, out=offsets[1:])
    tokens = rng.integers(0, 4096, size=int(offsets[-1]), dtype=np.uint16)
    for _ in range(5):
        idx = rng.integers(0, len(lunghezze), size=40)
        for T in (int(lunghezze[idx].max()), int(lunghezze.max()) + 3):
            atteso = _padding_loop(tokens, offsets, idx, T, PAD_TEST)
            ottenuto = train3.build_batch(tokens, offsets, idx, T, PAD_TEST)
            assert ottenuto.dtype == np.int64
            assert np.array_equal(ottenuto, atteso)


def test_stream_uguale_a_bio_stream_v2_tra_due_epoche(tmp_path, monkeypatch):
    monkeypatch.setattr(v2_train, "V2_DATA", tmp_path / "v2")
    n = 300
    spec = CORPORA["italiano_pieno"]
    persons = corpus.load_persons(n)
    prepare_epoche = 2
    v2_train.prepare(spec, persons, prepare_epoche)
    _, offsets0, _ = v2_train._leggi_epoca(spec, n, 0)
    n_epoca = len(offsets0) - 1
    T = max(v2_train._max_len(spec, n, e) for e in range(prepare_epoche))

    bps = 64
    start = n_epoca - 3          # il primo batch attraversa il confine di epoca
    riferimento = v2_train._bio_stream(spec, n, 0, start, n_epoca)
    stream = train3._batch_stream(spec, n, 0, start, n_epoca, bps, T, PAD_TEST)
    for _ in range(3):           # tre batch consecutivi
        atteso = np.full((bps, T), PAD_TEST, dtype=np.int64)
        for i in range(bps):
            b = next(riferimento)
            atteso[i, :len(b)] = b
        assert np.array_equal(next(stream), atteso)


def _uguali(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)  # NaN uguale a NaN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="serve CUDA")
def test_run_job_riga_completa_e_ripresa_da_done(tmp_path, monkeypatch):
    monkeypatch.setattr(v2_train, "V2_DATA", tmp_path / "v2")
    monkeypatch.setattr(train3, "V3_RUNS", tmp_path / "v3")
    persons = corpus.load_persons(300)
    job = Job(stage="T3_test", corpus="italiano_pieno", protocol="passi", exposures=2,
              size="0.15M", lr_mult=1.0, seed=0, n_persons=300, eval_persons=20)

    riga = train3.run_job(job, persons, compile=False)
    assert set(riga) == set(RESULT_COLUMNS_V2)
    assert math.isfinite(riga["final_loss"])

    di_nuovo = train3.run_job(job, persons, compile=False)
    assert _uguali(riga, di_nuovo)
