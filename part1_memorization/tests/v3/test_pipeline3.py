"""Test della pipeline v3: casi per stripper, valutazione con stripper e misura end-to-end con checkpoint finti."""
from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from src.config import ATTRS
from src.v2 import corpus
from src.v2 import spec as v2_spec
from src.v2 import train as v2_train
from src.v2.model2 import GPT2
from src.v2.spec import CORPORA, JOB_KEYS, SIZES_V2, Job
from src.v2.train import get_tokenizer
from src.v3 import pipeline3
from src.v3.pipeline3 import cases, evaluate_stripped, run_pipeline
from src.v3.spec3 import PIPE_COLUMNS
from src.v3.strip import ListStripper, SpacyStripper


@pytest.mark.parametrize("stripper_cls", [SpacyStripper, ListStripper])
def test_cases(stripper_cls):
    persone = corpus.load_persons(3)
    stripper = stripper_cls()
    casi = cases(stripper, persone, 3)
    assert len(casi) == 3 * len(ATTRS)
    i, attr, domanda, core, atteso, canonico = casi[0]
    assert (i, attr) == (0, ATTRS[0])
    assert core == stripper.strip(domanda)
    assert not atteso.endswith(".")
    assert canonico in corpus._canonici(attr)


def test_evaluate_stripped_keys(tmp_path, monkeypatch):
    # il tokenizer di italiano_pieno si addestra su V2_DATA: lo si scrive in tmp_path
    monkeypatch.setattr(v2_spec, "V2_DATA", tmp_path, raising=False)
    monkeypatch.setattr(v2_train, "V2_DATA", tmp_path, raising=False)
    persone = corpus.load_persons(20)
    spec = CORPORA[pipeline3.BASE_PIENO]
    tok = get_tokenizer(spec, persone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=256, n_layer=2, n_embd=64, n_head=2).to(device)
    risultato = evaluate_stripped(ListStripper())(model, tok, spec, persone, 2)
    attesi = ({f"acc_{a}" for a in ATTRS} | {f"cross_{a}" for a in ATTRS}
              | {"acc_mean", "cross_mean", "cross_lenient_mean"})
    assert set(risultato) == attesi
    assert math.isnan(risultato["cross_mean"])
    assert math.isnan(risultato["cross_lenient_mean"])


def _job(corpus_name: str) -> Job:
    return Job(stage="A_main", corpus=corpus_name, protocol="passi", exposures=1, size="0.60M",
               lr_mult=1.0, seed=0, n_persons=20)


def _checkpoint(job: Job, radice, persone) -> None:
    # stessa taglia che il caricatore legge da SIZES_V2, con contesto ridotto
    tok = get_tokenizer(CORPORA[job.corpus], persone)
    size = SIZES_V2[job.size]
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=256, n_layer=size.n_layer,
                 n_embd=size.n_embd, n_head=size.n_head)
    percorso = radice / job.run_id / "ckpt.pt"
    percorso.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, percorso)


def _riga_csv(job: Job) -> dict:
    """Riga di un CSV di risultati: prima le colonne di JOB_KEYS."""
    return {chiave: getattr(job, chiave) for chiave in JOB_KEYS}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="run_pipeline va misurato su cuda")
def test_run_pipeline(tmp_path, monkeypatch):
    v2_res, v3_res = tmp_path / "v2_results", tmp_path / "v3_results"
    v2_runs, v3_runs = tmp_path / "v2_runs", tmp_path / "v3_runs"
    for cartella in (v2_res, v3_res):
        cartella.mkdir()
    monkeypatch.setattr(pipeline3, "V2_RESULTS", v2_res)
    monkeypatch.setattr(pipeline3, "V3_RESULTS", v3_res)
    monkeypatch.setattr(pipeline3, "V2_RUNS", v2_runs)
    monkeypatch.setattr(pipeline3, "V3_RUNS", v3_runs)
    monkeypatch.setattr(pipeline3, "PIPE_QUESTIONS", 4)
    monkeypatch.setattr(pipeline3, "PIPE_SIZES", ("0.60M",))

    persone = corpus.load_persons(20)
    jobs_e = [_job("telegrafico_auto"), _job("telegrafico_lista")]
    jobs_a = [_job("interlingua"), _job(pipeline3.BASE_PIENO)]
    for job in jobs_e:
        _checkpoint(job, v3_runs, persone)
    for job in jobs_a:
        _checkpoint(job, v2_runs, persone)
    pd.DataFrame([_riga_csv(j) for j in jobs_e]).to_csv(v3_res / "E_main.csv", index=False)
    pd.DataFrame([_riga_csv(j) for j in jobs_a]).to_csv(v2_res / "A_main.csv", index=False)

    out = tmp_path / "pipe.csv"
    run_pipeline(out)

    righe = pd.read_csv(out)
    assert list(righe.columns) == list(PIPE_COLUMNS)
    assert len(righe) == 3 * 1 * 2
    for colonna in ("strip_cpu_ms", "core_gpu_ms", "realize_cpu_ms", "total_ms", "pieno_ms"):
        assert (righe[colonna] > 0).all(), colonna
