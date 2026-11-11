"""Test di src/v2/infer.py su CPU con un GPT2 minuscolo in tmp_path."""
import math

import pandas as pd
import pytest
import torch

from src.config import ATTRS, ModelSize
from src.v2 import corpus, infer, train
from src.v2.model2 import GPT2
from src.v2.spec import INFER_COLUMNS, Job

JOB = Job(stage="A_main", corpus="italiano_pieno", protocol="passi", exposures=10,
          size="tiny", lr_mult=1.0, seed=0, n_persons=30)
N_PERSONE = 30
N_DOMANDE = 12


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "V2_RUNS", tmp_path / "runs")
    monkeypatch.setattr(infer, "V2_RESULTS", tmp_path / "results")
    monkeypatch.setattr(train, "V2_DATA", tmp_path / "data")
    monkeypatch.setattr(infer, "SIZES_V2", {"tiny": ModelSize("tiny", 1, 16, 2, 2e-3)})
    monkeypatch.setattr(infer, "C_QUESTIONS", N_DOMANDE)
    monkeypatch.setattr(infer, "WARMUP_CALLS", 2)
    persone = corpus.load_persons(N_PERSONE)
    tok = train.get_tokenizer(infer.CORPORA[JOB.corpus], persone)
    return tmp_path, persone, tok


def _salva_ckpt(run_id: str, tok, tmp_path) -> None:
    torch.manual_seed(0)
    modello = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=128, n_layer=1, n_embd=16, n_head=2)
    cartella = tmp_path / "runs" / run_id
    cartella.mkdir(parents=True, exist_ok=True)
    torch.save({"model": modello.state_dict()}, cartella / "ckpt.pt")


def _scrivi_a_main(tmp_path) -> None:
    base = {"stage": JOB.stage, "corpus": JOB.corpus, "protocol": JOB.protocol,
            "exposures": JOB.exposures, "size": JOB.size, "lr_mult": JOB.lr_mult,
            "n_persons": JOB.n_persons}
    cartella = tmp_path / "results"
    cartella.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {**base, "seed": 1},   # seed non richiesto
        {**base, "seed": 0},   # riga del modello core
    ]).to_csv(cartella / "A_main.csv", index=False)


def test_answer_tokens_differenza_e_minimo_uno(ambiente):
    _, _, tok = ambiente
    atteso = len(tok.encode("Marta Rossi").ids) - len(tok.encode("Marta").ids)
    assert infer.answer_tokens(tok, "Marta", "Rossi") == max(1, atteso)
    assert infer.answer_tokens(tok, "Marta", "") >= 1


def test_load_ripiega_su_a_lr(ambiente):
    tmp_path, persone, tok = ambiente
    _salva_ckpt(JOB.run_id.replace("A_main", "A_lr"), tok, tmp_path)
    modello, tok_caricato, spec = infer.load(JOB, persone)
    assert not modello.training
    assert spec.name == JOB.corpus
    assert tok_caricato.get_vocab_size() == tok.get_vocab_size()


def test_load_senza_checkpoint_solleva(ambiente):
    _, persone, _ = ambiente
    with pytest.raises(FileNotFoundError):
        infer.load(JOB, persone)


def test_core_job_prende_il_seme_zero(ambiente):
    tmp_path, _, _ = ambiente
    _scrivi_a_main(tmp_path)
    assert infer.core_job(JOB.corpus, JOB.size) == JOB
    with pytest.raises(ValueError):
        infer.core_job("telegrafico", JOB.size)


def test_c_cases_in_e_ordine(ambiente):
    _, persone, tok = ambiente
    spec = infer.CORPORA[JOB.corpus]
    casi = infer.c_cases(spec, persone)
    assert len(casi) == N_DOMANDE
    assert all(c.kind == "in" for c in casi)
    assert casi[0].person == 0
    assert [c.attr for c in casi[:len(ATTRS)]] == list(ATTRS)


@pytest.mark.parametrize("batch", (1, 64))
def test_measure_c_colonne_e_valori(ambiente, batch):
    tmp_path, persone, tok = ambiente
    _salva_ckpt(JOB.run_id, tok, tmp_path)
    riga = infer.measure_c(JOB, persone, batch)
    assert list(riga.keys()) == list(INFER_COLUMNS)
    for col in ("latency_ms_p50", "latency_ms_p90", "ms_per_decode_token", "answers_per_s",
                "flops_per_answer", "answer_tokens_mean"):
        assert math.isfinite(riga[col]) and riga[col] > 0, col
    assert riga["latency_ms_p90"] >= riga["latency_ms_p50"]
    assert 0.0 <= riga["acc"] <= 1.0
    assert riga["n_questions"] == N_DOMANDE
    assert riga["batch"] == batch
    assert riga["params_total"] > riga["params_nonemb"] > 0
    assert riga["flops_per_answer"] == pytest.approx(riga["flops_prefill"] + riga["flops_decode"])
    assert riga["corpus"] == JOB.corpus and riga["size"] == JOB.size and riga["seed"] == JOB.seed


def test_run_infer_c_salta_e_riprende(ambiente, monkeypatch):
    tmp_path, persone, tok = ambiente
    _salva_ckpt(JOB.run_id, tok, tmp_path)
    _scrivi_a_main(tmp_path)
    monkeypatch.setattr(infer.corpus, "load_persons", lambda n: persone)
    monkeypatch.setattr(infer, "C_FORMATS", (JOB.corpus,))
    monkeypatch.setattr(infer, "C_SIZES", (JOB.size,))
    monkeypatch.setattr(infer, "C_BATCHES", (1, 64))
    out = tmp_path / "inference_c.csv"

    infer.run_infer_c(out)
    infer.run_infer_c(out)  # riprendibile: nessuna riga duplicata
    righe = pd.read_csv(out)
    assert len(righe) == 2
    assert list(righe.columns) == list(INFER_COLUMNS)
    assert sorted(righe["batch"].tolist()) == [1, 64]
