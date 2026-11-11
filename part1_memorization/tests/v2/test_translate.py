"""Test di src.v2.translate su CPU con traduttori minuscoli."""
from __future__ import annotations

import csv
import math

import pytest
import torch

from src.config import ATTRS, EOS, ModelSize
from src.v2 import corpus, translate
from src.v2.spec import TRANSLATE_COLUMNS

N = 40


@pytest.fixture(scope="module")
def persone():
    return corpus.load_persons(N)


@pytest.fixture
def piccolo(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "V2_DATA", tmp_path / "data")
    monkeypatch.setattr(translate, "V2_RUNS", tmp_path / "runs")
    monkeypatch.setattr(translate, "VOCAB_SIZE", 512)
    monkeypatch.setattr(translate, "EPOCHS", 2)
    monkeypatch.setattr(translate, "C_TRANSLATE_TRAIN_PERSONS", (0, 20))
    monkeypatch.setattr(translate, "C_QUESTIONS", 12)
    monkeypatch.setattr(translate, "C_MAX_OUT_TOKENS", 8)
    monkeypatch.setattr(translate, "SIZES_V2", {"tiny": ModelSize("tiny", 1, 32, 2, 2e-3)})
    monkeypatch.setattr(translate, "C_TRANSLATE_SIZE", "tiny")
    monkeypatch.setattr(translate, "N_PERSONS", N)
    return tmp_path


def _leggi(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_pairs_strip(persone):
    coppie = translate.pairs(persone, [0, 1], "telegrafico", "strip")
    assert len(coppie) == 2 * len(ATTRS)
    src, tgt, attr, truth = coppie[0]
    assert attr == ATTRS[0]
    assert src == corpus.eval_prompt(persone[0], attr, "pieno", "it")
    assert tgt == corpus.eval_prompt(persone[0], attr, "telegrafico", "it")
    assert truth == persone[0].attrs[attr]


def test_pairs_trans(persone):
    src, tgt, attr, truth = translate.pairs(persone, [3], "interlingua", "trans")[1]
    assert attr == ATTRS[1]
    C = corpus.eval_prompt(persone[3], attr, "interlingua", "it")
    s = corpus.surface(attr, truth, "interlingua", "it")
    P = corpus.eval_prompt(persone[3], attr, "pieno", "it")
    assert src == f"{C} {s}"
    assert tgt == f"{P} {truth}"


def test_get_tok_speciali_e_cache(piccolo, persone):
    tok = translate.get_tok("telegrafico", persone)
    assert tok.token_to_id(translate.SEP) is not None
    assert tok.token_to_id(EOS) is not None
    assert (piccolo / "data" / "tok_translate_telegrafico" / "tokenizer.json").exists()


def test_train_e_ricaricamento(piccolo, persone):
    model, tok, info = translate.train("telegrafico", "strip", persone)
    n = 20 * len(ATTRS)
    assert info["train_steps"] == translate.EPOCHS * math.ceil(n / translate.SEQS_PER_STEP)
    assert info["train_flops"] > 0
    assert (piccolo / "runs" / "C_strip_telegrafico" / "ckpt.pt").exists()

    # il checkpoint completo viene caricato, senza nuovo training
    model2, _, info2 = translate.train("telegrafico", "strip", persone)
    assert info2 == info
    for (nome, a), (_, b) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.equal(a, b), nome


def test_run_statistiche(piccolo, persone):
    model, tok, _ = translate.train("telegrafico", "trans", persone)
    srcs = [c[0] for c in translate.pairs(persone, [0, 1], "telegrafico", "trans")]
    for batch in (1, 4):
        outs, st = translate.run(model, tok, srcs, batch)
        assert len(outs) == len(srcs)
        assert all(isinstance(o, str) for o in outs)
        assert set(st) == {"prefill_ms", "decode_ms", "latency_ms_mean", "answers_per_s",
                           "flops_per_request", "in_tokens_mean", "out_tokens_mean"}
        assert st["answers_per_s"] > 0
        assert st["flops_per_request"] > 0
        assert st["in_tokens_mean"] > 0
        assert 0 <= st["out_tokens_mean"] <= translate.C_MAX_OUT_TOKENS


def test_run_translate_righe_e_ripresa(piccolo, monkeypatch):
    monkeypatch.setattr(translate, "C_FORMATS", ("telegrafico",))
    csv_path = piccolo / "results" / "translate.csv"

    translate.run_translate(csv_path)
    righe = _leggi(csv_path)
    assert len(righe) == 2 * len(translate.C_BATCHES)
    assert tuple(righe[0]) == TRANSLATE_COLUMNS
    for r in righe:
        assert int(r["n_questions"]) == translate.C_QUESTIONS
        if r["model"].startswith("strip_"):
            assert math.isnan(float(r["values_em"]))
        else:
            assert 0.0 <= float(r["values_em"]) <= 1.0

    # rilancio: le combinazioni gia' scritte non vengono ripetute
    translate.run_translate(csv_path)
    assert len(_leggi(csv_path)) == len(righe)
