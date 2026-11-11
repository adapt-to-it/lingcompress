"""Test CPU della fase C end-to-end: caricatori e generazione finti, nessun modello reale."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.v2 import corpus, e2e, infer, kvgen, translate
from src.v2.spec import E2E_COLUMNS, Job


class FakeTok:
    def encode(self, testo):
        return SimpleNamespace(ids=[ord(c) for c in testo])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def token_to_id(self, _):
        return 1

    def get_vocab_size(self):
        return 256


class FakeModel(torch.nn.Module):
    def __init__(self, tag: str, ctx_len: int = 64):
        super().__init__()
        self.tag = tag
        self.ctx_len = ctx_len
        self.tok_emb = torch.nn.Embedding(4, 4)
        self.blocks = [torch.nn.Identity(), torch.nn.Identity()]
        self.lin = torch.nn.Linear(1, 1)

    def count_params(self):
        return 10, 100


@pytest.fixture
def finto(monkeypatch, tmp_path):
    persone = corpus.load_persons(1)
    chiamate = []

    def fake_train(fmt, direction, persons):
        chiamate.append(("train", fmt, direction))
        return FakeModel(f"{direction}_{fmt}"), FakeTok(), {}

    def fake_run(model, tok, srcs, batch):
        chiamate.append(("run", model.tag, batch))
        outs = list(srcs) if model.tag.startswith("strip") else [""] * len(srcs)
        return outs, {"latency_ms_mean": 2.0, "flops_per_request": 10.0}

    def fake_load(job, persons):
        chiamate.append(("load", job.corpus, job.size))
        return FakeModel("core", ctx_len=64), FakeTok(), None

    def fake_core_job(corpus_name, size):
        return Job(stage="A_main", corpus=corpus_name, protocol="passi", exposures=1, size=size,
                   lr_mult=1.0, seed=0, n_persons=1)

    def fake_measure(job, persons, batch):
        chiamate.append(("pieno", job.corpus, batch))
        return {"latency_ms_mean": 5.0, "flops_per_answer": 100.0, "acc": 0.5}

    def fake_generate(model, idx, max_new_tokens, stop_id=None):
        assert idx.shape[1] + max_new_tokens - 1 <= model.ctx_len
        return torch.zeros(idx.shape[0], max_new_tokens, dtype=torch.long), \
            {"prefill_s": 0.001, "decode_s": 0.001, "n_steps": 0}

    monkeypatch.setattr(corpus, "load_persons", lambda n: persone)
    monkeypatch.setattr(translate, "train", fake_train)
    monkeypatch.setattr(translate, "run", fake_run)
    monkeypatch.setattr(infer, "load", fake_load)
    monkeypatch.setattr(infer, "core_job", fake_core_job)
    monkeypatch.setattr(infer, "measure_c", fake_measure)
    monkeypatch.setattr(kvgen, "generate", fake_generate)
    monkeypatch.setattr(e2e, "C_QUESTIONS", 4)
    monkeypatch.setattr(e2e, "C_SIZES", ("0.60M",))
    return chiamate


def test_run_e2e_scrive_una_riga_per_combinazione(tmp_path, finto):
    out = tmp_path / "e2e.csv"
    e2e.run_e2e(out)

    df = pd.read_csv(out)
    assert list(df.columns) == list(E2E_COLUMNS)
    assert len(df) == len(e2e.E2E_FORMATS) * len(e2e.C_BATCHES)
    assert set(df["format"]) == set(e2e.E2E_FORMATS)
    assert (df["n_questions"] == 4).all()
    assert (df["pieno_core_ms"] == 5.0).all()
    assert (df["pieno_flops"] == 100.0).all()
    assert df["pipeline_ms"].tolist() == pytest.approx([6.0] * len(df))
    assert df["speedup_vs_pieno"].tolist() == pytest.approx((df["pieno_core_ms"] / df["pipeline_ms"]).tolist())
    assert (df["pipeline_flops"] > 20.0).all()  # stripper 10 + traduttore 10 + core


def test_run_e2e_salta_combinazioni_gia_presenti(tmp_path, finto):
    out = tmp_path / "e2e.csv"
    gia = []
    for batch in e2e.C_BATCHES:
        riga = {c: 0.0 for c in E2E_COLUMNS}
        riga.update(format="telegrafico", size="0.60M", batch=batch)
        gia.append(riga)
    pd.DataFrame(gia, columns=list(E2E_COLUMNS)).to_csv(out, index=False)

    e2e.run_e2e(out)

    assert not any(c[0] == "load" and c[1] == "telegrafico" for c in finto)
    assert not any(c[0] == "train" and c[1] == "telegrafico" for c in finto)
    df = pd.read_csv(out)
    assert len(df) == len(gia) + len(e2e.C_BATCHES)
    assert df.duplicated(subset=["format", "size", "batch"]).sum() == 0


def test_core_ritaglia_il_prompt_a_sinistra(monkeypatch):
    lunghezze = []

    def fake_generate(model, idx, max_new_tokens, stop_id=None):
        assert idx.shape[1] + max_new_tokens - 1 <= model.ctx_len
        lunghezze.append(idx.shape[1])
        return torch.zeros(idx.shape[0], max_new_tokens, dtype=torch.long), \
            {"prefill_s": 0.0, "decode_s": 0.0, "n_steps": 0}

    monkeypatch.setattr(kvgen, "generate", fake_generate)
    core = FakeModel("core", ctx_len=8)

    outs, st = e2e._core(core, FakeTok(), ["x" * 20], [3], batch=1)

    assert lunghezze == [8 - 3 + 1]
    assert len(outs[0]) == 3
