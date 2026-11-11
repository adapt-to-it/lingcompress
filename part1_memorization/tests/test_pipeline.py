import time

import pytest

import src.pipeline as pipeline
from src.config import ATTRS, EOS, MODEL_SIZES, PAD, RESULT_COLUMNS, ExpConfig
from src.data import epoch_texts, generate_persons

FMT = "italiano_pieno"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def make_cfg(**kw):
    base = dict(n_persons=60, exposures=2, batch_size=8, sizes=("0.25M",),
                formats=(FMT,), seeds=(0,), tag="test")
    base.update(kw)
    return ExpConfig(**base)


def test_run_one_returns_all_columns(dirs):
    cfg = make_cfg()
    persone = generate_persons(cfg.n_persons, cfg.data_seed)
    riga = pipeline.run_one(persone, FMT, MODEL_SIZES["0.25M"], 0, cfg)
    assert tuple(riga) == RESULT_COLUMNS
    assert 0.0 <= riga["acc_mean"] <= 1.0
    for attr in ATTRS:
        assert 0.0 <= riga[f"acc_{attr}"] <= 1.0
    assert riga["steps"] > 0
    assert riga["train_tokens"] > 0


def test_packed_rows_start_with_eos_and_keep_bios_whole(dirs):
    cfg = make_cfg()
    persone = generate_persons(cfg.n_persons, cfg.data_seed)
    tok = pipeline.train_tokenizer(persone, FMT, cfg)
    testi = epoch_texts(persone, FMT, 0, cfg.data_seed)
    lunghezza_max = max(len(e.ids) for e in tok.encode_batch(testi))
    ctx = ((2 * (lunghezza_max + 1) + 63) // 64) * 64
    righe = pipeline.epoch_tokens(persone, FMT, 0, tok, ctx, cfg)
    eos, pad = tok.token_to_id(EOS), tok.token_to_id(PAD)
    biografie = set(testi)

    assert righe.shape[1] == ctx
    assert 0 < len(righe) < len(persone)  # piu' biografie per riga
    for riga in righe:
        ids = [int(t) for t in riga if t != pad]
        assert ids[0] == eos and ids[-1] == eos
        parti = [[]]
        for t in ids:
            if t == eos:
                parti.append([])
            else:
                parti[-1].append(t)
        assert parti[0] == [] and parti[-1] == []
        for bio in parti[1:-1]:
            assert tok.decode(bio) in biografie


def test_resume_after_completion_returns_stored_result(dirs):
    cfg = make_cfg()
    persone = generate_persons(cfg.n_persons, cfg.data_seed)
    t0 = time.perf_counter()
    primo = pipeline.run_one(persone, FMT, MODEL_SIZES["0.25M"], 0, cfg)
    durata_primo = time.perf_counter() - t0

    t0 = time.perf_counter()
    secondo = pipeline.run_one(persone, FMT, MODEL_SIZES["0.25M"], 0, cfg)
    durata_secondo = time.perf_counter() - t0

    assert secondo == primo
    assert durata_secondo < durata_primo
