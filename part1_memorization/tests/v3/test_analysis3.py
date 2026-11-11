"""Test di analysis3 su CSV sintetici in tmp; V2_RESULTS puntato a una cartella inesistente."""
from __future__ import annotations

import pandas as pd

from src.v2.spec import RESULT_COLUMNS_V2
from src.v3 import analysis3
from src.v3.spec3 import D_SIZES

PARAMS = dict(zip(D_SIZES, (1e5, 2e5, 4e5, 8e5, 1.6e6, 3.2e6)))


def _rows(corpus: str, accs, vocab_size: int, seeds=(0, 1)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for size, acc in zip(D_SIZES, accs):
            row = dict.fromkeys(RESULT_COLUMNS_V2, 0.0)
            row.update(corpus=corpus, size=size, seed=seed, acc_mean=acc, vocab=vocab_size,
                       params_nonemb=PARAMS[size], params_total=PARAMS[size],
                       train_time_s=PARAMS[size] / 1e4)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS_V2))


def test_write_all_picks_best_vocab(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis3, "V2_RESULTS", tmp_path / "v2")  # assente: nessun riferimento bpe4096
    d = pd.concat([
        _rows("italiano_pieno@bpe512", [0.10, 0.30, 0.60, 0.85, 0.95, 0.97], 512),
        _rows("italiano_pieno@bpe1024", [0.20, 0.50, 0.80, 0.92, 0.95, 0.97], 1024),
        _rows("italiano_pieno@char", [0.05, 0.10, 0.20, 0.30, 0.35, 0.40], 100),
        _rows("telegrafico@bpe512", [0.20, 0.40, 0.60, 0.80, 0.90, 0.95], 512),
    ], ignore_index=True)
    d.to_csv(tmp_path / "D_main.csv", index=False)

    md = analysis3.write_all(tmp_path)

    assert (tmp_path / "summary_v3.md").is_file()
    assert (tmp_path / "acc_vs_params_total.png").is_file()
    assert (tmp_path / "tok_bio_vs_vocab.png").is_file()
    best = md.split("## D: vocabolario migliore")[1].split("## ")[0].splitlines()
    assert any(line.startswith("| italiano_pieno | bpe1024 | 90% |") for line in best)
    assert any(line.startswith("| telegrafico | bpe512 | 90% |") for line in best)


def test_write_all_missing_csv_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis3, "V2_RESULTS", tmp_path / "v2")
    md = analysis3.write_all(tmp_path)
    assert md.startswith("# Riepilogo esperimento v3")
    assert (tmp_path / "summary_v3.md").is_file()
    assert "## D:" not in md


def test_write_all_e_main_without_v2_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis3, "V2_RESULTS", tmp_path / "v2")
    _rows("telegrafico_auto", [0.2, 0.4, 0.6, 0.8, 0.9, 0.95], 0).to_csv(tmp_path / "E_main.csv", index=False)
    md = analysis3.write_all(tmp_path)
    assert "## E: telegrafico_auto vs telegrafico" not in md
    assert "## E: telegrafico_lista" not in md
