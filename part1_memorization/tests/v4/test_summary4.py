"""Test di src/v4/summary4.py su dati sintetici: 2 o 4 formati, con e senza F_forms."""
from __future__ import annotations

import math

import pandas as pd

from src.v4.analysis4 import IC, IP, TC, TL
from src.v4.spec4 import F_SIZES, FORMS_COLUMNS
from src.v4.summary4 import write_all

SHIFT = {IP: 0.0, IC: 0.10, TL: 0.20, TC: 0.35}   # traslazione in log10 dei parametri per formato
GRID = (3.5, 4.0, 4.5, 5.0)                       # log10 dei parametri non-embedding (accuratezza logistica)
SIZES = tuple(F_SIZES[:4])
SEEDS = (0, 1, 2)
SLOPE = 4.0


def _main(formats) -> pd.DataFrame:
    rows = []
    for fmt in formats:
        for size, x in zip(SIZES, GRID):
            params = 10 ** (x + SHIFT[fmt])
            for seed in SEEDS:
                acc = 1 / (1 + math.exp(-SLOPE * (x - 4.0))) + (seed - 1) * 0.01
                rows.append({"format": fmt, "size": size, "seed": seed, "lr_mult": 1.0,
                             "params_nonemb": params, "params_total": 1.5 * params,
                             "train_time_s": 1e-3 * 1.5 * params, "tokens_per_bio": 40.0,
                             "final_loss": 0.5, "acc_mean": acc, "acc_nome": acc})
    return pd.DataFrame(rows)


def _forms(formats) -> pd.DataFrame:
    rows = []
    for fmt in formats:
        for attr, bits in (("nome", 2.0), ("corso", 3.0)):
            rows.append({"format": fmt, "attr": attr, "n_forms": 6, "n_forms_core": 4,
                         "entropy_bits": bits + SHIFT[fmt]})
    return pd.DataFrame(rows, columns=FORMS_COLUMNS)


def _names(tmp_path, *names):
    return [tmp_path / n for n in names]


def test_two_formats_without_2x2(tmp_path):
    _main([IP, IC]).to_csv(tmp_path / "F_main.csv", index=False)
    _forms([IP, IC]).to_csv(tmp_path / "F_forms.csv", index=False)

    text = write_all(tmp_path)

    for header in ("## Stato", "## Forme", "## Soglie", "## Tempo di training al 90%",
                   "## Soglie per attributo", "## Accuratezza per attributo"):
        assert header in text, header
    assert "## 2x2" not in text
    for path in _names(tmp_path, "summary_v4.md", "F_thresholds.csv", "F_2x2.csv", "F_attr.csv",
                       "acc_vs_params_v4.png", "entropy_vs_params.png"):
        assert path.is_file(), path.name
    assert (tmp_path / "summary_v4.md").read_text(encoding="utf-8") == text


def test_four_formats_with_2x2(tmp_path):
    _main(list(SHIFT)).to_csv(tmp_path / "F_main.csv", index=False)

    text = write_all(tmp_path)

    assert "## 2x2 per livello" in text
    assert "share_reg" in text
    assert "%" in text
    dec = pd.read_csv(tmp_path / "F_2x2.csv")
    assert set(dec["level"]) == {0.5, 0.9}
    assert not (tmp_path / "entropy_vs_params.png").exists()   # senza F_forms non c'e' l'asse dei bit


def test_optional_sections_without_main(tmp_path):
    pd.DataFrame({"format": [IC, IC], "size": ["0.38M", "0.38M"], "lr_mult": [1.0, 2.0],
                  "acc_mean": [0.5, 0.6], "final_loss": [0.4, 0.3]}).to_csv(tmp_path / "F_lr.csv", index=False)
    pd.DataFrame({"size": ["0.38M"], "bench": [0.9]}).to_csv(tmp_path / "align_bench.csv", index=False)
    pd.DataFrame({"size": ["0.38M"], "control": [0.8]}).to_csv(tmp_path / "align_control.csv", index=False)

    text = write_all(tmp_path)

    for header in ("## Verifica lr", "## Allineamento: align_bench", "## Allineamento: align_control"):
        assert header in text, header
    assert "## Soglie" not in text
    assert not (tmp_path / "acc_vs_params_v4.png").exists()


def test_empty_dir_does_not_crash(tmp_path):
    (tmp_path / "F_main.csv").write_text("", encoding="utf-8")

    text = write_all(tmp_path)

    assert text.startswith("# Riepilogo esperimento v4")
    assert "## " not in text
    assert (tmp_path / "summary_v4.md").is_file()
