"""Test di src/report.py su un CSV sintetico."""
import math

import numpy as np
import pandas as pd
import pytest

from src.config import ATTRS, FORMATS, RESULT_COLUMNS
from src.report import thresholds, write_all

SIZES = {"0.25M": 1e5, "0.5M": 1e6, "1M": 1e7, "2M": 1e8}
TOKENS_PER_BIO = {"italiano_pieno": 60, "telegrafico": 40, "interlingua": 30}
CENTER = {"italiano_pieno": 6.0, "telegrafico": 5.5, "interlingua": 5.0}


def _synthetic_csv(path):
    rows = []
    for fmt in FORMATS:
        for i, (size, params) in enumerate(SIZES.items()):
            x = np.log10(params)
            acc = 1 / (1 + np.exp(-4 * (x - CENTER[fmt])))  # sigmoide monotona
            for seed in (0, 1):
                row = {c: 0.0 for c in RESULT_COLUMNS}
                row.update({
                    "tag": "quick", "format": fmt, "size": size, "seed": seed,
                    "n_persons": 100, "exposures": 1, "n_layer": 4, "n_embd": 72,
                    "n_head": 4, "params_nonemb": params, "params_total": 1.2 * params,
                    "ctx_len": 64, "tokens_per_bio": TOKENS_PER_BIO[fmt],
                    "train_tokens": 1e6 * (i + 1), "steps": 10,
                    "flops": 6 * 1.2 * params * 1e6 * (i + 1),
                    "train_time_s": 10.0 * (i + 1), "tokens_per_s": 1000.0,
                    "peak_vram_mb": 100.0, "final_loss": 1.0, "eval_time_s": 1.0,
                    "acc_mean": acc,
                })
                for a in ATTRS:
                    row[f"acc_{a}"] = acc
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RESULT_COLUMNS)).to_csv(path, index=False)
    return path


def _two_format_df(curves):
    """curves: {format: [acc at 1e5, 1e6, 1e7, 1e8]}, due seed identici."""
    rows = []
    for fmt, accs in curves.items():
        for size, params, acc in zip(SIZES, SIZES.values(), accs):
            for seed in (0, 1):
                rows.append({"format": fmt, "size": size, "seed": seed,
                             "params_nonemb": params, "acc_mean": acc,
                             "tokens_per_bio": 50})
    return pd.DataFrame(rows)


def test_threshold_interpolation_matches_hand_value():
    # 0.5 cade tra 1e6 (0.4) e 1e7 (0.6): t = 0.5, log10 = 6 + 0.5 -> 10**6.5
    df = _two_format_df({"italiano_pieno": [0.2, 0.4, 0.6, 0.95]})
    thr = thresholds(df, levels=(0.5, 0.9))
    row05 = thr[(thr.format == "italiano_pieno") & (thr.level == 0.5)].iloc[0]
    assert row05.params == pytest.approx(10 ** 6.5, rel=1e-9)
    assert not row05.lower_bound
    # 0.9 cade tra 1e7 (0.6) e 1e8 (0.95): t = 0.3 / 0.35
    row09 = thr[(thr.format == "italiano_pieno") & (thr.level == 0.9)].iloc[0]
    assert row09.params == pytest.approx(10 ** (7 + 0.3 / 0.35), rel=1e-9)


def test_threshold_ratio_vs_pieno_and_lower_bound():
    df = _two_format_df({
        "italiano_pieno": [0.2, 0.4, 0.6, 0.95],
        "telegrafico": [0.6, 0.7, 0.8, 0.95],  # gia' sopra 0.5 alla taglia minima
    })
    thr = thresholds(df, levels=(0.5, 0.9))
    tele = thr[(thr.format == "telegrafico") & (thr.level == 0.5)].iloc[0]
    assert tele.lower_bound
    assert tele.params == pytest.approx(1e5)
    pieno = thr[(thr.format == "italiano_pieno") & (thr.level == 0.5)].iloc[0]
    assert pieno.ratio_vs_pieno == pytest.approx(1.0)
    assert tele.ratio_vs_pieno == pytest.approx(1e5 / 10 ** 6.5)


def test_threshold_nan_when_never_reached():
    df = _two_format_df({"interlingua": [0.1, 0.2, 0.3, 0.4]})
    thr = thresholds(df, levels=(0.5, 0.9))
    row = thr[(thr.format == "interlingua") & (thr.level == 0.5)].iloc[0]
    assert math.isnan(row.params)
    assert not row.lower_bound


def test_write_all_creates_files(tmp_path):
    csv = _synthetic_csv(tmp_path / "results.csv")
    out = tmp_path / "out"
    md = write_all(csv, out)
    assert (out / "accuracy_vs_params.png").is_file()
    assert (out / "compute.png").is_file()
    assert (out / "summary.md").is_file()
    assert md == (out / "summary.md").read_text(encoding="utf-8")
    assert "Soglie di accuratezza" in md
    assert "italiano_pieno" in md
