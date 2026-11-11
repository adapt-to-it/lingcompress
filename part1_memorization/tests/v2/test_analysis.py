"""Test dell'analisi v2 su CSV sintetici."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v2.analysis import _ratio, thresholds, write_all
from src.v2.spec import RESULT_COLUMNS_V2

SIZES3 = (1e5, 1e6, 1e7)
SIZES4 = (1.5e5, 3.1e5, 1.49e6, 4e6)


def _row(stage, corpus, size, params, acc, seed, exposures=200, protocol="passi", lr=1.0):
    row = {c: np.nan for c in RESULT_COLUMNS_V2}
    row.update(stage=stage, corpus=corpus, protocol=protocol, exposures=exposures, size=size,
               lr_mult=lr, seed=seed, n_persons=20000, eval_persons=5000,
               params_nonemb=params, params_total=1.5 * params, acc_mean=acc,
               cross_mean=acc / 2, cross_lenient_mean=acc * 0.6, bits_per_param=acc * 2,
               steady_tokens_per_s=1000.0, steady_facts_per_s=10.0, train_time_s=60.0,
               final_loss=1.0)
    return row


def _curve(corpus, accs, stage="A_main"):
    """Tre taglie (1e5, 1e6, 1e7), due seed con la stessa accuratezza: media esatta uguale ad accs."""
    return [_row(stage, corpus, f"s{i}", p, a, seed)
            for i, (p, a) in enumerate(zip(SIZES3, accs)) for seed in (0, 1)]


def _sig(params, center, k=2.0):
    return 1 / (1 + math.exp(-k * (math.log10(params) - center)))


def _frame(stage, specs, exposures=200, protocol="passi", lr=1.0):
    """specs: [(corpus, centro della sigmoide)]; due seed simmetrici (media = sigmoide)."""
    rows = []
    for corpus, center in specs:
        for params in SIZES4:
            for seed, delta in ((0, 0.01), (1, -0.01)):
                acc = _sig(params, center) + delta
                rows.append(_row(stage, corpus, str(params), params, acc, seed, exposures, protocol, lr))
    return pd.DataFrame(rows)


def _write_v1(parent: Path) -> None:
    for name, shift in (("results.csv", 0.0), ("results_ctrl_steps.csv", 0.3)):
        rows = []
        for size, params in (("a", 1e5), ("b", 1e6), ("c", 1e7)):
            for seed in (0, 1):
                rows.append({"format": "italiano_pieno", "size": size, "seed": seed,
                             "params_nonemb": params, "acc_mean": min(0.2 + shift + (params == 1e6) * 0.5, 1.0)})
        pd.DataFrame(rows).to_csv(parent / name, index=False)


def _write_stages(results: Path) -> None:
    frames = {
        "A_lr": pd.concat([_frame("A_lr", [("italiano_pieno", 5.5)], lr=lr) for lr in (0.5, 1.0, 2.0)]),
        "A_main": _frame("A_main", [("italiano_pieno", 5.5), ("telegrafico", 5.2), ("interlingua", 5.0)]),
        "A_exp": pd.concat([_frame("A_exp", [("italiano_pieno", 5.5)], exposures=e, protocol="esposizioni")
                            for e in (50, 100)]),
        "A_esp": _frame("A_esp", [("italiano_pieno", 5.4), ("telegrafico", 5.1)], exposures=400,
                        protocol="esposizioni"),
        "A_shared": _frame("A_shared", [("italiano_pieno@shared", 5.5), ("interlingua@shared", 5.0)]),
        "A_abl": _frame("A_abl", [("interlingua_v1enc", 4.8), ("italiano_pieno@nodigits", 5.3)]),
        "B_main": _frame("B_main", [("multi_pieno", 5.0), ("multi_bridge", 5.1)]),
    }
    for name, df in frames.items():
        df.to_csv(results / f"{name}.csv", index=False)


def test_threshold_interpolation_by_hand():
    df = pd.DataFrame(_curve("A", [0.2, 0.8, 0.95]))
    thr = thresholds(df, ["corpus"])
    t50 = thr[thr["level"] == 0.5].iloc[0]
    t90 = thr[thr["level"] == 0.9].iloc[0]
    # 0.5 tra 1e5 (0.2) e 1e6 (0.8): t = 0.5, quindi log10 = 5.5
    assert t50["params_nonemb"] == pytest.approx(10 ** 5.5, rel=1e-9)
    # 0.9 tra 1e6 (0.8) e 1e7 (0.95): t = 0.1 / 0.15, quindi log10 = 6 + 2/3
    assert t90["params_nonemb"] == pytest.approx(10 ** (6 + 2 / 3), rel=1e-9)
    # params_total interpolato allo stesso modo: 1.5 * 10^5.5
    assert t50["params_total"] == pytest.approx(1.5 * 10 ** 5.5, rel=1e-9)
    assert not t50["lower_bound"]
    assert t50["n_seeds"] == 2


def test_lower_bound_and_unreached():
    thr = thresholds(pd.DataFrame(_curve("L", [0.6, 0.7, 0.8])), ["corpus"])
    lo = thr[thr["level"] == 0.5].iloc[0]
    hi = thr[thr["level"] == 0.9].iloc[0]
    assert bool(lo["lower_bound"]) and lo["params_nonemb"] == pytest.approx(1e5)
    assert not bool(hi["lower_bound"]) and math.isnan(hi["params_nonemb"])


def test_ratio_vs_base():
    df = pd.DataFrame(_curve("italiano_pieno", [0.2, 0.8, 0.95]) + _curve("X", [0.2, 0.5, 0.9]))
    thr = thresholds(df, ["corpus"])
    rat = _ratio(thr, "italiano_pieno", [])
    x50 = thr.index[(thr["corpus"] == "X") & (thr["level"] == 0.5)][0]
    # X raggiunge 0.5 esattamente a 1e6, italiano_pieno a 10^5.5: rapporto 10^0.5
    assert rat.loc[x50, "ratio_nonemb"] == pytest.approx(10 ** 0.5, rel=1e-9)
    base50 = thr.index[(thr["corpus"] == "italiano_pieno") & (thr["level"] == 0.5)][0]
    assert rat.loc[base50, "ratio_nonemb"] == pytest.approx(1.0)


def test_write_all_all_stages(tmp_path):
    results = tmp_path / "v2"
    results.mkdir()
    _write_v1(tmp_path)
    _write_stages(results)
    md = write_all(results)
    for name in ("acc_vs_params_passi.png", "bits_per_param_vs_exposures.png",
                 "multilingua.png", "cross_vs_size.png", "summary_v2.md"):
        assert (results / name).is_file(), name
    assert (results / "summary_v2.md").read_text(encoding="utf-8") == md
    assert "Confronto con v1" in md
    assert "Ablazioni" in md
    assert "Learning rate" in md


def test_write_all_only_main(tmp_path):
    results = tmp_path / "v2"
    results.mkdir()
    pd.DataFrame(_curve("italiano_pieno", [0.2, 0.8, 0.95])).to_csv(results / "A_main.csv", index=False)
    md = write_all(results)
    assert (results / "acc_vs_params_passi.png").is_file()
    assert (results / "bits_per_param_vs_exposures.png").is_file()
    assert (results / "summary_v2.md").is_file()
    assert "Soglie di accuratezza" in md
    assert not (results / "cross_vs_size.png").exists()
    assert not (results / "multilingua.png").exists()
