"""Test di src/v4/analysis4.py su dati sintetici: 4 formati, 4 taglie, 3 seed."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.v4.analysis4 import attr_thresholds, decompose_2x2, thresholds_ci, train_time_at

IP = "italiano_pieno"
IC = "italiano_canonico"
TL = "telegrafico_lista"
TC = "telegrafico_canonico"
SHIFT = {IP: 0.0, IC: 0.10, TL: 0.20, TC: 0.35}   # traslazione in log10 dei parametri per formato
GRID = (3.5, 4.0, 4.5, 5.0)                       # log10 dei parametri comuni (accuratezza logistica)
SIZES = ("S1", "S2", "S3", "S4")
SEEDS = (0, 1, 2)
SLOPE = 4.0
N_BOOT = 30
QUANTITIES = ("reg_full", "reg_strip", "strip_vario", "strip_canon", "interaction", "total",
              "share_reg", "share_strip")

# rapporti attesi: le soglie di ogni formato sono traslate di SHIFT, quindi i rapporti sono potenze di 10
KNOWN = {
    "reg_full": 10 ** (SHIFT[IC] - SHIFT[IP]),
    "reg_strip": 10 ** (SHIFT[TC] - SHIFT[TL]),
    "strip_vario": 10 ** (SHIFT[TL] - SHIFT[IP]),
    "strip_canon": 10 ** (SHIFT[TC] - SHIFT[IC]),
    "interaction": 10 ** ((SHIFT[TC] - SHIFT[TL]) - (SHIFT[IC] - SHIFT[IP])),
    "total": 10 ** (SHIFT[TC] - SHIFT[IP]),
    "share_reg": ((SHIFT[IC] - SHIFT[IP]) + (SHIFT[TC] - SHIFT[TL])) / 2 / (SHIFT[TC] - SHIFT[IP]),
}


def _synthetic(scale: dict | None = None) -> pd.DataFrame:
    """Accuratezza logistica in log10 dei parametri comuni; ogni formato e' traslato di SHIFT.
    Il rumore per seed ha media zero: la media sui seed e' la curva esatta."""
    scale = scale or {}
    rows = []
    for fmt, shift in SHIFT.items():
        for size, x in zip(SIZES, GRID):
            params = 10 ** (x + shift)
            for seed in SEEDS:
                acc = scale.get(fmt, 1.0) / (1 + math.exp(-SLOPE * (x - 4.0))) + (seed - 1) * 0.01
                rows.append({"format": fmt, "size": size, "seed": seed,
                             "params_nonemb": params, "params_total": 1.5 * params,
                             "train_time_s": 1e-3 * 1.5 * params,
                             "acc_mean": acc, "acc_nome": acc})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return _synthetic()


@pytest.fixture(scope="module")
def fitted(synthetic):
    return thresholds_ci(synthetic, n_boot=N_BOOT)


def _row(dec: pd.DataFrame, level: float) -> pd.Series:
    return dec[dec["level"] == level].iloc[0]


def test_thresholds_ci_shape(fitted):
    thr, draws = fitted
    assert list(draws.columns) == ["draw", "format", "level", "params_total"]
    assert len(draws) == N_BOOT * 4 * 2
    assert len(thr) == 4 * 2
    for col in ("params_nonemb", "params_total", "seed_min", "seed_max", "lo", "hi", "lower_bound", "n_seeds"):
        assert col in thr.columns
    assert (thr["n_seeds"] == len(SEEDS)).all()


def test_thresholds_ci_seed_range_and_interval(fitted):
    thr, _ = fitted
    ok = thr.dropna(subset=["params_total", "lo", "hi"])
    assert len(ok) == len(thr)
    assert (ok["seed_min"] <= ok["seed_max"]).all()
    assert (ok["lo"] <= ok["hi"]).all()


def test_thresholds_ci_is_deterministic(synthetic, fitted):
    _, draws = fitted
    _, again = thresholds_ci(synthetic, n_boot=N_BOOT)
    pd.testing.assert_frame_equal(draws, again)


@pytest.mark.parametrize("level", [0.5, 0.9])
def test_known_ratios_within_5_percent(fitted, level):
    thr, draws = fitted
    row = _row(decompose_2x2(thr, draws), level)
    for q, expected in KNOWN.items():
        assert row[q] == pytest.approx(expected, rel=0.05), q


def test_share_reg_plus_share_strip_is_one(fitted):
    thr, draws = fitted
    dec = decompose_2x2(thr, draws)
    assert (dec["share_reg"] + dec["share_strip"]).to_numpy() == pytest.approx(1.0)


def test_point_within_bootstrap_interval(fitted):
    thr, draws = fitted
    dec = decompose_2x2(thr, draws)
    for _, row in dec.iterrows():
        for q in QUANTITIES:
            if math.isfinite(row[q]):
                assert row[f"{q}_lo"] <= row[q] <= row[f"{q}_hi"], q


def test_nan_when_format_never_reaches_90(synthetic):
    # telegrafico_canonico non supera 0.8 di accuratezza: nessuna soglia al 90%
    capped = _synthetic({TC: 0.8})
    thr, draws = thresholds_ci(capped, n_boot=N_BOOT)
    t90 = thr[(thr["format"] == TC) & (thr["level"] == 0.9)]
    assert t90["params_total"].isna().all()
    assert t90["lo"].isna().all() and t90["hi"].isna().all()
    t50 = thr[(thr["format"] == TC) & (thr["level"] == 0.5)]
    assert t50["params_total"].notna().all()

    dec = decompose_2x2(thr, draws)
    row90 = _row(dec, 0.9)
    assert math.isnan(row90["total"])
    assert math.isnan(row90["share_reg"]) and math.isnan(row90["share_strip"])
    assert math.isfinite(row90["reg_full"])
    assert math.isfinite(_row(dec, 0.5)["total"])


def test_attr_thresholds_ratio_reg(synthetic):
    at = attr_thresholds(synthetic)
    assert set(at.columns) == {"format", "attr", "level", "params_total", "ratio_reg"}
    assert set(at["attr"]) == {"nome"}
    at90 = at[at["level"] == 0.9].set_index("format")["ratio_reg"]
    assert at90[IP] == 1.0 and at90[TL] == 1.0
    assert at90[IC] == pytest.approx(10 ** (SHIFT[IC] - SHIFT[IP]), rel=0.05)
    assert at90[TC] == pytest.approx(10 ** (SHIFT[TC] - SHIFT[TL]), rel=0.05)


def test_train_time_at_power_law(synthetic):
    thr, _ = thresholds_ci(synthetic, n_boot=0)
    tt = train_time_at(synthetic, thr, level=0.9).set_index("format")
    assert len(tt) == 4
    t90 = thr[thr["level"] == 0.9].set_index("format")["params_total"]
    for fmt in SHIFT:
        assert tt.loc[fmt, "params_total"] == pytest.approx(t90[fmt])
        assert tt.loc[fmt, "train_time_s"] == pytest.approx(1e-3 * t90[fmt], rel=1e-6)


def test_train_time_at_nan_outside_range(synthetic):
    thr = pd.DataFrame({"format": [IP, IC], "level": [0.9, 0.9],
                        "params_total": [1e12, math.nan]})
    tt = train_time_at(synthetic, thr, level=0.9)
    assert tt["train_time_s"].isna().all()
