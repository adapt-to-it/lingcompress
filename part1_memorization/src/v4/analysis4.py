"""Analisi v4: soglie con intervallo bootstrap, scomposizione 2x2 dei rapporti e tempo di addestramento. Funzioni pure."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.v2.analysis import thresholds
from src.v4.spec4 import VARIO

IP = "italiano_pieno"
IC = "italiano_canonico"
TL = "telegrafico_lista"
TC = "telegrafico_canonico"
REG_BASE = {IC: IP, TC: TL}   # canonico -> vario di riferimento per la regolarita'
QUANTITIES = ("reg_full", "reg_strip", "strip_vario", "strip_canon", "interaction", "total",
              "share_reg", "share_strip")
CI_PCT = (2.5, 97.5)


def _div(a: float, b: float) -> float:
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else math.nan


def _ci(vals) -> tuple[float, float]:
    """Percentili 2.5 e 97.5 dei valori finiti; nan se non ce ne sono."""
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return math.nan, math.nan
    lo, hi = np.percentile(arr, CI_PCT)
    return float(lo), float(hi)


def _resample_seeds(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Per ogni formato e taglia ricampiona i seed con reinserimento; ogni seed scelto porta le sue righe."""
    parts = []
    for _, g in df.groupby(["format", "size"], sort=False):
        by_seed = dict(tuple(g.groupby("seed")))
        seeds = list(by_seed)
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        parts.extend(by_seed[s] for s in pick)
    return pd.concat(parts, ignore_index=True)


def _share_reg(reg_full: float, reg_strip: float, total: float) -> float:
    """Quota della regolarita' nel log del totale: media dei log dei due rapporti di regolarita' diviso log(total)."""
    if not all(math.isfinite(v) and v > 0 for v in (reg_full, reg_strip, total)):
        return math.nan
    log_total = math.log(total)
    if log_total == 0:
        return math.nan
    return (math.log(reg_full) + math.log(reg_strip)) / 2 / log_total


def _ratios(pt: dict) -> dict:
    """Rapporti della 2x2 su params_total; pt: formato -> params_total a un livello."""
    ip, ic, tl, tc = (pt.get(f, math.nan) for f in (IP, IC, TL, TC))
    reg_full = _div(ic, ip)
    reg_strip = _div(tc, tl)
    total = _div(tc, ip)
    share_reg = _share_reg(reg_full, reg_strip, total)
    return {
        "reg_full": reg_full,
        "reg_strip": reg_strip,
        "strip_vario": _div(tl, ip),
        "strip_canon": _div(tc, ic),
        "interaction": _div(reg_strip, reg_full),
        "total": total,
        "share_reg": share_reg,
        "share_strip": 1.0 - share_reg,
    }


def thresholds_ci(df: pd.DataFrame, levels=(0.5, 0.9), n_boot: int = 1000, seed: int = 0):
    """Soglie per formato e livello con intervallo bootstrap (ricampionamento dei seed). Ritorna (thr, draws)."""
    levels = tuple(levels)
    thr = thresholds(df, ["format"], levels)
    single = pd.concat([thresholds(sub, ["format"], levels) for _, sub in df.groupby("seed")],
                       ignore_index=True)
    spread = single.groupby(["format", "level"], as_index=False).agg(
        seed_min=("params_total", "min"), seed_max=("params_total", "max"))
    thr = thr.merge(spread, on=["format", "level"], how="left")

    rng = np.random.default_rng(seed)
    records = []
    for b in range(n_boot):
        res = thresholds(_resample_seeds(df, rng), ["format"], levels)
        records.extend((b, f, lvl, p) for f, lvl, p in res[["format", "level", "params_total"]].itertuples(index=False))
    draws = pd.DataFrame(records, columns=["draw", "format", "level", "params_total"])

    bounds = {k: _ci(g["params_total"]) for k, g in draws.groupby(["format", "level"])}
    lohi = [bounds.get((f, lvl), (math.nan, math.nan)) for f, lvl in zip(thr["format"], thr["level"])]
    thr["lo"] = [lo for lo, _ in lohi]
    thr["hi"] = [hi for _, hi in lohi]
    thr = thr[["format", "level", "params_nonemb", "params_total", "seed_min", "seed_max",
               "lo", "hi", "lower_bound", "n_seeds"]].reset_index(drop=True)
    return thr, draws


def decompose_2x2(thr: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Una riga per livello: rapporti della 2x2 su params_total, con _lo e _hi dai draw bootstrap."""
    point = {lvl: dict(zip(g["format"], g["params_total"])) for lvl, g in thr.groupby("level")}
    per: dict = {}
    for (_, lvl), g in draws.groupby(["draw", "level"]):
        per.setdefault(lvl, []).append(_ratios(dict(zip(g["format"], g["params_total"]))))
    rows = []
    for lvl, pt in point.items():
        ratios = _ratios(pt)
        row = {"level": lvl}
        for q in QUANTITIES:
            lo, hi = _ci(r[q] for r in per.get(lvl, []))
            row[q] = ratios[q]
            row[f"{q}_lo"] = lo
            row[f"{q}_hi"] = hi
        rows.append(row)
    cols = ["level", *(c for q in QUANTITIES for c in (q, f"{q}_lo", f"{q}_hi"))]
    return pd.DataFrame(rows, columns=cols)


def _train_time_at(df: pd.DataFrame, fmt: str, pt: float) -> float:
    """train_time_s medio interpolato in log-log su params_total medio per taglia, a params_total pt."""
    if not math.isfinite(pt) or pt <= 0:
        return math.nan
    per = df[df["format"] == fmt].groupby("size", as_index=False).agg(
        pt=("params_total", "mean"), tt=("train_time_s", "mean")).dropna().sort_values("pt")
    if not len(per):
        return math.nan
    x = np.log10(per["pt"].to_numpy(dtype=float))
    y = np.log10(per["tt"].to_numpy(dtype=float))
    t = math.log10(pt)
    if not (x[0] <= t <= x[-1]):
        return math.nan
    return float(10 ** np.interp(t, x, y))


def train_time_at(df: pd.DataFrame, thr: pd.DataFrame, level: float = 0.9) -> pd.DataFrame:
    """Per formato: train_time_s alla soglia level (nan fuori dal range delle taglie o senza soglia)."""
    rows = []
    for r in thr[thr["level"] == level].itertuples(index=False):
        rows.append({"format": r.format, "params_total": r.params_total,
                     "train_time_s": _train_time_at(df, r.format, r.params_total)})
    return pd.DataFrame(rows, columns=["format", "params_total", "train_time_s"])


def attr_thresholds(df: pd.DataFrame, levels=(0.5, 0.9)) -> pd.DataFrame:
    """Soglie per formato, attributo e livello (metrica acc_<attr>), con ratio_reg: canonico / vario."""
    attrs = [c[len("acc_"):] for c in df.columns if c.startswith("acc_") and c != "acc_mean"]
    parts = []
    for attr in attrs:
        thr = thresholds(df, ["format"], levels, metric=f"acc_{attr}")
        parts.append(thr.assign(attr=attr)[["format", "attr", "level", "params_total"]])
    cols = ["format", "attr", "level", "params_total"]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    pt = {(a, lvl, f): p for f, a, lvl, p in zip(out["format"], out["attr"], out["level"], out["params_total"])}
    ratio = []
    for f, a, lvl in zip(out["format"], out["attr"], out["level"]):
        if f in REG_BASE:
            ratio.append(_div(pt[(a, lvl, f)], pt.get((a, lvl, REG_BASE[f]), math.nan)))
        elif VARIO.get(f):
            ratio.append(1.0)
        else:
            ratio.append(math.nan)
    out["ratio_reg"] = ratio
    return out
