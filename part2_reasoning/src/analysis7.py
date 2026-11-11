"""Analisi di v7: frontiera accuratezza-token per modello, scala 2 -> 5 e migliore condizione concisa (10, 12), grafici ed errori campione."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analysis6 import _md, _order, _val, comparisons, sample_errors6, summary6
from src.spec5 import MODELS
from src.spec7 import COMPARISONS7

SCALE7_COLUMNS = ("model", "acc_2", "acc_5", "gap_5_2", "best_concise", "acc_best", "gap_best_2", "tok_ratio_5_2",
                  "tok_ratio_best_2")
CONCISE7 = (10, 12)
RECOVERY7 = (1, 3, 9, 11)
FRONTIER7_COLUMNS = ("model", "cond", "label", "acc", "tok_mean", "lat_mean_s")


def pareto(summary):
    """Copia di summary con la colonna bool 'frontier': per modello, una condizione e' sulla frontiera se nessun'altra
    condizione dello stesso modello ha acc >= e tok_mean <= con almeno una disuguaglianza stretta."""
    out = summary.copy()
    models = out["model"].to_numpy()
    acc = out["acc"].to_numpy(dtype=float)
    tok = out["tok_mean"].to_numpy(dtype=float)
    frontier = np.zeros(len(out), dtype=bool)
    for model in pd.unique(models):
        pos = np.flatnonzero(models == model)
        for i in pos:
            dominated = any(acc[j] >= acc[i] and tok[j] <= tok[i] and (acc[j] > acc[i] or tok[j] < tok[i])
                            for j in pos if j != i)
            frontier[i] = not dominated
    out["frontier"] = frontier
    return out


def scale_trend(summary):
    """Per modello: accuratezza a 2 e 5 con il gap, la migliore tra 10 e 12 con il gap da 2, e i rapporti di token su 2.
    best_concise e' None (accuratezza e rapporti NaN) se mancano sia 10 sia 12; in parità vince la prima."""
    rows = []
    bests = []
    for model in _order(summary["model"].unique(), list(MODELS)):
        idx = summary[summary["model"] == model].set_index("cond")
        present = [c for c in CONCISE7 if c in idx.index]
        best = max(present, key=lambda c: _val(idx, c, "acc")) if present else None
        acc_best = _val(idx, best, "acc") if best is not None else float("nan")
        tok_best = _val(idx, best, "tok_mean") if best is not None else float("nan")
        acc_2 = _val(idx, 2, "acc")
        acc_5 = _val(idx, 5, "acc")
        tok_2 = _val(idx, 2, "tok_mean")
        bests.append(best)
        rows.append(dict(
            model=model, acc_2=acc_2, acc_5=acc_5, gap_5_2=acc_5 - acc_2, acc_best=acc_best,
            gap_best_2=acc_best - acc_2, tok_ratio_5_2=_val(idx, 5, "tok_mean") / tok_2,
            tok_ratio_best_2=tok_best / tok_2,
        ))
    df = pd.DataFrame(rows, columns=[c for c in SCALE7_COLUMNS if c != "best_concise"])
    # Colonna object: None resta None (con int sarebbe diventata float con NaN).
    df.insert(4, "best_concise", pd.Series(bests, index=df.index, dtype=object))
    return df


def plot7(summary, out_dir):
    """Un png per modello (frontier_{model}.png) con due pannelli come plot6; nel primo i punti di frontiera sono uniti
    da una linea in ordine di tok_mean. I punti con valori NaN sono saltati. Senza colonna frontier la calcola pareto."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "frontier" not in summary.columns:
        summary = pareto(summary)
    paths = []
    for model, g in summary.groupby("model", sort=False):
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        acc = g["acc"].to_numpy(dtype=float)
        lo = g["acc_lo"].to_numpy(dtype=float)
        hi = g["acc_hi"].to_numpy(dtype=float)
        axes = fig.subplots(1, 2)
        panels = (("tok_mean", "token medi per risposta"), ("lat_mean_s", "latenza media per risposta (s)"))
        for i, (ax, (xcol, xlabel)) in enumerate(zip(axes, panels)):
            x = g[xcol].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(acc) & np.isfinite(lo) & np.isfinite(hi)
            h = g[ok]
            xs = h[xcol].to_numpy(dtype=float)
            ys = h["acc"].to_numpy(dtype=float)
            yerr = np.vstack([ys - h["acc_lo"].to_numpy(dtype=float), h["acc_hi"].to_numpy(dtype=float) - ys])
            ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4)
            for xi, yi, cond, label in zip(xs, ys, h["cond"], h["label"]):
                ax.annotate(f"{cond} {label}", (xi, yi), textcoords="offset points", xytext=(5, 5))
            if i == 0:
                f = h[h["frontier"].to_numpy(dtype=bool)].sort_values("tok_mean")
                ax.plot(f["tok_mean"].to_numpy(dtype=float), f["acc"].to_numpy(dtype=float), "-", color="C1")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("accuracy (test)")
        fig.suptitle(model)
        path = out_dir / f"frontier_{model}.png"
        fig.savefig(path, dpi=120)
        paths.append(path)
    return paths


def write_all7(results_dir):
    """Scrive summary.csv (con frontier), comparisons.csv, scale.csv, errors_sample.jsonl (solo large, condizioni 10 e 12)
    e i grafici in plots/; ritorna il markdown."""
    results_dir = Path(results_dir)
    summary = pareto(summary6(results_dir))
    comp = comparisons(results_dir, COMPARISONS7)
    scale = scale_trend(summary)
    summary.to_csv(results_dir / "summary.csv", index=False)
    comp.to_csv(results_dir / "comparisons.csv", index=False)
    scale.to_csv(results_dir / "scale.csv", index=False)
    with open(results_dir / "errors_sample.jsonl", "w", encoding="utf-8") as f:
        for e in sample_errors6(results_dir, conds=CONCISE7):
            if e["model"] == "large":
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    plot7(summary, results_dir / "plots")
    front = summary.loc[summary["frontier"], list(FRONTIER7_COLUMNS)]
    recovery = summary.loc[summary["cond"].isin(RECOVERY7), ["model", "cond", "label", "frac_wrong_eq", "recovery"]]
    return ("## Riepilogo\n\n" + _md(summary) + "\n\n## Frontiera\n\n" + _md(front)
            + "\n\n## Confronti a coppie\n\n" + _md(comp) + "\n\n## Recupero\n\n" + _md(recovery)
            + "\n\n## Scala\n\n" + _md(scale) + "\n")
