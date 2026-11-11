"""Grafici e tabelle riassuntive da results.csv."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # niente finestre: solo file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ATTRS, FORMATS  # noqa: E402

BASE_FORMAT = "italiano_pieno"  # formato di riferimento per i rapporti


def _formats(df: pd.DataFrame) -> list[str]:
    """Formati presenti nel CSV, nell'ordine canonico."""
    present = set(df["format"])
    return [f for f in FORMATS if f in present]


def _average(df: pd.DataFrame) -> pd.DataFrame:
    """Media sui seed per ogni coppia (format, size)."""
    return df.groupby(["format", "size"], as_index=False).mean(numeric_only=True)


def _cross(xs: np.ndarray, ys: np.ndarray, params: np.ndarray, level: float):
    """Params dove acc_mean attraversa level, interpolando linearmente in log10(params)."""
    hit = np.nonzero(ys >= level)[0]
    if hit.size == 0:
        return math.nan, False
    i = hit[0]
    if i == 0:
        # gia' raggiunto alla taglia piu' piccola: limite inferiore
        return params[0], True
    t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
    logp = xs[i - 1] + t * (xs[i] - xs[i - 1])
    return 10 ** logp, False


def thresholds(df: pd.DataFrame, levels=(0.5, 0.9)) -> pd.DataFrame:
    """Per formato: params_nonemb interpolati per ogni livello di acc_mean e rapporto vs italiano_pieno."""
    avg = _average(df)
    rows = []
    for fmt in _formats(df):
        sub = avg[avg["format"] == fmt].sort_values("params_nonemb")
        params = sub["params_nonemb"].to_numpy(dtype=float)
        xs = np.log10(params)
        ys = sub["acc_mean"].to_numpy(dtype=float)
        for level in levels:
            value, lower = _cross(xs, ys, params, level)
            rows.append({"format": fmt, "level": float(level), "params": float(value),
                         "lower_bound": bool(lower)})
    out = pd.DataFrame(rows, columns=["format", "level", "params", "lower_bound"])

    base = {r.level: r.params for r in out.itertuples() if r.format == BASE_FORMAT}
    ratios = []
    for r in out.itertuples():
        b = base.get(r.level, math.nan)
        ok = math.isfinite(r.params) and math.isfinite(b) and b != 0
        ratios.append(r.params / b if ok else math.nan)
    out["ratio_vs_pieno"] = ratios
    return out


def _plot_accuracy(avg: pd.DataFrame, formats: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt in formats:
        sub = avg[avg["format"] == fmt].sort_values("params_nonemb")
        ax.plot(sub["params_nonemb"], sub["acc_mean"], marker="o", label=fmt)
    for level in (0.5, 0.9):
        ax.axhline(level, linestyle="--", color="grey", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("params_nonemb")
    ax.set_ylabel("acc_mean")
    ax.set_title("Accuratezza vs parametri non-embedding")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_compute(avg: pd.DataFrame, formats: list[str], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for fmt in formats:
        sub = avg[avg["format"] == fmt]
        ax1.plot(sub["flops"], sub["acc_mean"], marker="o", label=fmt)
        ax2.plot(sub["train_time_s"], sub["acc_mean"], marker="o", label=fmt)
    ax1.set_xscale("log")
    ax1.set_xlabel("flops")
    ax1.set_ylabel("acc_mean")
    ax2.set_xscale("log")
    ax2.set_xlabel("train_time_s")
    ax2.set_ylabel("acc_mean")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "NaN" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _summary_md(df: pd.DataFrame, avg: pd.DataFrame, thr: pd.DataFrame) -> str:
    formats = _formats(df)
    parts = ["# Riepilogo esperimento", ""]

    # tokens_per_bio medio per formato
    tpb = avg.groupby("format")["tokens_per_bio"].mean()
    parts += ["## Token per biografia (media per formato)", "",
              _md_table(["format", "tokens_per_bio"],
                        [[f, tpb[f]] for f in formats]), ""]

    # soglie
    parts += ["## Soglie di accuratezza", "",
              _md_table(["format", "level", "params", "lower_bound", "ratio_vs_pieno"],
                        [[r.format, r.level, r.params, r.lower_bound, r.ratio_vs_pieno]
                         for r in thr.itertuples()]), ""]

    # totali su tutte le run, rapporto vs italiano_pieno
    cols = ["train_tokens", "flops", "train_time_s"]
    totals = df.groupby("format")[cols].sum().reindex(formats)
    base = totals.loc[BASE_FORMAT] if BASE_FORMAT in totals.index else None
    rows = []
    for f in formats:
        row = [f] + [totals.loc[f, c] for c in cols]
        for c in cols:
            ok = base is not None and base[c] != 0
            row.append(totals.loc[f, c] / base[c] if ok else math.nan)
        rows.append(row)
    parts += ["## Totali per formato (somma su tutte le run)", "",
              _md_table(["format", *cols, *(f"{c}_vs_pieno" for c in cols)], rows), ""]

    # accuratezza per taglia e per attributo
    headers = ["format", "size", "params_nonemb", "acc_mean", *(f"acc_{a}" for a in ATTRS)]
    rows = []
    for f in formats:
        sub = avg[avg["format"] == f].sort_values("params_nonemb")
        for r in sub.itertuples(index=False):
            rd = r._asdict()
            rows.append([f, rd["size"], rd["params_nonemb"], rd["acc_mean"],
                         *(rd[f"acc_{a}"] for a in ATTRS)])
    parts += ["## Accuratezza per formato e taglia, con attributi", "",
              _md_table(headers, rows), ""]

    return "\n".join(parts)


def write_all(csv_path: Path, out_dir: Path) -> str:
    """Scrive accuracy_vs_params.png, compute.png, summary.md in out_dir; ritorna il markdown."""
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    avg = _average(df)
    formats = _formats(df)
    thr = thresholds(df)

    _plot_accuracy(avg, formats, out_dir / "accuracy_vs_params.png")
    _plot_compute(avg, formats, out_dir / "compute.png")
    md = _summary_md(df, avg, thr)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    return md
