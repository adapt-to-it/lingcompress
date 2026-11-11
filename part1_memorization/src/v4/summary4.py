"""Riepilogo v4: soglie con intervallo bootstrap, 2x2, tempo di training, forme, grafici e riepilogo markdown."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # niente finestre: solo file
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import report as v1  # noqa: E402
from src.v4.analysis4 import (  # noqa: E402
    IC, IP, QUANTITIES, TC, TL, attr_thresholds, decompose_2x2, thresholds_ci, train_time_at)
from src.v4.spec4 import F_ORDER, F_SIZES  # noqa: E402

N_BOOT = 1000
HI = 0.9
FOUR = (IP, IC, TL, TC)
ACC_SIZES = tuple(F_SIZES[F_SIZES.index("0.31M"):F_SIZES.index("0.52M") + 1])


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def _pct(level) -> str:
    return f"{round(float(level) * 100)}%"


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body, ""]


def _order(names) -> list[str]:
    names = set(names)
    return [f for f in F_ORDER if f in names] + sorted(names - set(F_ORDER))


def _ratio(a: float, b: float) -> float:
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else math.nan


def _k(v) -> str:
    return f"{v / 1000:.0f}k" if math.isfinite(v) else "-"


def _f(v, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if math.isfinite(v) else "-"


def _share(v) -> str:
    return f"{100 * v:.1f}%" if math.isfinite(v) else "-"


def _span(v, lo, hi, fmt) -> str:
    return f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"


def _raw_md(df: pd.DataFrame) -> str:
    return v1._md_table(list(df.columns), df.values.tolist())


def _stato_md(main: pd.DataFrame) -> str:
    rows = []
    for fmt in _order(main["format"].unique()):
        sub = main[main["format"] == fmt]
        sizes = sub.groupby("size")["params_total"].mean().sort_values().index
        seeds = sorted(sub["seed"].unique())
        rows.append([fmt, len(sub), ", ".join(sizes), ", ".join(str(int(s)) for s in seeds)])
    return v1._md_table(["formato", "run", "taglie", "seed"], rows)


def _forme_md(main: pd.DataFrame | None, forms: pd.DataFrame | None, bits: pd.Series | None) -> str:
    tok = main.groupby("format")["tokens_per_bio"].mean() if main is not None else pd.Series(dtype=float)
    bits = bits if bits is not None else pd.Series(dtype=float)
    rows = []
    for fmt in _order(set(tok.index) | set(bits.index)):
        cells = [fmt, _f(tok.get(fmt, math.nan), 1), _f(bits.get(fmt, math.nan))]
        sub = forms[forms["format"] == fmt] if forms is not None else None
        if sub is None or not len(sub):
            rows.append([*cells, "-", "-", "-"])
            continue
        for r in sub.itertuples(index=False):
            rows.append([*cells, r.attr, f"{r.n_forms:.0f}", f"{r.n_forms_core:.0f}"])
    headers = ["formato", "tokens_per_bio", "bit/bio", "attributo", "n_forms", "n_forms_core"]
    return v1._md_table(headers, rows)


def _soglie_md(thr: pd.DataFrame) -> str:
    base = {r["level"]: r["params_total"] for r in thr.to_dict("records") if r["format"] == IP}
    rows = []
    for r in thr.to_dict("records"):
        rel = _ratio(r["params_total"], base.get(r["level"], math.nan))
        rows.append([r["format"], _pct(r["level"]), _k(r["params_total"]),
                     f"[{_k(r['lo'])}, {_k(r['hi'])}]", f"{_k(r['seed_min'])}-{_k(r['seed_max'])}", _f(rel)])
    headers = ["formato", "livello", "params_total", "[lo, hi]", "seed min-max", "rapporto vs italiano_pieno"]
    return v1._md_table(headers, rows)


def _due_md(dec: pd.DataFrame, thr: pd.DataFrame) -> str | None:
    """Solo i livelli in cui tutti e 4 i formati hanno la soglia; None se nessuno."""
    levels = []
    for lvl in dec["level"]:
        present = set(thr.loc[(thr["level"] == lvl) & thr["params_total"].notna(), "format"])
        if set(FOUR) <= present:
            levels.append(lvl)
    if not levels:
        return None
    rows = []
    for lvl in levels:
        r = dec[dec["level"] == lvl].iloc[0]
        cells = [_span(r[q], r[f"{q}_lo"], r[f"{q}_hi"], _share if q.startswith("share") else _f)
                 for q in QUANTITIES]
        rows.append([_pct(lvl), *cells])
    return v1._md_table(["livello", *QUANTITIES], rows)


def _tempo_md(main: pd.DataFrame, thr: pd.DataFrame) -> str:
    tt = train_time_at(main, thr, level=HI)
    rows = [[r["format"], _k(r["params_total"]), f"{r['train_time_s']:.0f}" if math.isfinite(r["train_time_s"])
             else "-"] for r in tt.to_dict("records")]
    return v1._md_table(["formato", "params_total 90%", "train_time_s 90%"], rows)


def _attr_md(at: pd.DataFrame) -> str:
    rows = [[r["format"], r["attr"], _pct(r["level"]), _k(r["params_total"]), _f(r["ratio_reg"])]
            for r in at.to_dict("records")]
    return v1._md_table(["formato", "attributo", "livello", "params_total", "ratio_reg (canonico / vario)"], rows)


def _acc_attr_md(main: pd.DataFrame) -> str | None:
    sub = main[main["size"].isin(ACC_SIZES)]
    if not len(sub):
        return None
    cols = [*(c for c in main.columns if c.startswith("acc_") and c != "acc_mean"), "acc_mean"]
    rows = []
    for fmt in _order(sub["format"].unique()):
        for size in ACC_SIZES:
            s = sub[(sub["format"] == fmt) & (sub["size"] == size)]
            if not len(s):
                continue
            rows.append([fmt, size, _k(s["params_total"].mean()), *(_f(s[c].mean(), 3) for c in cols)])
    if not rows:
        return None
    return v1._md_table(["formato", "taglia", "params_total", *cols], rows)


def _lr_md(lr: pd.DataFrame) -> str:
    g = lr.groupby(["format", "size", "lr_mult"], as_index=False).agg(
        acc_mean=("acc_mean", "mean"), final_loss=("final_loss", "mean"))
    rows = g[["format", "size", "lr_mult", "acc_mean", "final_loss"]].values.tolist()
    return v1._md_table(["formato", "taglia", "lr_mult", "acc_mean", "final_loss"], rows)


def _plot_acc(main: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt in _order(main["format"].unique()):
        g = main[main["format"] == fmt].groupby("size").agg(
            params=("params_total", "mean"), acc=("acc_mean", "mean"),
            lo=("acc_mean", "min"), hi=("acc_mean", "max")).sort_values("params")
        line, = ax.plot(g["params"], g["acc"], marker="o", label=fmt)
        ax.fill_between(g["params"], g["lo"], g["hi"], color=line.get_color(), alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("params_total")
    ax.set_ylabel("acc_mean")
    ax.set_title("Accuratezza vs parametri totali (banda: min-max sui seed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_entropy(thr: pd.DataFrame, bits: pd.Series, path: Path) -> None:
    """x: bit per biografia del formato; y: params_total alla soglia, con errori lo/hi. Nessun dato, nessun file."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    drawn = False
    for lvl, marker in ((0.5, "o"), (0.9, "s")):
        pts = []
        for fmt, x in bits.items():
            r = thr[(thr["format"] == fmt) & (thr["level"] == lvl)]
            if not len(r) or not math.isfinite(r["params_total"].iloc[0]) or not math.isfinite(x):
                continue
            pts.append((fmt, float(x), r.iloc[0]))
        if not pts:
            continue
        drawn = True
        xs = [p[1] for p in pts]
        ys = [float(p[2]["params_total"]) for p in pts]
        low = [max(0.0, y - float(p[2]["lo"])) if math.isfinite(p[2]["lo"]) else 0.0 for p, y in zip(pts, ys)]
        high = [max(0.0, float(p[2]["hi"]) - y) if math.isfinite(p[2]["hi"]) else 0.0 for p, y in zip(pts, ys)]
        ax.errorbar(xs, ys, yerr=[low, high], marker=marker, linestyle="none", capsize=3, label=_pct(lvl))
        for (fmt, _, _), x, y in zip(pts, xs, ys):
            ax.annotate(f"{fmt} {_pct(lvl)}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("bit per biografia (somma di entropy_bits)")
    ax.set_ylabel("params_total alla soglia")
    ax.set_title("Soglia di parametri vs entropia della forma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_all(results_dir: Path) -> str:
    """Legge i CSV v4 (opzionali), scrive grafici, CSV e summary_v4.md in results_dir; ritorna il markdown."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    main = _read_csv(results_dir / "F_main.csv")
    lr = _read_csv(results_dir / "F_lr.csv")
    forms = _read_csv(results_dir / "F_forms.csv")
    align_bench = _read_csv(results_dir / "align_bench.csv")
    align_control = _read_csv(results_dir / "align_control.csv")
    bits = forms.groupby("format")["entropy_bits"].sum() if forms is not None else None
    parts = ["# Riepilogo esperimento v4", ""]

    if main is not None:
        parts += _section("Stato", _stato_md(main))
    if main is not None or forms is not None:
        parts += _section("Forme", _forme_md(main, forms, bits))
    if main is not None:
        thr, draws = thresholds_ci(main, n_boot=N_BOOT)
        dec = decompose_2x2(thr, draws)
        at = attr_thresholds(main)
        thr.to_csv(results_dir / "F_thresholds.csv", index=False)
        dec.to_csv(results_dir / "F_2x2.csv", index=False)
        at.to_csv(results_dir / "F_attr.csv", index=False)
        parts += _section("Soglie", _soglie_md(thr))
        due = _due_md(dec, thr)
        if due is not None:
            parts += _section("2x2 per livello", due)
        parts += _section("Tempo di training al 90%", _tempo_md(main, thr))
        parts += _section("Soglie per attributo", _attr_md(at))
        acc = _acc_attr_md(main)
        if acc is not None:
            parts += _section("Accuratezza per attributo (0.31M-0.52M, media sui seed)", acc)
        _plot_acc(main, results_dir / "acc_vs_params_v4.png")
        if bits is not None:
            _plot_entropy(thr, bits, results_dir / "entropy_vs_params.png")
    if lr is not None:
        parts += _section("Verifica lr", _lr_md(lr))
    if align_bench is not None:
        parts += _section("Allineamento: align_bench", _raw_md(align_bench))
    if align_control is not None:
        parts += _section("Allineamento: align_control", _raw_md(align_control))

    md = "\n".join(parts)
    (results_dir / "summary_v4.md").write_text(md, encoding="utf-8")
    return md
