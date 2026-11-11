"""Analisi v2: soglie, rapporti, bit/param, grafici e riepilogo markdown."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # niente finestre: solo file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import report as v1  # noqa: E402

STAGES = ("A_lr", "A_main", "A_exp", "A_esp", "A_shared", "A_abl", "B_main")
BASE = "italiano_pieno"  # riferimento per i rapporti (come v1)
ABL_BASE = {"interlingua_v1enc": "interlingua", "italiano_pieno@nodigits": "italiano_pieno"}
MULTI = ("multi_pieno", "multi_bridge")


def thresholds(df: pd.DataFrame, group_cols: list[str], levels=(0.5, 0.9), metric: str = "acc_mean") -> pd.DataFrame:
    """Per gruppo: params_nonemb e params_total interpolati in log al livello della metrica (media sui seed)."""
    out_cols = [*group_cols, "level", "params_nonemb", "params_total", "lower_bound", "n_seeds"]
    per = df.groupby([*group_cols, "size"], as_index=False).agg(
        params_nonemb=("params_nonemb", "mean"),
        params_total=("params_total", "mean"),
        metric=(metric, "mean"),
        n_seeds=("seed", "nunique"),
    )
    rows = []
    for keys, sub in per.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        sub = sub.sort_values("params_nonemb")
        params = sub["params_nonemb"].to_numpy(dtype=float)
        total = sub["params_total"].to_numpy(dtype=float)
        ys = sub["metric"].to_numpy(dtype=float)
        n_seeds = int(sub["n_seeds"].max())
        for level in levels:
            value, lower = v1._cross(np.log10(params), ys, params, level)
            value_total, _ = v1._cross(np.log10(total), ys, total, level)
            rows.append({**dict(zip(group_cols, keys)), "level": float(level),
                         "params_nonemb": float(value), "params_total": float(value_total),
                         "lower_bound": bool(lower), "n_seeds": n_seeds})
    return pd.DataFrame(rows, columns=out_cols)


def _ratio(thr: pd.DataFrame, base, by: list[str]) -> pd.DataFrame:
    """Rapporti params / params della base, stesso livello e stessi by; base: corpus unico o dict corpus -> base."""
    recs = thr.to_dict("records")
    ref = {(r["corpus"], tuple(r[k] for k in by), r["level"]): r for r in recs}
    rows = []
    for r in recs:
        bc = base if isinstance(base, str) else base.get(r["corpus"])
        b = ref.get((bc, tuple(r[k] for k in by), r["level"])) if bc is not None else None
        row = {}
        for col, name in (("params_nonemb", "ratio_nonemb"), ("params_total", "ratio_total")):
            num, den = (r[col], b[col]) if b is not None else (math.nan, math.nan)
            ok = math.isfinite(num) and math.isfinite(den) and den != 0
            row[name] = num / den if ok else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["ratio_nonemb", "ratio_total"])


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def _pool(primary: pd.DataFrame | None, fallback: pd.DataFrame | None, corpora) -> pd.DataFrame | None:
    """Righe di primary; per i corpora che primary non ha, quelle di fallback (baseline da A_main)."""
    frames = [] if primary is None else [primary]
    if fallback is not None:
        have = set() if primary is None else set(primary["corpus"])
        extra = fallback[fallback["corpus"].isin(corpora) & ~fallback["corpus"].isin(have)]
        if len(extra):
            frames.append(extra)
    return pd.concat(frames, ignore_index=True) if frames else None


def _pct(level) -> str:
    return f"{round(float(level) * 100)}%"


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body, ""]


def _thr_md(thr: pd.DataFrame, by: list[str], bases) -> str:
    """Tabella soglie; bases = [(etichetta, base)] con base corpus unico o dict corpus -> base."""
    headers = ["corpus", *by, "livello", "params_nonemb", "params_total", "lower_bound", "n_seeds"]
    for label, _ in bases:
        headers += [f"rapporto_nonemb_vs_{label}", f"rapporto_total_vs_{label}"]
    ratios = [_ratio(thr, base, by) for _, base in bases]
    rows = []
    for i, r in enumerate(thr.to_dict("records")):
        row = [r["corpus"], *(r[k] for k in by), _pct(r["level"]), r["params_nonemb"],
               r["params_total"], r["lower_bound"], r["n_seeds"]]
        for rat in ratios:
            row += [rat.iloc[i]["ratio_nonemb"], rat.iloc[i]["ratio_total"]]
        rows.append(row)
    return v1._md_table(headers, rows)


def _lr_md(lr: pd.DataFrame) -> str:
    """lr_mult scelto per corpus e taglia: miglior acc_mean medio sui seed, a parità final_loss più basso."""
    g = lr.groupby(["corpus", "size", "lr_mult"], as_index=False).agg(
        acc_mean=("acc_mean", "mean"), final_loss=("final_loss", "mean"))
    g = g.sort_values(["acc_mean", "final_loss"], ascending=[False, True], na_position="last")
    best = g.drop_duplicates(["corpus", "size"], keep="first").sort_values(["corpus", "size"])
    rows = best[["corpus", "size", "lr_mult", "acc_mean", "final_loss"]].values.tolist()
    return v1._md_table(["corpus", "size", "lr_mult scelto", "acc_mean", "final_loss"], rows)


def _means(df: pd.DataFrame, ycols: list[str]) -> pd.DataFrame:
    """Medie sui seed per corpus e taglia; params = params_nonemb medio."""
    return df.groupby(["corpus", "size"], as_index=False).agg(
        params=("params_nonemb", "mean"), **{y: (y, "mean") for y in ycols})


def _lines(g: pd.DataFrame, ycol: str, style: str = "-", suffix: str = ""):
    out = []
    for corpus, sub in g.groupby("corpus"):
        sub = sub.sort_values("params")
        out.append((f"{corpus}{suffix}", sub["params"].to_numpy(dtype=float),
                    sub[ycol].to_numpy(dtype=float), None, style))
    return out


def _save(path: Path, series, title: str, xlabel: str, ylabel: str, logx: bool) -> None:
    """series: lista di (etichetta, x, y, errore|None, stile); senza dati non scrive nulla."""
    series = [s for s in series if len(s[1])]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, x, y, err, style in series:
        if err is None:
            ax.plot(x, y, marker="o", linestyle=style, label=label)
        else:
            ax.errorbar(x, y, yerr=err, marker="o", linestyle=style, capsize=3, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_acc_passi(main: pd.DataFrame, path: Path) -> None:
    g = main.groupby(["corpus", "size"]).agg(
        params=("params_nonemb", "mean"), acc=("acc_mean", "mean"),
        sd=("acc_mean", "std"), n=("acc_mean", "count")).reset_index()
    series = []
    for corpus, sub in g.groupby("corpus"):
        sub = sub.sort_values("params")
        sem = sub["sd"].fillna(0.0) / np.sqrt(sub["n"])
        series.append((corpus, sub["params"].to_numpy(dtype=float), sub["acc"].to_numpy(dtype=float),
                       sem.to_numpy(dtype=float), "-"))
    _save(path, series, "Accuratezza vs parametri non-embedding (passi)",
          "params_nonemb", "acc_mean (errore standard)", logx=True)


def _bits(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per corpus ed esposizioni: massimo sulle taglie della media sui seed di bits_per_param."""
    df = pd.concat(frames, ignore_index=True)
    per_size = df.groupby(["corpus", "exposures", "size"], as_index=False)["bits_per_param"].mean()
    return per_size.groupby(["corpus", "exposures"], as_index=False)["bits_per_param"].max()


def _plot_bits(bits: pd.DataFrame, path: Path) -> None:
    series = []
    for corpus, sub in bits.groupby("corpus"):
        sub = sub.sort_values("exposures")
        series.append((corpus, sub["exposures"].to_numpy(dtype=float),
                       sub["bits_per_param"].to_numpy(dtype=float), None, "-"))
    _save(path, series, "Bit per parametro vs esposizioni (massimo sulle taglie)",
          "esposizioni", "bits_per_param", logx=False)


def _pivot_md(bits: pd.DataFrame) -> str:
    piv = bits.pivot(index="corpus", columns="exposures", values="bits_per_param")
    headers = ["corpus", *(str(c) for c in piv.columns)]
    rows = [[corpus, *piv.loc[corpus].tolist()] for corpus in piv.index]
    return v1._md_table(headers, rows)


def _speed_md(frames: list[pd.DataFrame]) -> str:
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(["stage", "corpus"], as_index=False).agg(
        steady_tokens_per_s=("steady_tokens_per_s", "mean"),
        steady_facts_per_s=("steady_facts_per_s", "mean"),
        train_time_s=("train_time_s", "sum"))
    headers = ["stage", "corpus", "steady_tokens_per_s (media)", "steady_facts_per_s (media)",
               "train_time_s (somma)"]
    return v1._md_table(headers, g.values.tolist())


def _v2_rows(thr: pd.DataFrame) -> list[list]:
    ratio = _ratio(thr, BASE, [])["ratio_nonemb"].tolist()
    return [["v2 main", r["corpus"], _pct(r["level"]), r["params_nonemb"], r["lower_bound"], q]
            for r, q in zip(thr.to_dict("records"), ratio)]


def _v1_rows(label: str, thr: pd.DataFrame) -> list[list]:
    return [[label, r["format"], _pct(r["level"]), r["params"], r["lower_bound"], r["ratio_vs_pieno"]]
            for r in thr.to_dict("records")]


def write_all(results_dir: Path) -> str:
    """Legge results/v2/*.csv (+ v1), scrive grafici e summary_v2.md in results_dir; ritorna il markdown."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data = {s: _read_csv(results_dir / f"{s}.csv") for s in STAGES}
    main, A_exp, B = data["A_main"], data["A_exp"], data["B_main"]
    parts = ["# Riepilogo esperimento v2", ""]

    if data["A_lr"] is not None:
        parts += _section("Learning rate scelto per corpus e taglia", _lr_md(data["A_lr"]))

    thr_main = None
    if main is not None:
        thr_main = thresholds(main, ["corpus"])
        _plot_acc_passi(main, results_dir / "acc_vs_params_passi.png")
        parts += _section("Soglie di accuratezza, A_main (rapporti vs italiano_pieno)",
                          _thr_md(thr_main, [], [("italiano_pieno", BASE)]))

    if data["A_esp"] is not None:
        esp = _pool(data["A_esp"], main, (BASE,))
        parts += _section("Soglie per numero di esposizioni, A_esp (rapporti vs italiano_pieno allo stesso numero)",
                          _thr_md(thresholds(esp, ["corpus", "exposures"]), ["exposures"],
                                  [("italiano_pieno", BASE)]))

    if data["A_shared"] is not None:
        shared = _pool(data["A_shared"], main, (BASE,))
        parts += _section("Soglie con tokenizer condiviso, A_shared (rapporti vs italiano_pieno)",
                          _thr_md(thresholds(shared, ["corpus"]), [], [("italiano_pieno", BASE)]))

    if data["A_abl"] is not None:
        abl = _pool(data["A_abl"], main, tuple(ABL_BASE.values()))
        base_map = {**{b: b for b in ABL_BASE.values()}, **ABL_BASE}
        parts += _section("Ablazioni, A_abl (rapporti vs corpus base)",
                          _thr_md(thresholds(abl, ["corpus"]), [], [("base", base_map)]))

    if B is not None:
        multi_thr = thresholds(_pool(B, main, ("interlingua", BASE)), ["corpus"])
        parts += _section("Soglie multilingua, B_main (rapporti vs interlingua e vs italiano_pieno)",
                          _thr_md(multi_thr, [], [("interlingua", "interlingua"), ("italiano_pieno", BASE)]))

    bit_frames = [f for f in (main, A_exp) if f is not None]
    if bit_frames:
        bits = _bits(bit_frames)
        _plot_bits(bits, results_dir / "bits_per_param_vs_exposures.png")
        parts += _section("Bit per parametro per corpus ed esposizioni (massimo sulle taglie)", _pivot_md(bits))

    ml = _pool(B, main, ("interlingua",))
    if ml is not None:
        ml = ml[ml["corpus"].isin(("multi_pieno", "multi_bridge", "interlingua"))]
        if len(ml):
            _save(results_dir / "multilingua.png", _lines(_means(ml, ["acc_mean"]), "acc_mean"),
                  "Accuratezza: multilingua vs interlingua", "params_nonemb", "acc_mean", logx=True)

    if B is not None:
        cm = B[B["corpus"].isin(MULTI)]
        if len(cm):
            g = _means(cm, ["cross_mean", "cross_lenient_mean"])
            _save(results_dir / "cross_vs_size.png",
                  _lines(g, "cross_mean", "-", " cross_mean")
                  + _lines(g, "cross_lenient_mean", "--", " cross_lenient_mean"),
                  "Recall cross-lingua vs parametri non-embedding", "params_nonemb", "cross", logx=True)
            cross_rows = g.sort_values(["corpus", "params"])[
                ["corpus", "size", "params", "cross_mean", "cross_lenient_mean"]].values.tolist()
            parts += _section("Recall cross-lingua per taglia (B_main)",
                              v1._md_table(["corpus", "size", "params_nonemb", "cross_mean", "cross_lenient_mean"],
                                           cross_rows))

    frames_all = [f for f in data.values() if f is not None]
    if frames_all:
        parts += _section("Velocità e tempo di addestramento per stage e corpus", _speed_md(frames_all))

    v1_eq = _read_csv(results_dir.parent / "results.csv")
    v1_steps = _read_csv(results_dir.parent / "results_ctrl_steps.csv")
    cmp_rows: list[list] = []
    if thr_main is not None:
        cmp_rows += _v2_rows(thr_main)
    if v1_eq is not None:
        cmp_rows += _v1_rows("v1 pari esposizioni", v1.thresholds(v1_eq))
    if v1_steps is not None:
        ref = v1_steps if v1_eq is None else pd.concat(
            [v1_steps, v1_eq[v1_eq["format"] == BASE]], ignore_index=True)
        cmp_rows += _v1_rows("v1 pari passi", v1.thresholds(ref))
    if cmp_rows:
        parts += _section("Confronto con v1 (soglie su params_nonemb)",
                          v1._md_table(["fonte", "corpus/format", "livello", "params_nonemb",
                                        "lower_bound", "rapporto_vs_italiano_pieno"], cmp_rows))

    md = "\n".join(parts)
    (results_dir / "summary_v2.md").write_text(md, encoding="utf-8")
    return md
