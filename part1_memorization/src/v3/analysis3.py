"""Analisi v3: sweep del vocabolario (D), stripper e realizzatore (E), grafici e riepilogo markdown."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # niente finestre: solo file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import report as v1  # noqa: E402
from src.v2.analysis import thresholds  # noqa: E402
from src.v2.spec import V2_RESULTS  # noqa: E402
from src.v3.spec3 import D_FORMATS, D_SIZES, E_SIZES  # noqa: E402

BASE = "italiano_pieno"  # riferimento per i rapporti (come v2)
REF_VOCAB = "bpe4096"    # vocabolario dei run v2 riusati come riferimento
LEVELS = (0.5, 0.9)
HI = 0.9


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def _v2_rows(stage: str, corpora, sizes) -> pd.DataFrame | None:
    """Righe v2 di riferimento per corpora e taglie; None se il CSV manca o non ha righe."""
    df = _read_csv(V2_RESULTS / f"{stage}.csv")
    if df is None:
        return None
    df = df[df["corpus"].isin(corpora) & df["size"].isin(sizes)]
    return df if len(df) else None


def _pct(level) -> str:
    return f"{round(float(level) * 100)}%"


def _section(title: str, body: str) -> list[str]:
    return [f"## {title}", "", body, ""]


def _div(a: float, b: float) -> float:
    return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else math.nan


def _vs_base(thr: pd.DataFrame, key: str, base, by: list[str] | None = None) -> list[tuple[float, float]]:
    """Per riga di thr: params_nonemb e params_total rapportati alla riga con key == base, stesso livello e stessi by."""
    by = by or []
    recs = thr.to_dict("records")
    ref = {(r[key], *(r[k] for k in by), r["level"]): r for r in recs}
    out = []
    for r in recs:
        b = ref.get((base, *(r[k] for k in by), r["level"]))
        out.append(tuple(_div(r[col], b[col]) if b is not None else math.nan
                         for col in ("params_nonemb", "params_total")))
    return out


def _tot(thr: pd.DataFrame, fmt: str, vocab: str, level: float) -> float:
    s = thr[(thr["format"] == fmt) & (thr["vocab"] == vocab) & (thr["level"] == level)]
    return float(s["params_total"].iloc[0]) if len(s) else math.nan


def _pieno_min(thr: pd.DataFrame, level: float) -> float:
    """params_total minimo di italiano_pieno, su tutti i vocabolari, alla soglia level."""
    vals = thr[(thr["format"] == BASE) & (thr["level"] == level)]["params_total"].dropna()
    return float(vals.min()) if len(vals) else math.nan


def _best(thr: pd.DataFrame) -> list[dict]:
    """Per formato: vocabolario con params_total minimo alla soglia 90%; se nessuno la raggiunge, al 50%."""
    out = []
    for fmt in dict.fromkeys(thr["format"]):
        sub = thr[thr["format"] == fmt]
        for level in (HI, 0.5):
            ok = sub[(sub["level"] == level) & sub["params_total"].notna()]
            if len(ok):
                r = ok.loc[ok["params_total"].idxmin()]
                out.append({"format": fmt, "vocab": r["vocab"], "level": level,
                            "params_total": float(r["params_total"])})
                break
    return out


def _train_time_90(d: pd.DataFrame, thr: pd.DataFrame) -> list[list]:
    """Per formato e vocabolario: train_time_s interpolato in log-log su params_total (media sui seed) a params_total del 90%."""
    rows = []
    for r in thr[thr["level"] == HI].to_dict("records"):
        if not math.isfinite(r["params_total"]):
            continue
        sub = d[(d["format"] == r["format"]) & (d["vocab"] == r["vocab"])]
        per = sub.groupby("size", as_index=False).agg(
            pt=("params_total", "mean"), tt=("train_time_s", "mean")).dropna().sort_values("pt")
        tt = math.nan
        if len(per):
            x = np.log10(per["pt"].to_numpy(dtype=float))
            y = np.log10(per["tt"].to_numpy(dtype=float))
            t = math.log10(r["params_total"])
            if x[0] <= t <= x[-1]:
                tt = 10 ** float(np.interp(t, x, y))
        rows.append([r["format"], r["vocab"], r["params_total"], tt])
    return rows


def _load_d(results_dir: Path) -> pd.DataFrame | None:
    """D_main più i riferimenti v2 bpe4096, con colonne format (prima di @) e vocab (dopo @)."""
    frames = []
    main = _read_csv(results_dir / "D_main.csv")
    if main is not None:
        frames.append(main)
    for stage, corpora in (("A_main", (BASE, "telegrafico")), ("A_abl", ("interlingua_v1enc",))):
        ref = _v2_rows(stage, corpora, D_SIZES)
        if ref is not None:
            frames.append(ref.assign(corpus=ref["corpus"] + f"@{REF_VOCAB}"))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).rename(columns={"vocab": "vocab_size"})
    df = df[df["corpus"].str.contains("@", regex=False, na=False)]
    if not len(df):
        return None
    parts = df["corpus"].str.split("@", n=1, expand=True)
    return df.assign(format=parts[0], vocab=parts[1]).reset_index(drop=True)


def _d_sections(d: pd.DataFrame) -> list[str]:
    parts = []
    acc = d.groupby(["format", "vocab", "size"], as_index=False).agg(
        params_nonemb=("params_nonemb", "mean"), acc_mean=("acc_mean", "mean"))
    acc = acc.sort_values(["format", "vocab", "params_nonemb"])
    parts += _section("D: accuratezza per formato, vocabolario e taglia (media sui seed)",
                      v1._md_table(["format", "vocab", "size", "params_nonemb", "acc_mean"],
                                   acc.values.tolist()))

    tok = d.groupby(["format", "vocab"], as_index=False).agg(
        tokens_per_bio=("tokens_per_bio", "mean"), vocab_size=("vocab_size", "mean"))
    parts += _section("D: token per bio e dimensione del vocabolario",
                      v1._md_table(["format", "vocab", "tokens_per_bio", "vocab_size"], tok.values.tolist()))

    thr = thresholds(d, ["format", "vocab"])
    rel = _vs_base(thr, "format", BASE, ["vocab"])
    rows = [[r["format"], r["vocab"], _pct(r["level"]), r["params_nonemb"], r["params_total"],
             r["lower_bound"], r["n_seeds"], rn, rt]
            for r, (rn, rt) in zip(thr.to_dict("records"), rel)]
    parts += _section("D: soglie 50% e 90% (rapporti vs italiano_pieno allo stesso vocabolario)",
                      v1._md_table(["format", "vocab", "livello", "params_nonemb", "params_total",
                                    "lower_bound", "n_seeds", "rapporto_nonemb_vs_pieno_stesso_vocab",
                                    "rapporto_total_vs_pieno_stesso_vocab"], rows))

    rows = []
    for b in _best(thr):
        lvl = b["level"]
        rows.append([b["format"], b["vocab"], _pct(lvl), b["params_total"],
                     _div(b["params_total"], _pieno_min(thr, lvl)),
                     _div(b["params_total"], _tot(thr, BASE, b["vocab"], lvl))])
    parts += _section("D: vocabolario migliore per formato (params_total minimo al 90%, altrimenti al 50%)",
                      v1._md_table(["format", "vocab migliore", "livello", "params_total",
                                    "rapporto vs italiano_pieno migliore",
                                    "rapporto vs italiano_pieno stesso vocab"], rows))

    parts += _section("D: tempo di addestramento fino al 90% (train_time_s, log-log su params_total)",
                      v1._md_table(["format", "vocab", "params_total 90%", "train_time_s 90%"],
                                   _train_time_90(d, thr)))
    return parts


def _lr_md(lr: pd.DataFrame) -> str:
    """lr_mult scelto per corpus e taglia: miglior acc_mean medio sui seed, a parità final_loss più basso."""
    g = lr.groupby(["corpus", "size", "lr_mult"], as_index=False).agg(
        acc_mean=("acc_mean", "mean"), final_loss=("final_loss", "mean"))
    g = g.sort_values(["acc_mean", "final_loss"], ascending=[False, True], na_position="last")
    best = g.drop_duplicates(["corpus", "size"], keep="first").sort_values(["corpus", "size"])
    rows = best[["corpus", "size", "lr_mult", "acc_mean", "final_loss"]].values.tolist()
    return v1._md_table(["corpus", "size", "lr_mult scelto", "acc_mean", "final_loss"], rows)


def _mean_md(df: pd.DataFrame) -> str:
    """Medie delle colonne numeriche, raggruppate per le colonne non numeriche (se ci sono)."""
    keys = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if keys:
        g = df.groupby(keys, as_index=False).mean(numeric_only=True)
    else:
        g = df.mean(numeric_only=True).to_frame().T
    return v1._md_table(list(g.columns), g.values.tolist())


def _e_main_md(e: pd.DataFrame) -> list[str]:
    parts = []
    auto = e[e["corpus"] == "telegrafico_auto"]
    ref = _v2_rows("A_main", ("telegrafico",), E_SIZES)
    if len(auto) and ref is not None:
        thr = thresholds(pd.concat([auto, ref], ignore_index=True), ["corpus"])
        rel = _vs_base(thr, "corpus", "telegrafico")
        rows = [[r["corpus"], _pct(r["level"]), r["params_nonemb"], r["params_total"], r["lower_bound"],
                 r["n_seeds"], rn, rt]
                for r, (rn, rt) in zip(thr.to_dict("records"), rel)]
        parts += _section("E: telegrafico_auto vs telegrafico (v2, bpe4096)",
                          v1._md_table(["corpus", "livello", "params_nonemb", "params_total", "lower_bound",
                                        "n_seeds", "rapporto_nonemb_vs_telegrafico",
                                        "rapporto_total_vs_telegrafico"], rows))
    lista = e[e["corpus"] == "telegrafico_lista"]
    if len(lista):
        g = lista.groupby("size", as_index=False).agg(
            params_nonemb=("params_nonemb", "mean"), acc_mean=("acc_mean", "mean"),
            n_seeds=("seed", "nunique")).sort_values("params_nonemb")
        parts += _section("E: telegrafico_lista, accuratezza per taglia",
                          v1._md_table(["size", "params_nonemb", "acc_mean", "n_seeds"],
                                       g[["size", "params_nonemb", "acc_mean", "n_seeds"]].values.tolist()))
    return parts


def _save(path: Path, series, title: str, xlabel: str, ylabel: str) -> None:
    """series: lista di (etichetta, x, y)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, x, y in series:
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_acc(d: pd.DataFrame, path: Path) -> None:
    """Un pannello per formato, una linea per vocabolario: acc_mean vs params_total (media sui seed)."""
    known = set(d["format"])
    fmts = [f for f in D_FORMATS if f in known] + sorted(known - set(D_FORMATS))
    fig, axes = plt.subplots(1, len(fmts), figsize=(4.5 * len(fmts), 4.2), sharey=True, squeeze=False)
    for ax, fmt in zip(axes[0], fmts):
        g = d[d["format"] == fmt].groupby(["vocab", "size"], as_index=False).agg(
            params_total=("params_total", "mean"), acc=("acc_mean", "mean"))
        for vocab, sub in g.groupby("vocab"):
            sub = sub.sort_values("params_total")
            ax.plot(sub["params_total"], sub["acc"], marker="o", label=vocab)
        for level in LEVELS:
            ax.axhline(level, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("params_total")
        ax.set_title(fmt)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0][0].set_ylabel("acc_mean")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_tok(d: pd.DataFrame, path: Path) -> None:
    g = d.groupby(["format", "vocab_size"], as_index=False)["tokens_per_bio"].mean()
    series = []
    for fmt, sub in g.groupby("format"):
        sub = sub.sort_values("vocab_size")
        series.append((fmt, sub["vocab_size"].to_numpy(dtype=float), sub["tokens_per_bio"].to_numpy(dtype=float)))
    _save(path, series, "Token per bio vs dimensione del vocabolario", "vocab", "tokens_per_bio")


def _plot_pipeline(pipe: pd.DataFrame, e2e: pd.DataFrame | None, path: Path) -> None:
    """total_ms per formato, taglia e batch, confrontato con pieno_ms; se e2e_c esiste, anche pipeline_ms (neurale)."""
    rows = pipe.assign(label=pipe["format"] + " " + pipe["size"].astype(str)
                       + " b" + pipe["batch"].astype(str)).reset_index(drop=True)
    series = [("total_ms", "total_ms"), ("pieno_ms", "pieno_ms")]
    if e2e is not None and {"format", "size", "batch", "pipeline_ms"} <= set(e2e.columns):
        rows = rows.merge(e2e[["format", "size", "batch", "pipeline_ms"]],
                          on=["format", "size", "batch"], how="left")
        series.append(("pipeline_ms (neurale, e2e_c)", "pipeline_ms"))
    x = np.arange(len(rows))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(rows) + 3), 4.5))
    for i, (label, col) in enumerate(series):
        ax.bar(x - 0.4 + w * (i + 0.5), rows[col].to_numpy(dtype=float), w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=45, ha="right")
    ax.set_ylabel("ms (batch=1: per richiesta; batch=64: per lotto)")
    ax.set_title("Tempi della pipeline vs pieno")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_all(results_dir: Path) -> str:
    """Legge i CSV v3 (+ riferimenti v2 bpe4096), scrive grafici e summary_v3.md in results_dir; ritorna il markdown."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    parts = ["# Riepilogo esperimento v3", ""]

    d = _load_d(results_dir)
    if d is not None:
        parts += _d_sections(d)
        _plot_acc(d, results_dir / "acc_vs_params_total.png")
        _plot_tok(d, results_dir / "tok_bio_vs_vocab.png")

    lr = _read_csv(results_dir / "D_lr.csv")
    if lr is not None:
        parts += _section("D_lr: lr_mult scelto per corpus e taglia", _lr_md(lr))

    lat = _read_csv(results_dir / "D_latency.csv")
    if lat is not None:
        g = lat.groupby(["format", "vocab", "size", "batch"], as_index=False).agg(
            latency_ms_mean=("latency_ms_mean", "mean"), ms_per_token=("ms_per_token", "mean"),
            answers_per_s=("answers_per_s", "mean"))
        parts += _section("D: latenza di risposta (media sui seed)",
                          v1._md_table(list(g.columns), g.values.tolist()))

    timing = _read_csv(results_dir / "D_timing.csv")
    if timing is not None:
        g = timing.groupby(["format", "vocab", "tokens_step", "T"], as_index=False).agg(
            head_ms=("head_ms", "mean"), body_ms=("body_ms", "mean"))
        parts += _section("D: tempi di passo (testa e corpo, media)",
                          v1._md_table(list(g.columns), g.values.tolist()))

    e = _read_csv(results_dir / "E_main.csv")
    if e is not None:
        parts += _e_main_md(e)

    sc = _read_csv(results_dir / "E_strip_compare.csv")
    if sc is not None:
        parts += _section("E: confronto stripper (medie)", _mean_md(sc))

    pipe = _read_csv(results_dir / "E_pipeline.csv")
    if pipe is not None:
        parts += _section("E: pipeline", v1._md_table(list(pipe.columns), pipe.values.tolist()))
        if {"format", "size", "batch", "total_ms", "pieno_ms"} <= set(pipe.columns):
            e2e = _read_csv(V2_RESULTS / "e2e_c.csv")
            _plot_pipeline(pipe, e2e, results_dir / "pipeline_compare.png")

    md = "\n".join(parts)
    (results_dir / "summary_v3.md").write_text(md, encoding="utf-8")
    return md
