"""Analisi dei risultati di v5: accuratezza con IC bootstrap, test di McNemar a coppie, costo in token e latenza, errori campione."""
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import binomtest

from src.spec5 import FORMAT_NAMES, FORMATS, GSM8K, MODELS, OUTPUT_FIELDS

SUMMARY_COLUMNS = ("model", "format", "n_seeds", "acc", "acc_lo", "acc_hi", "conform_rate", "tok_mean", "tok_median",
                   "lat_mean_s", "ms_per_token_mean", "answers_per_s", "tokens_per_s", "speedup_vs_A")
PAIRWISE_COLUMNS = ("model", "format_a", "format_b", "n_cells", "acc_diff", "n01", "n10", "p")


def _order(values, ref):
    """Ordina i valori secondo l'ordine di riferimento (modelli o formati); i valori ignoti vanno in fondo."""
    return sorted(values, key=lambda v: ref.index(v) if v in ref else len(ref))


def _runs(results_dir, stage="main", split="test"):
    """Righe di runs.csv per stage e split (le run 'sweep' dell'lr non entrano nell'analisi)."""
    runs = pd.read_csv(Path(results_dir) / "runs.csv")
    return runs[(runs["stage"] == stage) & (runs["split"] == split)]


def _correct(df):
    """Serie 0/1 di correttezza indicizzata per id, per allineare i problemi tra seed."""
    return pd.Series(df["correct"].astype(float).to_numpy(), index=df["id"].to_numpy())


def load_outputs(results_dir, run_id, split="test"):
    """Output per problema di una run: RESULTS/outputs/{run_id}_{split}.jsonl con OUTPUT_FIELDS."""
    path = Path(results_dir) / "outputs" / f"{run_id}_{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows, columns=list(OUTPUT_FIELDS))


def bootstrap_ci(correct, n_boot=2000, seed=0):
    """Media e IC al 95% (percentili) con ricampionamento dei problemi; correct e' problemi x seed con valori 0/1."""
    m = np.asarray(correct, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    n = m.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = m[idx].mean(axis=(1, 2))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(m.mean()), float(lo), float(hi)


def mcnemar(n01, n10):
    """p-value esatto bilaterale di McNemar sulle coppie discordanti; 1.0 se non ce ne sono."""
    n = int(n01) + int(n10)
    if n == 0:
        return 1.0
    return float(binomtest(int(min(n01, n10)), n, 0.5, alternative="two-sided").pvalue)


def _latency(results_dir, runs):
    """Righe di latency.csv delle sole run principali."""
    lat = pd.read_csv(Path(results_dir) / "latency.csv")
    return lat[lat["run_id"].isin(runs["run_id"])]


def summary_table(results_dir):
    """Una riga per modello x formato (stage main, split test)."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    lat = _latency(results_dir, runs)
    rows = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        for fmt in _order(runs.loc[runs["model"] == model, "format"].unique(), FORMATS):
            sub = runs[(runs["model"] == model) & (runs["format"] == fmt)].sort_values("seed")
            outs = [load_outputs(results_dir, r) for r in sub["run_id"]]
            series = [_correct(o) for o in outs]
            common = series[0].index
            for s in series[1:]:
                common = common.intersection(s.index)
            if len(common) == 0:
                raise ValueError(f"{model} {fmt}: nessun id in comune tra i seed")
            matrix = np.column_stack([s.loc[common].to_numpy() for s in series])
            acc, lo, hi = bootstrap_ci(matrix)
            pooled = pd.concat(outs)
            lat_sub = lat[(lat["model"] == model) & (lat["format"] == fmt)]
            rows.append(dict(
                model=model, format=fmt, n_seeds=len(outs), acc=acc, acc_lo=lo, acc_hi=hi,
                conform_rate=float(pooled["conform"].astype(float).mean()),
                tok_mean=float(pooled["n_tokens"].astype(float).mean()),
                tok_median=float(pooled["n_tokens"].astype(float).median()),
                lat_mean_s=float(lat_sub["lat_mean_s"].mean()),
                ms_per_token_mean=float(lat_sub["ms_per_token_mean"].mean()),
                answers_per_s=float(sub["answers_per_s"].mean()),
                tokens_per_s=float(sub["tokens_per_s"].mean()),
            ))
    df = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    base = {r.model: r.lat_mean_s for r in df.itertuples() if r.format == "A"}
    df["speedup_vs_A"] = [base.get(m, np.nan) / l for m, l in zip(df["model"], df["lat_mean_s"])]
    return df


def pairwise(results_dir):
    """Per modello e coppia di formati: differenza di accuratezza e McNemar sommando i seed presenti in entrambi.
    n01 = formato_a sbagliato e formato_b giusto; n10 = formato_a giusto e formato_b sbagliato."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    rows = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        sub = runs[runs["model"] == model]
        for fa, fb in itertools.combinations(_order(sub["format"].unique(), FORMATS), 2):
            ra = sub[sub["format"] == fa].set_index("seed")["run_id"]
            rb = sub[sub["format"] == fb].set_index("seed")["run_id"]
            seeds = sorted(set(ra.index) & set(rb.index))
            if not seeds:
                continue
            cells_a, cells_b = [], []
            n01 = n10 = 0
            for seed in seeds:
                sa = _correct(load_outputs(results_dir, ra[seed]))
                sb = _correct(load_outputs(results_dir, rb[seed]))
                common = sa.index.intersection(sb.index)
                a = sa.loc[common].to_numpy()
                b = sb.loc[common].to_numpy()
                cells_a.append(a)
                cells_b.append(b)
                n01 += int(((a == 0) & (b == 1)).sum())
                n10 += int(((a == 1) & (b == 0)).sum())
            a = np.concatenate(cells_a)
            b = np.concatenate(cells_b)
            rows.append(dict(model=model, format_a=fa, format_b=fb, n_cells=len(a),
                             acc_diff=float(a.mean() - b.mean()), n01=n01, n10=n10, p=mcnemar(n01, n10)))
    return pd.DataFrame(rows, columns=list(PAIRWISE_COLUMNS))


def plot_acc_vs_tokens(summary, out_dir):
    """Un png per modello: accuratezza con IC in funzione dei token medi per risposta. Ritorna i percorsi."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, g in summary.groupby("model", sort=False):
        fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        acc = g["acc"].to_numpy(dtype=float)
        x = g["tok_mean"].to_numpy(dtype=float)
        yerr = np.vstack([acc - g["acc_lo"].to_numpy(dtype=float), g["acc_hi"].to_numpy(dtype=float) - acc])
        ax.errorbar(x, acc, yerr=yerr, fmt="o", capsize=4)
        for xi, yi, fmt in zip(x, acc, g["format"]):
            ax.annotate(f"{fmt} {FORMAT_NAMES[fmt]}", (xi, yi), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("token medi per risposta")
        ax.set_ylabel("accuracy (test)")
        ax.set_title(model)
        path = out_dir / f"acc_vs_tokens_{model}.png"
        fig.savefig(path, dpi=120)
        paths.append(path)
    return paths


def _questions():
    """Domande del test GSM8K per id; vuoto se il file non e' presente."""
    path = GSM8K / "test.jsonl"
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row["question"]
    return out


def sample_errors(results_dir, k=20, seed=0):
    """Fino a k risposte sbagliate per modello x formato dalla run di seed 0 (stage main, split test)."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    runs = runs[runs["seed"] == 0]
    questions = _questions()
    rng = random.Random(seed)
    out = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        for fmt in _order(runs.loc[runs["model"] == model, "format"].unique(), FORMATS):
            run_id = runs[(runs["model"] == model) & (runs["format"] == fmt)]["run_id"].iloc[0]
            df = load_outputs(results_dir, run_id)
            wrong = [r for r in df.to_dict("records") if not r["correct"]]
            for r in rng.sample(wrong, min(k, len(wrong))):
                out.append({
                    "model": model, "format": fmt, "id": r["id"], "question": questions.get(r["id"], ""),
                    "output": r["output"], "pred": r["pred"], "gold": r["gold"], "conform": bool(r["conform"]),
                })
    return out


def _md(df):
    """Tabella markdown semplice (senza dipendenze esterne); i float a 3 decimali, NaN come '-'."""
    def cell(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def write_all(results_dir):
    """Scrive summary.csv, pairwise.csv, errors_sample.jsonl e i grafici in plots/; ritorna il markdown di riepilogo."""
    results_dir = Path(results_dir)
    summary = summary_table(results_dir)
    pw = pairwise(results_dir)
    summary.to_csv(results_dir / "summary.csv", index=False)
    pw.to_csv(results_dir / "pairwise.csv", index=False)
    with open(results_dir / "errors_sample.jsonl", "w", encoding="utf-8") as f:
        for e in sample_errors(results_dir):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    plot_acc_vs_tokens(summary, results_dir / "plots")
    return "## Riepilogo\n\n" + _md(summary) + "\n\n## Confronti a coppie\n\n" + _md(pw) + "\n"
