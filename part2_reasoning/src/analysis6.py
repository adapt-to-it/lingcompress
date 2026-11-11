"""Analisi di v6: accuratezza con IC bootstrap, McNemar tra condizioni, equazioni sbagliate, scomposizione dei contributi, grafici ed errori campione."""
import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analysis5 import bootstrap_ci, mcnemar
from src.spec5 import GSM8K, MODELS
from src.spec6 import ALGEBRA_RE, COMPARISONS, COND_LABELS, OUTPUT6_FIELDS

SUMMARY6_COLUMNS = ("model", "cond", "label", "n_seeds", "acc", "acc_lo", "acc_hi", "conform_rate", "tok_mean",
                    "tok_median", "forced_mean", "frac_wrong_eq", "recovery", "acc_algebra", "note_rate", "solve_rate",
                    "lat_mean_s", "ms_per_token_mean", "answers_per_s", "tokens_per_s", "speedup_vs_c1")
COMPARISON6_COLUMNS = ("model", "cond_a", "cond_b", "label_a", "label_b", "n_cells", "acc_diff", "n01", "n10", "p",
                       "p_per_seed")
DECOMPOSITION_COLUMNS = ("model", "gap_A_C5", "gram", "calc", "c6_gram", "residual", "tok_ratio_8_2", "lat_ratio_8_2")

# Equazioni 'a op b = c' (copiate da scripts/extra_metrics.py).
EQ = re.compile(r"((?:\(?\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?%?\s*\)?\s*[-+*/x×]\s*)+\(?\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?%?\s*\)?)"
                r"\s*=\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
ALGEBRA = re.compile(ALGEBRA_RE)


def _order(values, ref):
    """Ordina i valori secondo l'ordine di riferimento (modelli); i valori ignoti vanno in fondo."""
    return sorted(values, key=lambda v: ref.index(v) if v in ref else len(ref))


def _eval(expr):
    e = expr.strip().replace(",", "").replace("$", "").replace("×", "*").replace("x", "*").replace("%", "/100")
    return eval(compile(ast.parse(e, mode="eval"), "<eq>", "eval"), {"__builtins__": {}})


def wrong_equations(text):
    """Numero di uguaglianze 'a op b = c' ricalcolabili e sbagliate (tolleranza 1% o 0,01), e numero totale."""
    bad = tot = 0
    for expr, val in EQ.findall(str(text)):
        try:
            got, want = _eval(expr), float(val.replace(",", ""))
        except Exception:
            continue
        tot += 1
        bad += abs(got - want) > max(0.01, 0.01 * abs(want))
    return bad, tot


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


def algebra_ids():
    """Id del test GSM8K la cui soluzione ufficiale (testo prima di "####") matcha ALGEBRA_RE; vuoto se il file manca."""
    path = GSM8K / "test.jsonl"
    out = set()
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if ALGEBRA.search(row["answer"].split("####")[0]):
                    out.add(row["id"])
    return out


def _runs(results_dir):
    """Righe di runs.csv dello split test."""
    runs = pd.read_csv(Path(results_dir) / "runs.csv")
    return runs[runs["split"] == "test"]


def _latency(results_dir, runs):
    """Righe di latency.csv delle sole run considerate."""
    lat = pd.read_csv(Path(results_dir) / "latency.csv")
    return lat[lat["run_id"].isin(runs["run_id"])]


def _conds(runs, model):
    """Condizioni presenti per il modello, in ordine numerico."""
    return sorted(int(c) for c in runs.loc[runs["model"] == model, "cond"].unique())


def _correct(df):
    """Serie 0/1 di correttezza indicizzata per id, per allineare i problemi tra seed."""
    return pd.Series(df["correct"].astype(float).to_numpy(), index=df["id"].to_numpy())


def load_outputs(results_dir, run_id):
    """Output per problema di una run: RESULTS/outputs/{run_id}_test.jsonl con OUTPUT6_FIELDS."""
    path = Path(results_dir) / "outputs" / f"{run_id}_test.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows, columns=list(OUTPUT6_FIELDS))


def summary6(results_dir):
    """Una riga per modello x condizione (split test): accuratezza con IC, token, equazioni, algebra, latenza."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    lat = _latency(results_dir, runs)
    algebra = list(algebra_ids())
    rows = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        for cond in _conds(runs, model):
            sub = runs[(runs["model"] == model) & (runs["cond"] == cond)].sort_values("seed")
            outs = [load_outputs(results_dir, r) for r in sub["run_id"]]
            series = [_correct(o) for o in outs]
            common = series[0].index
            for s in series[1:]:
                common = common.intersection(s.index)
            if len(common) == 0:
                raise ValueError(f"{model} c{cond}: nessun id in comune tra i seed")
            matrix = np.column_stack([s.loc[common].to_numpy() for s in series])
            acc, lo, hi = bootstrap_ci(matrix)
            pooled = pd.concat(outs)
            correct = pooled["correct"].astype(float).to_numpy()
            has_bad = np.array([wrong_equations(t)[0] > 0 for t in pooled["output"]], dtype=bool)
            on_algebra = pooled["id"].isin(algebra).to_numpy()
            lat_sub = lat[lat["run_id"].isin(sub["run_id"])]
            rows.append(dict(
                model=model, cond=cond, label=COND_LABELS[cond], n_seeds=len(outs), acc=acc, acc_lo=lo, acc_hi=hi,
                conform_rate=float(pooled["conform"].astype(float).mean()),
                tok_mean=float(pooled["n_tokens"].astype(float).mean()),
                tok_median=float(pooled["n_tokens"].astype(float).median()),
                forced_mean=float(pooled["n_forced"].astype(float).mean()),
                frac_wrong_eq=float(has_bad.mean()),
                recovery=float(correct[has_bad].mean()) if has_bad.any() else float("nan"),
                acc_algebra=float(correct[on_algebra].mean()) if on_algebra.any() else float("nan"),
                note_rate=float(sub["note_rate"].mean()), solve_rate=float(sub["solve_rate"].mean()),
                lat_mean_s=float(lat_sub["lat_mean_s"].mean()),
                ms_per_token_mean=float(lat_sub["ms_per_token_mean"].mean()),
                answers_per_s=float(sub["answers_per_s"].mean()), tokens_per_s=float(sub["tokens_per_s"].mean()),
            ))
    df = pd.DataFrame(rows, columns=list(SUMMARY6_COLUMNS))
    base = {r.model: r.lat_mean_s for r in df.itertuples() if r.cond == 1}
    df["speedup_vs_c1"] = [base.get(m, np.nan) / l for m, l in zip(df["model"], df["lat_mean_s"])]
    return df


def comparisons(results_dir, pairs=COMPARISONS):
    """Per modello e coppia (a, b) di COMPARISONS presente: diff = acc_b - acc_a e McNemar sommando i seed presenti in entrambi.
    n01 = a sbagliato e b giusto; n10 = a giusto e b sbagliato; p_per_seed = p di ogni seed, in ordine."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    rows = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        sub = runs[runs["model"] == model]
        conds = _conds(runs, model)
        for a, b in pairs:
            if a not in conds or b not in conds:
                continue
            ra = sub[sub["cond"] == a].set_index("seed")["run_id"]
            rb = sub[sub["cond"] == b].set_index("seed")["run_id"]
            seeds = sorted(set(ra.index) & set(rb.index))
            if not seeds:
                continue
            cells_a, cells_b, per_seed = [], [], []
            n01 = n10 = 0
            for seed in seeds:
                sa = _correct(load_outputs(results_dir, ra[seed]))
                sb = _correct(load_outputs(results_dir, rb[seed]))
                common = sa.index.intersection(sb.index)
                x = sa.loc[common].to_numpy()
                y = sb.loc[common].to_numpy()
                cells_a.append(x)
                cells_b.append(y)
                s01 = int(((x == 0) & (y == 1)).sum())
                s10 = int(((x == 1) & (y == 0)).sum())
                n01 += s01
                n10 += s10
                per_seed.append(mcnemar(s01, s10))
            x = np.concatenate(cells_a)
            y = np.concatenate(cells_b)
            rows.append(dict(model=model, cond_a=a, cond_b=b, label_a=COND_LABELS[a], label_b=COND_LABELS[b],
                             n_cells=len(x), acc_diff=float(y.mean() - x.mean()), n01=n01, n10=n10,
                             p=mcnemar(n01, n10), p_per_seed=" / ".join(f"{p:.3g}" for p in per_seed)))
    return pd.DataFrame(rows, columns=list(COMPARISON6_COLUMNS))


def _val(idx, cond, col):
    """Valore di una colonna per condizione (indice = cond); NaN se la condizione manca."""
    return float(idx.loc[cond, col]) if cond in idx.index else float("nan")


def decomposition(summary):
    """Per modello: contributi per differenza di accuratezza (gap_A_C5, gram, calc, c6_gram, residual) e rapporti 8/2."""
    rows = []
    for model, g in summary.groupby("model", sort=False):
        idx = g.set_index("cond")
        rows.append(dict(
            model=model,
            gap_A_C5=_val(idx, 1, "acc") - _val(idx, 3, "acc"),
            gram=_val(idx, 4, "acc") - _val(idx, 3, "acc"),
            calc=_val(idx, 5, "acc") - _val(idx, 4, "acc"),
            c6_gram=_val(idx, 8, "acc") - _val(idx, 5, "acc"),
            residual=_val(idx, 8, "acc") - _val(idx, 2, "acc"),
            tok_ratio_8_2=_val(idx, 8, "tok_mean") / _val(idx, 2, "tok_mean"),
            lat_ratio_8_2=_val(idx, 8, "lat_mean_s") / _val(idx, 2, "lat_mean_s"),
        ))
    return pd.DataFrame(rows, columns=list(DECOMPOSITION_COLUMNS))


def plot6(summary, out_dir):
    """Un png per modello con due pannelli: accuratezza con IC in funzione dei token medi e della latenza media."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, g in summary.groupby("model", sort=False):
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        acc = g["acc"].to_numpy(dtype=float)
        yerr = np.vstack([acc - g["acc_lo"].to_numpy(dtype=float), g["acc_hi"].to_numpy(dtype=float) - acc])
        axes = fig.subplots(1, 2)
        panels = (("tok_mean", "token medi per risposta"), ("lat_mean_s", "latenza media per risposta (s)"))
        for ax, (xcol, xlabel) in zip(axes, panels):
            x = g[xcol].to_numpy(dtype=float)
            ax.errorbar(x, acc, yerr=yerr, fmt="o", capsize=4)
            for xi, yi, cond, label in zip(x, acc, g["cond"], g["label"]):
                ax.annotate(f"{cond} {label}", (xi, yi), textcoords="offset points", xytext=(5, 5))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("accuracy (test)")
        fig.suptitle(model)
        path = out_dir / f"acc_vs_tokens_{model}.png"
        fig.savefig(path, dpi=120)
        paths.append(path)
    return paths


def sample_errors6(results_dir, conds=(2, 5, 8), k=20, seed=0):
    """Fino a k risposte sbagliate per modello x condizione dalla run di seed 0 (split test)."""
    results_dir = Path(results_dir)
    runs = _runs(results_dir)
    runs = runs[runs["seed"] == 0]
    questions = _questions()
    rng = random.Random(seed)
    out = []
    for model in _order(runs["model"].unique(), list(MODELS)):
        for cond in conds:
            hit = runs[(runs["model"] == model) & (runs["cond"] == cond)]
            if hit.empty:
                continue
            df = load_outputs(results_dir, hit["run_id"].iloc[0])
            wrong = [r for r in df.to_dict("records") if not r["correct"]]
            for r in rng.sample(wrong, min(k, len(wrong))):
                out.append({
                    "model": model, "cond": cond, "label": COND_LABELS[cond], "id": r["id"],
                    "question": questions.get(r["id"], ""), "output": r["output"], "pred": r["pred"],
                    "gold": r["gold"], "conform": bool(r["conform"]),
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


def write_all6(results_dir):
    """Scrive summary.csv, comparisons.csv, decomposition.csv, errors_sample.jsonl e i grafici in plots/; ritorna il markdown."""
    results_dir = Path(results_dir)
    summary = summary6(results_dir)
    comp = comparisons(results_dir)
    dec = decomposition(summary)
    summary.to_csv(results_dir / "summary.csv", index=False)
    comp.to_csv(results_dir / "comparisons.csv", index=False)
    dec.to_csv(results_dir / "decomposition.csv", index=False)
    with open(results_dir / "errors_sample.jsonl", "w", encoding="utf-8") as f:
        for e in sample_errors6(results_dir):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    plot6(summary, results_dir / "plots")
    return ("## Riepilogo\n\n" + _md(summary) + "\n\n## Confronti a coppie\n\n" + _md(comp)
            + "\n\n## Scomposizione\n\n" + _md(dec) + "\n")
