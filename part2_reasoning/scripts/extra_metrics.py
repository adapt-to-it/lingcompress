"""Metriche aggiuntive sugli output dei run principali: equazioni sbagliate, troncamenti, McNemar per seed."""
import ast
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spec5 import MAX_NEW_TOKENS, RESULTS  # noqa: E402

EQ = re.compile(r"((?:\(?\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?%?\s*\)?\s*[-+*/x×]\s*)+\(?\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?%?\s*\)?)"
                r"\s*=\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def _eval(expr):
    e = expr.strip().replace(",", "").replace("$", "").replace("×", "*").replace("x", "*").replace("%", "/100")
    return eval(compile(ast.parse(e, mode="eval"), "<eq>", "eval"), {"__builtins__": {}})


def wrong_equations(text):
    """Numero di uguaglianze 'a op b = c' ricalcolabili e sbagliate (tolleranza 1% o 0,01)."""
    bad = tot = 0
    for expr, val in EQ.findall(str(text)):
        try:
            got, want = _eval(expr), float(val.replace(",", ""))
        except Exception:
            continue
        tot += 1
        bad += abs(got - want) > max(0.01, 0.01 * abs(want))
    return bad, tot


runs = pd.read_csv(RESULTS / "runs.csv")
runs = runs[(runs.stage == "main") & (runs.split == "test")]
rows, outs = [], {}
for r in runs.itertuples():
    df = pd.read_json(RESULTS / "outputs" / f"{r.run_id}_test.jsonl", lines=True, dtype={"output": str, "id": str})
    outs[(r.model, r.format, r.seed)] = df.set_index("id")["correct"].astype(int)
    eq = df["output"].map(wrong_equations)
    has_bad = eq.map(lambda t: t[0] > 0)
    rows.append({"model": r.model, "format": r.format, "seed": r.seed, "accuracy": df.correct.mean(),
                 "frac_wrong_equation": has_bad.mean(),
                 "acc_if_wrong_equation": df.correct[has_bad].mean() if has_bad.any() else float("nan"),
                 "frac_truncated": (df.n_tokens >= MAX_NEW_TOKENS[r.format]).mean(),
                 "acc_if_conform": df.correct[df.conform].mean(),
                 "acc_if_not_conform": df.correct[~df.conform].mean() if (~df.conform).any() else float("nan")})
per_run = pd.DataFrame(rows)
per_run.groupby(["model", "format"]).mean(numeric_only=True).drop(columns="seed").round(4).to_csv(RESULTS / "extra_metrics.csv")

mc = []
for (model, fa, sd), a in outs.items():
    for fb in "ABCD":
        if fb <= fa or (model, fb, sd) not in outs:
            continue
        b = outs[(model, fb, sd)].reindex(a.index)
        n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        mc.append({"model": model, "format_a": fa, "format_b": fb, "seed": sd, "acc_diff": a.mean() - b.mean(),
                   "n01": n01, "n10": n10, "p": p})
pd.DataFrame(mc).sort_values(["model", "format_a", "format_b", "seed"]).round(5).to_csv(RESULTS / "mcnemar_per_seed.csv", index=False)
print(pd.read_csv(RESULTS / "extra_metrics.csv").to_string(index=False))
print(pd.DataFrame(mc).query("format_b != 'D'").sort_values(["model", "format_a", "format_b", "seed"]).round(4).to_string(index=False))
