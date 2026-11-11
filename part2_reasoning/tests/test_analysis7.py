"""Test di src/analysis7.py su file sintetici: modelli small e large, condizioni 1, 2, 3, 5, 9-12, 2 seed, 10 problemi."""
import csv
import json
import math

import pandas as pd
import pytest

from src import analysis6
from src.analysis6 import summary6
from src.analysis7 import pareto, scale_trend, write_all7
from src.spec6 import CONDITIONS, LATENCY6_COLUMNS, RUN6_COLUMNS

SEEDS = (0, 1)
# Problemi corretti per condizione (uguali nei due seed, quindi accuratezza = k / 10).
CORRECT_N = {
    "small": {1: 5, 2: 6, 3: 4, 5: 5, 9: 3, 10: 7, 11: 6, 12: 6},
    "large": {1: 4, 2: 5, 3: 3, 5: 6, 9: 5, 10: 7, 11: 4, 12: 8},
}
TOK = {
    "small": {1: 20, 2: 20, 3: 10, 5: 10, 9: 12, 10: 14, 11: 8, 12: 9},
    "large": {1: 20, 2: 20, 3: 10, 5: 10, 9: 12, 10: 15, 11: 8, 12: 9},
}


def _build(root):
    """Scrive runs.csv, latency.csv e outputs/ sotto root e ritorna root."""
    (root / "outputs").mkdir(parents=True)
    with open(root / "runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN6_COLUMNS)
        w.writeheader()
        for model, conds in CORRECT_N.items():
            for cond in conds:
                for seed in SEEDS:
                    spec = CONDITIONS[cond]
                    w.writerow(dict(run_id=f"{model}_c{cond}_s{seed}", model=model, cond=cond, format=spec["fmt"],
                                    grammar=spec["grammar"], calc=spec["calc"], seed=seed, split="test", n_eval=10,
                                    accuracy=0.5, conform_rate=1.0, tok_mean=TOK[model][cond],
                                    tok_median=TOK[model][cond], forced_mean=0.0, throughput_time_s=1.0,
                                    answers_per_s=10.0, tokens_per_s=100.0, note_rate=0.0, solve_rate=0.0))
    with open(root / "latency.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LATENCY6_COLUMNS)
        w.writeheader()
        for model, conds in CORRECT_N.items():
            for cond in conds:
                for seed in SEEDS:
                    lat = TOK[model][cond] / 10.0
                    w.writerow(dict(run_id=f"{model}_c{cond}_s{seed}", model=model, cond=cond,
                                    format=CONDITIONS[cond]["fmt"], n=10, lat_mean_s=lat, lat_median_s=lat,
                                    ms_per_token_mean=100.0, tok_mean=TOK[model][cond]))
    for model, conds in CORRECT_N.items():
        for cond, k in conds.items():
            for seed in SEEDS:
                with open(root / "outputs" / f"{model}_c{cond}_s{seed}_test.jsonl", "w", encoding="utf-8") as f:
                    for i in range(10):
                        row = dict(id=f"p{i}", output="5 + 5 = 10", n_tokens=TOK[model][cond], n_forced=0,
                                   pred="1", gold="1", correct=i < k, conform=True)
                        f.write(json.dumps(row) + "\n")
    return root


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis6, "algebra_ids", lambda: set())
    return _build(tmp_path)


def _frontier(df):
    return {(r.model, r.cond) for r in df.itertuples() if r.frontier}


def test_pareto_frontier_flags(results):
    df = pareto(summary6(results))
    # small: c10 (acc 0.7, 14 token) e c11 (0.6, 8 token); large: c11 (0.4, 8 token) e c12 (0.8, 9 token).
    assert _frontier(df) == {("small", 10), ("small", 11), ("large", 11), ("large", 12)}
    assert df["frontier"].dtype == bool


def test_pareto_ties_are_not_dominated():
    s = pd.DataFrame([dict(model="m", cond=1, acc=0.5, tok_mean=10.0),
                      dict(model="m", cond=2, acc=0.5, tok_mean=10.0),
                      dict(model="m", cond=3, acc=0.4, tok_mean=10.0)])
    df = pareto(s)
    assert list(df["frontier"]) == [True, True, False]
    assert "frontier" not in s.columns


def test_scale_trend_values(results):
    scale = scale_trend(pareto(summary6(results))).set_index("model")
    s = scale.loc["small"]
    assert s["acc_2"] == pytest.approx(0.6)
    assert s["acc_5"] == pytest.approx(0.5)
    assert s["gap_5_2"] == pytest.approx(-0.1)
    assert s["best_concise"] == 10
    assert s["acc_best"] == pytest.approx(0.7)
    assert s["gap_best_2"] == pytest.approx(0.1)
    assert s["tok_ratio_5_2"] == pytest.approx(10 / 20)
    assert s["tok_ratio_best_2"] == pytest.approx(14 / 20)
    g = scale.loc["large"]
    assert g["acc_2"] == pytest.approx(0.5)
    assert g["acc_5"] == pytest.approx(0.6)
    assert g["gap_5_2"] == pytest.approx(0.1)
    assert g["best_concise"] == 12
    assert g["acc_best"] == pytest.approx(0.8)
    assert g["gap_best_2"] == pytest.approx(0.3)
    assert g["tok_ratio_5_2"] == pytest.approx(10 / 20)
    assert g["tok_ratio_best_2"] == pytest.approx(9 / 20)


def test_scale_trend_without_concise_conditions():
    s = pd.DataFrame([dict(model="m", cond=2, acc=0.5, tok_mean=10.0), dict(model="m", cond=5, acc=0.6, tok_mean=5.0)])
    row = scale_trend(s).iloc[0]
    assert row["best_concise"] is None
    assert math.isnan(row["acc_best"])
    assert math.isnan(row["gap_best_2"])
    assert math.isnan(row["tok_ratio_best_2"])
    assert row["tok_ratio_5_2"] == pytest.approx(0.5)


def test_write_all7_writes_files(results):
    md = write_all7(results)
    for name in ("summary.csv", "comparisons.csv", "scale.csv", "errors_sample.jsonl"):
        assert (results / name).exists()
    assert (results / "plots" / "frontier_small.png").exists()
    assert (results / "plots" / "frontier_large.png").exists()
    for section in ("## Riepilogo", "## Frontiera", "## Confronti a coppie", "## Recupero", "## Scala"):
        assert section in md
    with open(results / "errors_sample.jsonl", encoding="utf-8") as f:
        errors = [json.loads(line) for line in f if line.strip()]
    assert errors
    assert {e["model"] for e in errors} == {"large"}
    assert {e["cond"] for e in errors} <= {10, 12}
