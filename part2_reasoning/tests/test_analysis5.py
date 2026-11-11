"""Test di src/analysis5.py su file sintetici: 1 modello, 2 formati, 2 seed, 10 problemi."""
import csv
import json

import numpy as np
import pytest

from src.analysis5 import bootstrap_ci, mcnemar, pairwise, sample_errors, summary_table, write_all
from src.spec5 import LATENCY_COLUMNS, OUTPUT_FIELDS, RUN_COLUMNS

# Correttezza per problema (id p0..p9) per run; A sempre con 20 token, B con 5.
CORRECT = {
    ("small_A_lr0.0002_s0"): [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    ("small_A_lr0.0002_s1"): [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ("small_B_lr0.0002_s0"): [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    ("small_B_lr0.0002_s1"): [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
}
RUNS = {
    "small_A_lr0.0002_s0": ("small", "A", 0, "main"),
    "small_A_lr0.0002_s1": ("small", "A", 1, "main"),
    "small_B_lr0.0002_s0": ("small", "B", 0, "main"),
    "small_B_lr0.0002_s1": ("small", "B", 1, "main"),
    "small_A_lr0.0001_s0": ("small", "A", 0, "sweep"),
}
TOKENS = {"A": 20, "B": 5}
LAT = {"A": 2.0, "B": 1.0}


def _build(root):
    """Scrive runs.csv, latency.csv e outputs/ sotto root e ritorna root."""
    (root / "outputs").mkdir(parents=True)
    with open(root / "runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        w.writeheader()
        for run_id, (model, fmt, seed, stage) in RUNS.items():
            w.writerow(dict(run_id=run_id, model=model, format=fmt, lr=2e-4, seed=seed, stage=stage, n_train=100,
                            steps=10, train_time_s=1.0, final_loss=0.5, split="test", n_eval=10, accuracy=0.5,
                            conform_rate=1.0, tok_mean=1.0, tok_median=1.0, throughput_time_s=1.0,
                            answers_per_s=10.0 if fmt == "A" else 20.0, tokens_per_s=100.0))
    with open(root / "latency.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LATENCY_COLUMNS)
        w.writeheader()
        for run_id, (model, fmt, seed, stage) in RUNS.items():
            lat = 10.0 if stage == "sweep" else LAT[fmt]    # la sweep non deve entrare nella latenza
            w.writerow(dict(run_id=run_id, model=model, format=fmt, n=200, lat_mean_s=lat, lat_median_s=lat,
                            ms_per_token_mean=1000 * lat / TOKENS[fmt], tok_mean=TOKENS[fmt]))
    for run_id, correct in CORRECT.items():
        fmt = RUNS[run_id][1]
        with open(root / "outputs" / f"{run_id}_test.jsonl", "w", encoding="utf-8") as f:
            for i, c in enumerate(correct):
                row = dict(id=f"p{i}", output="x", n_tokens=TOKENS[fmt], pred="1", gold="1",
                           correct=bool(c), conform=not (run_id == "small_A_lr0.0002_s0" and i == 9))
                f.write(json.dumps(row) + "\n")
    return root


@pytest.fixture
def results(tmp_path):
    return _build(tmp_path)


def test_bootstrap_ci_contains_mean():
    correct = np.array([[1, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1]])
    mean, lo, hi = bootstrap_ci(correct)
    assert mean == pytest.approx(correct.mean())
    assert lo <= mean <= hi


def test_bootstrap_ci_is_deterministic():
    correct = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    assert bootstrap_ci(correct) == bootstrap_ci(correct)


def test_mcnemar_exact():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(0, 4) == pytest.approx(0.125)
    assert mcnemar(2, 2) == pytest.approx(1.0)


def test_summary_acc_and_speedup(results):
    df = summary_table(results).set_index("format")
    assert list(df.index) == ["A", "B"]
    assert df.loc["A", "n_seeds"] == 2
    assert df.loc["A", "acc"] == pytest.approx(13 / 20)
    assert df.loc["B", "acc"] == pytest.approx(9 / 20)
    assert df.loc["A", "acc_lo"] <= df.loc["A", "acc"] <= df.loc["A", "acc_hi"]
    assert df.loc["A", "conform_rate"] == pytest.approx(19 / 20)
    assert df.loc["A", "tok_mean"] == pytest.approx(20)
    assert df.loc["B", "tok_median"] == pytest.approx(5)
    # latenza: solo run main, la sweep (10 s) resta fuori
    assert df.loc["A", "lat_mean_s"] == pytest.approx(2.0)
    assert df.loc["A", "speedup_vs_A"] == pytest.approx(1.0)
    assert df.loc["B", "speedup_vs_A"] == pytest.approx(2.0)
    assert df.loc["B", "answers_per_s"] == pytest.approx(20.0)


def test_pairwise_mcnemar_pooled(results):
    df = pairwise(results)
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["format_a"], row["format_b"]) == ("A", "B")
    assert row["n_cells"] == 20
    assert row["n01"] == 0
    assert row["n10"] == 4
    assert row["acc_diff"] == pytest.approx(0.2)
    assert row["p"] == pytest.approx(0.125)


def test_sample_errors_only_wrong_seed0(results):
    errors = sample_errors(results, k=20, seed=0)
    assert {e["format"] for e in errors} == {"A", "B"}
    a_ids = {e["id"] for e in errors if e["format"] == "A"}
    assert a_ids == {"p7", "p8", "p9"}
    assert len([e for e in errors if e["format"] == "B"]) == 5
    assert all(e["question"] == "" for e in errors)
    assert set(errors[0]) == {"model", "format", "id", "question", "output", "pred", "gold", "conform"}


def test_write_all_writes_files(results):
    md = write_all(results)
    assert (results / "summary.csv").exists()
    assert (results / "pairwise.csv").exists()
    assert (results / "errors_sample.jsonl").exists()
    assert (results / "plots" / "acc_vs_tokens_small.png").exists()
    assert "## Riepilogo" in md
    assert "## Confronti a coppie" in md
    assert "| A |" in md
