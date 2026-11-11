"""Test di src/analysis6.py su file sintetici: 1 modello, condizioni 1-8, 2 seed, 10 problemi."""
import csv
import json
import math

import pytest

from src import analysis6
from src.analysis6 import (algebra_ids, comparisons, decomposition, sample_errors6, summary6, wrong_equations,
                           write_all6)
from src.spec6 import CONDITIONS, LATENCY6_COLUMNS, RUN6_COLUMNS

# Correttezza per condizione: (seed 0, seed 1) su p0..p9.
CORRECT = {
    1: ([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
    2: ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
    3: ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    4: ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    5: ([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    6: ([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
    7: ([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
    8: ([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
}
# Accuratezza attesa (media su 20 risposte): 13, 15, 9, 11, 17, 7, 14, 18 su 20.
# Equazioni sbagliate: (condizione, indice del seed, problema) -> output con "5 * 5 = 26".
WRONG_EQ = {(8, 0, 0), (8, 0, 1), (8, 1, 0)}
TOK = {1: 20, 2: 20, 3: 10, 4: 10, 5: 10, 6: 8, 7: 8, 8: 8}
LAT = {1: 2.0, 2: 2.5, 3: 1.0, 4: 1.2, 5: 1.5, 6: 1.1, 7: 1.3, 8: 2.0}
SEEDS = (0, 1)


def _build(root):
    """Scrive runs.csv, latency.csv e outputs/ sotto root e ritorna root."""
    (root / "outputs").mkdir(parents=True)
    with open(root / "runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN6_COLUMNS)
        w.writeheader()
        for cond in CORRECT:
            for seed in SEEDS:
                run_id = f"small_c{cond}_s{seed}"
                spec = CONDITIONS[cond]
                w.writerow(dict(run_id=run_id, model="small", cond=cond, format=spec["fmt"],
                                grammar=spec["grammar"], calc=spec["calc"], seed=seed, split="test", n_eval=10,
                                accuracy=0.5, conform_rate=1.0, tok_mean=TOK[cond], tok_median=TOK[cond],
                                forced_mean=0.0, throughput_time_s=1.0, answers_per_s=10.0, tokens_per_s=100.0,
                                note_rate=0.0, solve_rate=0.0))
    with open(root / "latency.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LATENCY6_COLUMNS)
        w.writeheader()
        for cond in CORRECT:
            for seed in SEEDS:
                w.writerow(dict(run_id=f"small_c{cond}_s{seed}", model="small", cond=cond,
                                format=CONDITIONS[cond]["fmt"], n=10, lat_mean_s=LAT[cond],
                                lat_median_s=LAT[cond], ms_per_token_mean=1000 * LAT[cond] / TOK[cond],
                                tok_mean=TOK[cond]))
    for cond, pair in CORRECT.items():
        for seed, correct in zip(SEEDS, pair):
            with open(root / "outputs" / f"small_c{cond}_s{seed}_test.jsonl", "w", encoding="utf-8") as f:
                for i, c in enumerate(correct):
                    wrong = (cond, seed, i) in WRONG_EQ
                    row = dict(id=f"p{i}", output="5 * 5 = 26" if wrong else "5 + 5 = 10", n_tokens=TOK[cond],
                               n_forced=1 if cond >= 6 else 0, pred="1", gold="1", correct=bool(c),
                               conform=not (cond == 5 and seed == 0 and i == 9))
                    f.write(json.dumps(row) + "\n")
    return root


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis6, "algebra_ids", lambda: {"p0", "p1", "p2"})
    return _build(tmp_path)


def _row(df, cond):
    return df[df["cond"] == cond].iloc[0]


def test_wrong_equations_counts_bad_and_total():
    assert wrong_equations("5 * 5 = 26 and 5 + 5 = 10") == (1, 2)
    assert wrong_equations("no equations here") == (0, 0)


def test_algebra_ids_matches_letter_unknowns(tmp_path, monkeypatch):
    rows = [{"id": "t0", "question": "q", "answer": "Let x be the number. 2x = 10.\n#### 5"},
            {"id": "t1", "question": "q", "answer": "She has 3 apples and 4 bags.\n#### 12"}]
    with open(tmp_path / "test.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(analysis6, "GSM8K", tmp_path)
    assert algebra_ids() == {"t0"}


def test_summary_ci_contains_mean(results):
    df = summary6(results)
    assert list(df["cond"]) == list(range(1, 9))
    row = _row(df, 1)
    assert row["n_seeds"] == 2
    assert row["acc"] == pytest.approx(13 / 20)
    assert row["acc_lo"] <= row["acc"] <= row["acc_hi"]
    assert row["tok_mean"] == pytest.approx(20)
    assert row["conform_rate"] == pytest.approx(1.0)


def test_summary_recovery_and_frac_wrong_eq(results):
    df = summary6(results)
    row8 = _row(df, 8)
    assert row8["frac_wrong_eq"] == pytest.approx(3 / 20)
    # risposte con equazione sbagliata: corrette in 2 casi su 3
    assert row8["recovery"] == pytest.approx(2 / 3)
    assert math.isnan(_row(df, 1)["recovery"])
    assert row8["acc_algebra"] == pytest.approx(5 / 6)
    assert row8["forced_mean"] == pytest.approx(1.0)
    assert _row(df, 5)["conform_rate"] == pytest.approx(19 / 20)


def test_summary_speedup_vs_c1(results):
    df = summary6(results)
    assert _row(df, 1)["speedup_vs_c1"] == pytest.approx(1.0)
    assert _row(df, 3)["speedup_vs_c1"] == pytest.approx(2.0)
    assert _row(df, 8)["speedup_vs_c1"] == pytest.approx(1.0)


def test_comparisons_pooled_mcnemar(results):
    df = comparisons(results)
    row = df[(df["cond_a"] == 3) & (df["cond_b"] == 4)].iloc[0]
    assert row["n_cells"] == 20
    assert row["acc_diff"] == pytest.approx(0.10)
    assert row["n01"] == 2
    assert row["n10"] == 0
    assert row["p"] == pytest.approx(0.5)
    assert row["p_per_seed"] == "1 / 1"


def test_decomposition_values(results):
    dec = decomposition(summary6(results)).iloc[0]
    assert dec["gap_A_C5"] == pytest.approx(0.65 - 0.45)
    assert dec["gram"] == pytest.approx(0.55 - 0.45)
    assert dec["calc"] == pytest.approx(0.85 - 0.55)
    assert dec["c6_gram"] == pytest.approx(0.90 - 0.85)
    assert dec["residual"] == pytest.approx(0.90 - 0.75)
    assert dec["tok_ratio_8_2"] == pytest.approx(8 / 20)
    assert dec["lat_ratio_8_2"] == pytest.approx(2.0 / 2.5)


def test_sample_errors_only_wrong_seed0(results):
    errors = sample_errors6(results, k=20, seed=0)
    assert len(errors) == 4
    assert {(e["cond"], e["id"]) for e in errors} == {(2, "p8"), (2, "p9"), (5, "p9"), (8, "p0")}
    assert set(errors[0]) == {"model", "cond", "label", "id", "question", "output", "pred", "gold", "conform"}


def test_write_all6_writes_files(results):
    md = write_all6(results)
    assert (results / "summary.csv").exists()
    assert (results / "comparisons.csv").exists()
    assert (results / "decomposition.csv").exists()
    assert (results / "errors_sample.jsonl").exists()
    assert (results / "plots" / "acc_vs_tokens_small.png").exists()
    assert "## Riepilogo" in md
    assert "## Confronti a coppie" in md
    assert "## Scomposizione" in md
