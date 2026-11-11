"""Controllo: stesso formato ed esposizioni, batch diverso per pareggiare il numero di passi.

Esempio: python control_steps.py --format telegrafico --batch 128 --sizes 0.25M,0.5M,1M
Scrive in results/results_ctrl_steps.csv.
"""
import argparse
import csv
from dataclasses import replace

from src.config import MODEL_SIZES, RESULT_COLUMNS, RESULTS_DIR, full_config
from src.data import generate_persons
from src.pipeline import run_one

ap = argparse.ArgumentParser()
ap.add_argument("--format", required=True)
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--sizes", default="0.25M,0.5M,1M")
ap.add_argument("--seeds", default="0,1")
a = ap.parse_args()

cfg = replace(full_config(), batch_size=a.batch, tag=f"ctrl_b{a.batch}")
persons = generate_persons(cfg.n_persons, cfg.data_seed)
out = RESULTS_DIR / "results_ctrl_steps.csv"
for size in a.sizes.split(","):
    for seed in map(int, a.seeds.split(",")):
        row = run_one(persons, a.format, MODEL_SIZES[size], seed, cfg)
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{a.format} {size} seed {seed} batch {a.batch}: steps {row['steps']} acc {row['acc_mean']:.3f}", flush=True)
