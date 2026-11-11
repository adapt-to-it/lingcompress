"""Fase 0bis v4, controllo aggiuntivo: pieno bpe4096 0,38M seed 1, allineato (V 3840) e non allineato (3814), stesso codice."""
import functools

import pandas as pd

from src.v2 import corpus
from src.v2.spec import Job
from src.v3 import train3
from src.v3.fastgen import evaluate_fast
from src.v4.spec4 import V4_RESULTS, V4_RUNS

P = corpus.load_persons(20000)
righe = []
for stage, pad in (("F_ctrl", 64), ("F_ctrl_nopad", 1)):
    for seed in (0, 1):
        if stage == "F_ctrl" and seed == 0:
            continue   # gia' in align_control.csv
        row = train3.run_job(Job(stage, "italiano_pieno", "passi", 200, "0.38M", 2.0, seed), P,
                             evaluate=functools.partial(evaluate_fast, max_new_tokens=24), runs_dir=V4_RUNS, vocab_pad=pad)
        righe.append({"seed": seed, "V_modello": 3840 if pad == 64 else 3814, "acc_mean": row["acc_mean"],
                      "final_loss": row["final_loss"], "train_time_s": row["train_time_s"]})
        print(righe[-1], flush=True)
pd.DataFrame(righe).to_csv(V4_RESULTS / "align_control2.csv", index=False)
