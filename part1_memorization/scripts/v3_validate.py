"""Fase 0 v3: rieseguo 3 configurazioni v2 (seed 0) con il trainer ottimizzato e confronto acc e tempi."""
import time

import pandas as pd

from src.v2 import corpus
from src.v2.spec import Job, V2_RESULTS
from src.v3 import train3
from src.v3.spec3 import V3_RESULTS

CONFIGS = [("italiano_pieno", "0.38M"), ("italiano_pieno", "0.60M"), ("telegrafico", "0.38M")]
v2 = pd.read_csv(V2_RESULTS / "A_main.csv")
persons = corpus.load_persons(20000)
rows = []
for c, s in CONFIGS:
    ref = v2[(v2.corpus == c) & (v2["size"] == s) & (v2.seed == 0)].iloc[0]
    job = Job("V3_val", c, "passi", 200, s, float(ref.lr_mult), 0)
    t0 = time.perf_counter()
    r = train3.run_job(job, persons)
    wall = time.perf_counter() - t0
    rows.append(dict(corpus=c, size=s, acc_v2=ref.acc_mean, acc_v3=r["acc_mean"],
                     delta=r["acc_mean"] - ref.acc_mean, loss_v2=ref.final_loss, loss_v3=r["final_loss"],
                     train_v2_s=ref.train_time_s, train_v3_s=r["train_time_s"],
                     eval_v2_s=ref.eval_time_s, eval_v3_s=r["eval_time_s"],
                     tok_s_v2=ref.steady_tokens_per_s, tok_s_v3=r["steady_tokens_per_s"], wall_v3_s=wall))
    print(rows[-1], flush=True)
V3_RESULTS.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(V3_RESULTS / "phase0_validation.csv", index=False)
print(df.round(4).to_string())
