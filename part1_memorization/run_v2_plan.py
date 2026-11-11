"""Piano v2 concordato, a blocchi indipendenti e riprendibili.

Ordine: A_lr (3 taglie) -> A_main (8 taglie x 2 seed + 4M seed 0) -> A_abl -> B_main (3 seed) -> C (costi end-to-end)
-> A_shared -> A_exp E100 -> A_exp E500 -> A_esp. Dopo ogni blocco: report rigenerato.
Uso: python run_v2_plan.py [--from BLOCCO]
"""
import argparse

import pandas as pd

from src.v2.analysis import write_all
from src.v2.e2e import run_e2e
from src.v2.infer import run_infer_c
from src.v2.translate import run_translate
from src.v2.runner import best_lr, plan, run_stage
from src.v2.spec import SIZES_V2, V2_RESULTS

GRID = [s for s in SIZES_V2 if s != "4M"]   # 8 taglie log-spaziate
LR_SIZES = ("0.20M", "0.38M", "0.79M")
BASE = ("italiano_pieno", "telegrafico", "interlingua", "italiano_ricco")


def csv(stage):
    return V2_RESULTS / f"{stage}.csv"


def window(shift=0, k=6):
    """k taglie consecutive attorno alla transizione (50%) misurata in A_main, spostate di shift."""
    df = pd.read_csv(csv("A_main"))
    acc = df[df["size"].isin(GRID)].groupby(["corpus", "size"])["acc_mean"].mean()
    idx50 = []
    for c in BASE:
        vals = [acc.get((c, s), 0.0) for s in GRID]
        idx50.append(next((i for i, v in enumerate(vals) if v >= 0.5), len(GRID) - 1))
    start = min(max(min(idx50) - 2 + shift, 0), len(GRID) - k)
    return tuple(GRID[start:start + k])


def stage(phase, name, out=None, reuse=(), **ov):
    best = best_lr(pd.read_csv(csv("A_lr"))) if csv("A_lr").exists() else {}
    jobs = plan(phase, name, "full", best, **ov)
    run_stage(jobs, csv(out or name), reuse=reuse)


BLOCKS = {
    "A_lr": lambda: stage("A", "A_lr", sizes=LR_SIZES),
    "A_main": lambda: (stage("A", "A_main", reuse=(csv("A_lr"),), sizes=tuple(GRID), seeds=(0, 1)),
                       stage("A", "A_main", sizes=("4M",), seeds=(0,))),
    "A_abl": lambda: stage("A", "A_abl", sizes=window(), seeds=(0, 1)),
    "B_main": lambda: stage("B", "B_main", seeds=(0, 1, 2)),
    # Fase C: tempi di risposta (KV cache), stripper/traduttore, pipeline end-to-end; GPU libera
    "C": lambda: (run_infer_c(V2_RESULTS / "inference_c.csv"),
                  run_translate(V2_RESULTS / "translate_c.csv"),
                  run_e2e(V2_RESULTS / "e2e_c.csv")),
    "A_shared": lambda: stage("A", "A_shared", sizes=window(), seeds=(0, 1)),
    "A_exp100": lambda: stage("A", "A_exp", sizes=window(+1), seeds=(0, 1), exposures=(100,),
                              corpora=BASE[:3]),
    "A_exp500": lambda: stage("A", "A_exp", sizes=window(-1), seeds=(0, 1), exposures=(500,),
                              corpora=BASE[:3]),
    "A_esp": lambda: stage("A", "A_esp", sizes=window(), seeds=(0, 1)),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="A_lr", choices=list(BLOCKS))
    names = list(BLOCKS)
    for name in names[names.index(ap.parse_args().start):]:
        print(f"=== BLOCCO {name} INIZIO", flush=True)
        BLOCKS[name]()
        write_all(V2_RESULTS)
        print(f"=== BLOCCO {name} COMPLETATO (report aggiornato)", flush=True)
