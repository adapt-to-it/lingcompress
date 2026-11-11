"""Piano v3 a blocchi indipendenti e riprendibili (ordine concordato con l'utente).

E_train -> E_pipe -> D_timing -> D_bpe (512, 1024; lr check bpe512) -> D_char (seed 0; lr check) -> D_bpe2048.
Dopo ogni blocco: CSV in results/v3 e summary_v3.md rigenerato. Uso: python run_v3_plan.py [--from BLOCCO]
"""
import argparse
import functools
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from src.v2 import corpus, runner, train
from src.v2.model2 import GPT2
from src.v2.spec import JOB_KEYS, RESULT_COLUMNS_V2, SIZES_V2, V2_RESULTS, V2_RUNS, Job
from src.v3 import analysis3, pipeline3, strip, train3, vocab
from src.v3.fastgen import evaluate_fast
from src.v3.spec3 import (D_FORMATS, D_SIZES, E_FORMATS, E_SIZES, LAT_COLUMNS, LR_CHECK_MULTS, LR_CHECK_SIZE,
                          PIPE_BATCHES, PIPE_QUESTIONS, PIPE_SIZES, V3_RESULTS, V3_RUNS)

PERSONS = None
BEST = runner.best_lr(pd.read_csv(V2_RESULTS / "A_lr.csv"))
BASE = {"italiano_pieno": "italiano_pieno", "telegrafico": "telegrafico", "interlingua_v1enc": "interlingua",
        "telegrafico_auto": "telegrafico", "telegrafico_lista": "telegrafico"}


def v2_lr(fmt, size):
    return runner._lookup_lr(BEST, BASE[fmt], size)


def run_jobs(jobs, csv, evaluate_for, reuse=()):
    """Esegue i job mancanti in csv (chiave JOB_KEYS); riusa righe con stessa chiave (tranne lo stage) da reuse."""
    done = {runner._key(r) for r in runner._read_rows(csv)}
    donors = {runner._key(r)[1:]: r for p in reuse for r in runner._read_rows(p)}
    for k, job in enumerate(jobs, 1):
        key = runner._key(vars(job))
        if key in done:
            continue
        row = ({**donors[key[1:]], "stage": job.stage} if key[1:] in donors
               else train3.run_job(job, PERSONS, evaluate=evaluate_for(job)))
        runner._append(csv, row)
        done.add(key)
        print(f"[{k}/{len(jobs)}] {job.run_id} acc_mean={row['acc_mean']:.3f} train {row['train_time_s']:.0f}s", flush=True)


def e_train():
    jobs = [Job("E_main", "telegrafico_auto", "passi", 200, s, v2_lr("telegrafico", s), sd)
            for s in E_SIZES for sd in (0, 1)]
    jobs += [Job("E_main", "telegrafico_auto", "passi", 200, s, v2_lr("telegrafico", s), 0) for s in PIPE_SIZES[1:]]
    jobs += [Job("E_main", "telegrafico_lista", "passi", 200, s, v2_lr("telegrafico", s), 0) for s in PIPE_SIZES]
    strippers = {n: (strip.SpacyStripper() if k == "spacy" else strip.ListStripper()) for n, k in E_FORMATS.items()}
    run_jobs(jobs, V3_RESULTS / "E_main.csv", lambda j: pipeline3.evaluate_stripped(strippers[j.corpus]))


def d_specs(v):
    out = {}
    for fmt in D_FORMATS:
        spec = vocab.register(fmt, v)
        if not train._tok_path(spec, len(PERSONS)).exists():
            vocab.build_tokenizer(spec, v, PERSONS)
        out[fmt] = spec
    return out


def d_eval(job):
    spec = train.CORPORA[job.corpus]
    k = vocab.max_answer_tokens(spec, train.get_tokenizer(spec, PERSONS))
    return functools.partial(evaluate_fast, max_new_tokens=k)


def d_block(vocabs, seeds, lr_check):
    lr_csv, main_csv = V3_RESULTS / "D_lr.csv", V3_RESULTS / "D_main.csv"
    for v in vocabs:
        specs = d_specs(v)
        scale = 1.0
        if v in lr_check:
            run_jobs([Job("D_lr", specs[f].name, "passi", 200, LR_CHECK_SIZE, m, 0)
                      for f in D_FORMATS for m in LR_CHECK_MULTS], lr_csv, d_eval)
            lr = pd.read_csv(lr_csv)
            best = runner.best_lr(lr[lr.corpus.str.endswith("@" + v)])
            # lr di v2 a 0.38M = 2x: scala il lr di tutte le taglie come il migliore della verifica (per formato)
            scale = {f: best[(specs[f].name, LR_CHECK_SIZE)] / v2_lr(f, LR_CHECK_SIZE) for f in D_FORMATS}
        jobs = [Job("D_main", specs[f].name, "passi", 200, s,
                    v2_lr(f, s) * (scale[f] if isinstance(scale, dict) else scale), sd)
                for f in D_FORMATS for s in D_SIZES for sd in seeds]
        run_jobs(jobs, main_csv, d_eval, reuse=(lr_csv,))
        d_latency(jobs)


def _load(row, runs_dir):
    spec = train.CORPORA[row["corpus"]]
    tok = train.get_tokenizer(spec, PERSONS)
    job = Job(*[row[k] for k in JOB_KEYS])
    path = Path(runs_dir) / job.run_id / "ckpt.pt"
    for donor in ("A_lr", "D_lr"):   # righe riusate dalla verifica lr: il checkpoint sta nella cartella del donatore
        if not path.exists():
            path = Path(runs_dir) / Job(*([donor] + [row[k] for k in JOB_KEYS[1:]])).run_id / "ckpt.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    z = SIZES_V2[row["size"]]
    m = GPT2(tok.get_vocab_size(), state["pos_emb.weight"].shape[0], z.n_layer, z.n_embd, z.n_head)
    m.load_state_dict(state)
    return m.cuda().eval(), tok, spec


def d_latency(jobs, rows=None, runs_dir=V3_RUNS):
    """Latenza batch=1 e batch=64 sui modelli seed 0 dei job (o delle righe date)."""
    out = V3_RESULTS / "D_latency.csv"
    have = {(r["corpus"], r["size"], r["batch"]) for r in runner._read_rows(out)}
    if rows is None:
        main = pd.read_csv(V3_RESULTS / "D_main.csv")
        ids = {(j.corpus, j.size) for j in jobs if j.seed == 0}
        rows = [r for r in main.to_dict("records") if (r["corpus"], r["size"]) in ids and r["seed"] == 0]
    for r in rows:
        model = None
        for b in PIPE_BATCHES:
            if (r["corpus"], r["size"], b) in have:
                continue
            if model is None:
                model, tok, spec = _load(r, runs_dir)
            m = vocab.latency(model, tok, spec, PERSONS, PIPE_QUESTIONS, b)
            fmt, _, v = r["corpus"].partition("@")
            row = {**m, "corpus": r["corpus"], "format": fmt, "vocab": v or "bpe4096", "size": r["size"],
                   "seed": 0, "batch": b, "n_questions": PIPE_QUESTIONS,
                   "params_nonemb": r["params_nonemb"], "params_total": r["params_total"]}
            pd.DataFrame([row], columns=list(LAT_COLUMNS)).to_csv(out, mode="a", header=not out.exists(), index=False)


def d_ref_latency():
    """Latenza dei modelli bpe4096 di v2 (A_main pieno/telegrafico, A_abl interlingua_v1enc), seed 0."""
    rows = []
    for stage, fmts in (("A_main", ("italiano_pieno", "telegrafico")), ("A_abl", ("interlingua_v1enc",))):
        df = pd.read_csv(V2_RESULTS / f"{stage}.csv")
        rows += df[df.corpus.isin(fmts) & df["size"].isin(D_SIZES) & (df.seed == 0)].to_dict("records")
    d_latency([], rows, V2_RUNS)


def d_latency_all():
    """Ricalcola D_latency.csv per tutti i modelli seed 0 (riferimento bpe4096 + D_main), dopo il fix del warm-up."""
    d_ref_latency()
    main = pd.read_csv(V3_RESULTS / "D_main.csv")
    d_latency([], main[main.seed == 0].to_dict("records"))


BLOCKS = {
    "E_train": e_train,
    "E_pipe": lambda: pipeline3.run_pipeline(V3_RESULTS / "E_pipeline.csv"),
    "D_timing": lambda: subprocess.run([sys.executable, "scripts/v3_timing.py"], check=True,
                                       env={**__import__("os").environ, "PYTHONPATH": "."}),
    "D_bpe": lambda: (d_ref_latency(), d_block(("bpe512", "bpe1024"), (0, 1), ("bpe512",))),
    "D_char": lambda: d_block(("char",), (0,), ("char",)),
    "D_bpe2048": lambda: d_block(("bpe2048",), (0, 1), ()),
    # rifiniture a GPU libera: latenze con warm-up dei CUDA graph corretto e scomposizione dei tempi robusta
    "D_latency_all": d_latency_all,
    "E_pipe_redo": lambda: pipeline3.run_pipeline(V3_RESULTS / "E_pipeline.csv"),
    "D_timing_redo": lambda: subprocess.run([sys.executable, "scripts/v3_timing.py"], check=True,
                                            env={**__import__("os").environ, "PYTHONPATH": "."}),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="E_train", choices=list(BLOCKS))
    ap.add_argument("--only", action="store_true")
    a = ap.parse_args()
    PERSONS = corpus.load_persons(20000)
    for name in E_FORMATS:
        strip.register(name)
    V3_RESULTS.mkdir(parents=True, exist_ok=True)
    names = list(BLOCKS)
    for name in names[names.index(a.start):][: 1 if a.only else None]:
        print(f"=== BLOCCO {name} INIZIO", flush=True)
        BLOCKS[name]()
        analysis3.write_all(V3_RESULTS)
        print(f"=== BLOCCO {name} COMPLETATO (summary aggiornato)", flush=True)
