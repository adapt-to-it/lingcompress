"""Piano v5 a stadi riprendibili: build (dati SFT + statistiche), quick (pipeline su 200 esempi), sweep (lr su A,
modello piccolo, dev), main (modelli x formati x seed). Dopo ogni modello x formato: CSV, analisi e report.
Uso: python run_v5.py STADIO [--seeds 0 1] [--models small large] [--formats A B C D]
"""
import argparse
import gc
import json
import statistics
import sys
import time

import pandas as pd
import torch

from src import analysis5, eval5, rewrite, train5
from src.spec5 import (DEV_SIZE, FORMAT_NAMES, FORMATS, LR_SWEEP, MODELS, RESULTS, RUN_COLUMNS, LATENCY_COLUMNS,
                       REWRITES, RUNS, SEEDS, SFT)

OUT = RESULTS / "outputs"
REPORT = RESULTS.parent.parent / "REPORT_v5.md"
FINALE = "<!-- FINALE -->"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(csv, row, columns):
    pd.DataFrame([row], columns=list(columns)).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def _done(csv, run_id, split):
    if not csv.exists():
        return False
    d = pd.read_csv(csv)
    return bool(((d.run_id == run_id) & (d.split == split)).any())


def build():
    """File SFT, conteggi dei validi, token medi per soluzione (tokenizer Qwen3) ed esempi dello stesso problema."""
    from transformers import AutoTokenizer
    counts = rewrite.build_sft()
    print("SFT:", counts, flush=True)
    tok = AutoTokenizer.from_pretrained(MODELS["small"])
    rows = []
    for fmt in ("A", "C"):
        last = {}
        for r in _jsonl(REWRITES / f"{fmt}.jsonl"):
            last[r["id"]] = r
        ok = sum(r["ok"] for r in last.values())
        rows.append({"format": fmt, "valid": ok, "generated": len(last)})
    data = {fmt: _jsonl(SFT / f"{fmt}_train.jsonl") for fmt in FORMATS}
    stats = []
    for fmt in FORMATS:
        n = [len(tok(r["solution"]).input_ids) for r in data[fmt]]
        v = next((r["valid"] for r in rows if r["format"] == fmt), None)
        stats.append({"format": fmt, "name": FORMAT_NAMES[fmt], "valid_generated": v if v is not None else "derivato",
                      "n_common": len(n), "tok_mean": statistics.mean(n), "tok_median": statistics.median(n)})
    pd.DataFrame(stats).to_csv(RESULTS / "data_stats.csv", index=False)
    ex = ["# Esempi v5: stesso problema, quattro formati", ""]
    for pid in [data["A"][i]["id"] for i in (5, 400, 3000)]:
        q = next(r for r in data["A"] if r["id"] == pid)["question"]
        ex += [f"## {pid}", "", f"**Domanda.** {q}", ""]
        for fmt in FORMATS:
            sol = next(r for r in data[fmt] if r["id"] == pid)["solution"]
            ex += [f"**{fmt} ({FORMAT_NAMES[fmt]})**", "", "```", sol, "```", ""]
    (RESULTS / "examples_formats.md").write_text("\n".join(ex), encoding="utf-8")
    print(pd.DataFrame(stats).to_string(index=False), flush=True)


def _test_rows():
    return [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in _jsonl(SFT / "test.jsonl")]


def run_one(model_key, fmt, lr, seed, stage, split, eval_rows, train_rows=None, do_latency=False, tag=""):
    """Training (se l'adapter non c'e') + valutazione su split; righe in runs.csv e latency.csv."""
    rid = train5.run_id(model_key, fmt, lr, seed) + tag
    runs_csv = RESULTS / "runs.csv"
    if _done(runs_csv, rid, split):
        return
    tj = RUNS / rid / "train.json"
    if not tj.exists():
        info = train5.train(model_key, fmt, lr, seed, rows=train_rows, out_dir=RUNS / rid / "adapter")
        print(f"train {rid}: {info['steps']} step, {info['train_time_s']:.0f}s, loss {info['final_loss']:.3f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    info = json.loads(tj.read_text(encoding="utf-8"))
    model, tok = eval5.load_model(model_key, RUNS / rid / "adapter")
    OUT.mkdir(parents=True, exist_ok=True)
    m = eval5.evaluate(model, tok, fmt, eval_rows, out_path=OUT / f"{rid}_{split}.jsonl")
    row = {"run_id": rid, "model": model_key, "format": fmt, "lr": lr, "seed": seed, "stage": stage,
           "n_train": info["n_train"], "steps": info["steps"], "train_time_s": info["train_time_s"],
           "final_loss": info["final_loss"], "split": split, **m}
    _append(runs_csv, row, RUN_COLUMNS)
    print(f"eval {rid} {split}: acc {m['accuracy']:.3f} conf {m['conform_rate']:.3f} tok {m['tok_mean']:.1f} "
          f"{m['throughput_time_s']:.0f}s", flush=True)
    if do_latency:
        lat, items = eval5.latency(model, tok, fmt, eval_rows)
        _append(RESULTS / "latency.csv", {"run_id": rid, "model": model_key, "format": fmt, **lat}, LATENCY_COLUMNS)
        with open(OUT / f"{rid}_latency.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in items)
        print(f"latency {rid}: {lat['lat_mean_s']:.3f}s/risposta, {lat['ms_per_token_mean']:.1f} ms/token", flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()


def quick():
    """Pipeline end-to-end: modello piccolo, A, 200 esempi; test su 128 problemi, latenza su 20 (tempi per la stima)."""
    rows = train5.load_split("A")[0][:200]
    t = time.time()
    test = _test_rows()[:128]
    import src.spec5 as s
    s.LATENCY_N, eval5.LATENCY_N = 20, 20
    run_one("small", "A", 2e-4, 0, "quick", "test128", test, train_rows=rows, do_latency=True, tag="_quick200")
    print(f"quick: {time.time() - t:.0f}s", flush=True)


def best_lr():
    d = pd.read_csv(RESULTS / "runs.csv")
    d = d[(d.stage == "sweep") & (d.split == "dev")]
    return float(d.sort_values("accuracy", ascending=False).iloc[0]["lr"])


def sweep():
    dev = train5.load_split("A")[1]
    for lr in LR_SWEEP:
        run_one("small", "A", lr, 0, "sweep", "dev", dev)
    print("lr migliore:", best_lr(), flush=True)


def aggiorna_report(blocco):
    md = analysis5.write_all(RESULTS)
    if REPORT.exists() and FINALE in REPORT.read_text(encoding="utf-8"):
        return
    REPORT.write_text(f"# Esperimento v5: ragionamento compresso su GSM8K (LoRA)\n\n**Stato: in corso, ultimo "
                      f"blocco completato: {blocco}.** Tabelle generate da `results/v5/`.\n\n{md}\n", encoding="utf-8")


def main_plan(models, formats, seeds):
    lr = best_lr()
    test = _test_rows()
    for mk in models:
        for fmt in formats:
            for sd in seeds:
                if mk == "large" and sd not in seeds[:2] and "--all-seeds-large" not in sys.argv:
                    continue
                run_one(mk, fmt, lr, sd, "main", "test", test, do_latency=(sd == seeds[0]))
            aggiorna_report(f"{mk} x {fmt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "quick", "sweep", "main", "report"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--all-seeds-large", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.stage == "build":
        build()
    elif a.stage == "quick":
        quick()
    elif a.stage == "sweep":
        sweep()
    elif a.stage == "main":
        main_plan(a.models, a.formats, a.seeds)
    else:
        aggiorna_report("report")
