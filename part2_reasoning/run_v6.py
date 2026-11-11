"""Piano v6 a stadi riprendibili.
reuse  : condizioni 1 e 3 copiate da v5 (stessi output e latenze)
audit  : verifica della calcolatrice di A su 200 soluzioni di training
H      : condizioni 2, 4, 5 sui pesi v5
build6 : dati SFT C6 (validi tra i problemi di training v5), statistiche ed esempi A / C5 / C6
train6 : adapter C6 (stessi iperparametri di v5)
G      : condizioni 6, 7, 8
report : analisi e REPORT_v6.md (finche' non e' finale)
Uso: python run_v6.py STADIO [--models small large]
"""
import argparse
import gc
import json
import statistics
import sys

import pandas as pd
import torch

from src import analysis6, constrained, eval5, eval6, grammar6, train5
from src.spec5 import MODELS, RESULTS, RUNS, SFT, DEV_SIZE, FORMAT_NAMES
from src.spec6 import (CONDITIONS, COND_LABELS, LATENCY6_COLUMNS, LR6, RESULTS6, RUN6_COLUMNS, SEEDS6)

OUT6 = RESULTS6 / "outputs"
REPORT = RESULTS.parent.parent / "REPORT_v6.md"
FINALE = "<!-- FINALE -->"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(csv, row, columns):
    pd.DataFrame([row], columns=list(columns)).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def _done(run_id):
    csv = RESULTS6 / "runs.csv"
    return csv.exists() and (pd.read_csv(csv).run_id == run_id).any()


def _adapter(model, cond, seed):
    fmt = CONDITIONS[cond]["fmt"]
    name = {"A": "A", "C5": "C", "C6": "C6"}[fmt]
    return RUNS / f"{model}_{name}_lr{LR6:g}_s{seed}" / "adapter"


def _test_rows():
    return [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in _jsonl(SFT / "test.jsonl")]


def reuse():
    """Condizioni 1 (A) e 3 (C5) = run principali di v5: output con n_forced = 0, stesse metriche e latenze."""
    OUT6.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS / "runs.csv")
    runs = runs[(runs.stage == "main") & (runs.split == "test") & runs.format.isin(["A", "C"])]
    lat = pd.read_csv(RESULTS / "latency.csv")
    for r in runs.to_dict("records"):
        cond = 1 if r["format"] == "A" else 3
        rid = f"{r['model']}_c{cond}_s{r['seed']}"
        if _done(rid):
            continue
        with open(OUT6 / f"{rid}_test.jsonl", "w", encoding="utf-8") as f:
            for o in _jsonl(RESULTS / "outputs" / f"{r['run_id']}_test.jsonl"):
                f.write(json.dumps({**o, "n_forced": 0}, ensure_ascii=False) + "\n")
        c = CONDITIONS[cond]
        _append(RESULTS6 / "runs.csv", {**{k: r.get(k) for k in RUN6_COLUMNS}, "run_id": rid, "cond": cond,
                                        "format": c["fmt"], "grammar": c["grammar"], "calc": c["calc"],
                                        "forced_mean": 0.0, "note_rate": 0.0, "solve_rate": 0.0}, RUN6_COLUMNS)
        lr = lat[lat.run_id == r["run_id"]]
        if len(lr):
            _append(RESULTS6 / "latency.csv", {**lr.iloc[0].to_dict(), "run_id": rid, "cond": cond,
                                               "format": c["fmt"]}, LATENCY6_COLUMNS)
    print("reuse ok", flush=True)


def audit():
    """Calcolatrice di A: dove scatta su 200 soluzioni A di training e se il valore coincide con quello scritto."""
    rows = train5.load_split("A")[0][:200]
    res = constrained.audit_a_trigger([r["solution"] for r in rows])
    (RESULTS6 / "a_calc_audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in res.items() if not isinstance(v, list)}, flush=True)


def run_cond(model, cond, seed, test, latency=False):
    rid = f"{model}_c{cond}_s{seed}"
    if _done(rid):
        return
    c = CONDITIONS[cond]
    m, tok = eval5.load_model(model, _adapter(model, cond, seed))
    factory = constrained.make_processor(tok, c["fmt"], c["grammar"], c["calc"]) if (c["grammar"] or c["calc"]) else None
    met = eval6.evaluate6(m, tok, c["fmt"], test, processor_factory=factory, out_path=OUT6 / f"{rid}_test.jsonl")
    _append(RESULTS6 / "runs.csv", {"run_id": rid, "model": model, "cond": cond, "format": c["fmt"],
                                    "grammar": c["grammar"], "calc": c["calc"], "seed": seed, "split": "test", **met},
            RUN6_COLUMNS)
    print(f"eval {rid} ({COND_LABELS[cond]}): acc {met['accuracy']:.3f} conf {met['conform_rate']:.3f} "
          f"tok {met['tok_mean']:.1f} forced {met['forced_mean']:.1f} {met['throughput_time_s']:.0f}s", flush=True)
    if latency:
        lat, items = eval6.latency6(m, tok, c["fmt"], test, processor_factory=factory)
        _append(RESULTS6 / "latency.csv", {"run_id": rid, "model": model, "cond": cond, "format": c["fmt"], **lat},
                LATENCY6_COLUMNS)
        with open(OUT6 / f"{rid}_latency.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in items)
        print(f"latency {rid}: {lat['lat_mean_s']:.3f}s/risposta, {lat['ms_per_token_mean']:.1f} ms/token", flush=True)
    del m
    gc.collect(); torch.cuda.empty_cache()


def conds(models, which):
    test = _test_rows()
    for model in models:
        for cond in which:
            for seed in SEEDS6[model]:
                run_cond(model, cond, seed, test, latency=(seed == 0))
            report(f"{model} x condizione {cond}")


def build6():
    """C6 validi tra i problemi di training v5 (stesso dev escluso); statistiche ed esempi."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["small"])
    last = {}
    for r in _jsonl(RESULTS.parent.parent / "v5" / "data" / "rewrites" / "C6.jsonl"):
        last[r["id"]] = r
    a_all = _jsonl(SFT / "A_train.jsonl")
    c5_all = {r["id"]: r for r in _jsonl(SFT / "C_train.jsonl")}
    train_ids = [r["id"] for r in a_all[:-DEV_SIZE]]
    a_by = {r["id"]: r for r in a_all}
    valid = [i for i in train_ids if last.get(i, {}).get("ok")]
    rows = [{**a_by[i], "solution": last[i]["text"]} for i in valid]
    with open(SFT / "C6_train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    ntok = lambda s: len(tok(s).input_ids)  # noqa: E731
    st = [grammar6.stats_c6(r["solution"]) for r in rows]
    units = pd.Series([u for s in st for u in s["units"]]).value_counts()
    stats = {"train_ids": len(train_ids), "generated": sum(i in last for i in train_ids), "valid": len(valid),
             "reasons": pd.Series([last[i]["reason"] for i in train_ids if i in last and not last[i]["ok"]])
             .value_counts().to_dict(),
             "tok_C6": statistics.mean(ntok(r["solution"]) for r in rows),
             "tok_C5_same": statistics.mean(ntok(c5_all[r["id"]]["solution"]) for r in rows),
             "tok_A_same": statistics.mean(ntok(a_by[r["id"]]["solution"]) for r in rows),
             "note_rate": sum(s["n_note"] > 0 for s in st) / len(st), "notes_per_sol": statistics.mean(s["n_note"] for s in st),
             "solve_rate": sum(s["n_solve"] > 0 for s in st) / len(st), "units": units.to_dict()}
    (RESULTS6 / "data_c6.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    ex = ["# Esempi v6: stesso problema in A / C5 / C6", ""]
    for i in [rows[k]["id"] for k in (5, 400, 3000)]:
        ex += [f"## {i}", "", f"**Domanda.** {a_by[i]['question']}", ""]
        for name, sol in (("A", a_by[i]["solution"]), ("C5", c5_all[i]["solution"]), ("C6", last[i]["text"])):
            ex += [f"**{name}**", "", "```", sol, "```", ""]
    (RESULTS6 / "examples_A_C5_C6.md").write_text("\n".join(ex), encoding="utf-8")
    print({k: v for k, v in stats.items() if k != "units"}, flush=True)


def train6(models):
    rows = _jsonl(SFT / "C6_train.jsonl")
    for model in models:
        for seed in SEEDS6[model]:
            out = RUNS / f"{model}_C6_lr{LR6:g}_s{seed}" / "adapter"
            if (out.parent / "train.json").exists():
                continue
            info = train5.train(model, "C6", LR6, seed, rows=rows, out_dir=out)
            print(f"train {model} C6 s{seed}: {info['steps']} step, {info['train_time_s']:.0f}s, "
                  f"loss {info['final_loss']:.3f}", flush=True)
            return True  # un adapter per processo: nello stesso processo la memoria non si liberava (spill in RAM)
    return False


def xl_train():
    """Fase opzionale: Qwen3-4B-Base QLoRA, adapter A e C6 (seed 0), uno per processo."""
    for fmt, rows in (("A", train5.load_split("A")[0]), ("C6", _jsonl(SFT / "C6_train.jsonl"))):
        out = RUNS / f"xl_{fmt}_lr{LR6:g}_s0" / "adapter"
        if not (out.parent / "train.json").exists():
            info = train5.train("xl", fmt, LR6, 0, rows=rows, out_dir=out)
            print(f"train xl {fmt} s0: {info['steps']} step, {info['train_time_s']:.0f}s, "
                  f"loss {info['final_loss']:.3f}", flush=True)
            return True
    return False


def report(blocco):
    md = analysis6.write_all6(RESULTS6)
    if REPORT.exists() and FINALE in REPORT.read_text(encoding="utf-8"):
        return
    REPORT.write_text(f"# Esperimento v6: canonico migliorato, decodifica vincolata e calcolatrice\n\n**Stato: in corso, "
                      f"ultimo blocco completato: {blocco}.** Tabelle generate da `results/v6/`.\n\n{md}\n",
                      encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["reuse", "audit", "H", "build6", "train6", "G", "report", "xl_train", "xl_eval"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    RESULTS6.mkdir(parents=True, exist_ok=True)
    {"reuse": reuse, "audit": audit, "build6": build6, "report": lambda: report("report"),
     "H": lambda: conds(a.models, (2, 4, 5)), "train6": lambda: sys.exit(0 if train6(a.models) else 3),
     "G": lambda: conds(a.models, (6, 7, 8)),
     "xl_train": lambda: sys.exit(0 if xl_train() else 3), "xl_eval": lambda: conds(["xl"], (2, 8))}[a.stage]()
