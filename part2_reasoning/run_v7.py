"""Piano v7 a stadi riprendibili: prosa concisa A1 / A2.
reuse    : condizioni di v6 riprese (small/large 1, 2, 3, 5; xl 2 e 8): run, output e latenze copiati da results/v6
build7   : dati SFT A1 / A2 (validi tra i problemi di training v5), statistiche ed esempi A / A1 / A2 / C5
audit7   : verifica della calcolatrice di A su 200 soluzioni A1 / A2 di training
train7   : adapter A1 e A2 (stessi iperparametri di v5)
J        : condizioni 9-12 (small, large), latenze al seed 0
xl_train : adapter xl C (se manca) e xl A1 / A2 (seed 0), uno per processo
xl_eval  : condizioni 5 e 10 (A1) o 12 (A2) su xl
report   : analisi e REPORT_v7.md (finche' non e' finale)
Uso: python run_v7.py STADIO [--models small large] [--fmt A1]
"""
import argparse
import gc
import json
import re
import shutil
import statistics
import sys
from collections import Counter

import pandas as pd
import torch

from src import constrained, eval5, eval6, train5
from src.concise7 import load_attempts, select7
from src.spec5 import DEV_SIZE, MODELS, RUNS, SFT
from src.spec6 import CONDITIONS, COND_LABELS, LATENCY6_COLUMNS, LR6, RESULTS6, RUN6_COLUMNS, SEEDS6
from src.spec7 import CALC_AUDIT_MIN, FORMATS7, NEW_CONDS, RELATION_WORDS, REUSED_V6, REWRITE7, RESULTS7, SELECT7

OUT7 = RESULTS7 / "outputs"
REPORT = RESULTS7.parent.parent / "REPORT_v7.md"
FINALE = "<!-- FINALE -->"
WORD_RE = re.compile(r"[A-Za-z']+")


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(csv, row, columns):
    pd.DataFrame([row], columns=list(columns)).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def _done(run_id):
    csv = RESULTS7 / "runs.csv"
    return csv.exists() and (pd.read_csv(csv).run_id == run_id).any()


def _adapter(model, cond, seed):
    fmt = CONDITIONS[cond]["fmt"]
    name = {"A": "A", "C5": "C", "C6": "C6"}.get(fmt, fmt)     # A1 / A2 con il proprio nome
    return RUNS / f"{model}_{name}_lr{LR6:g}_s{seed}" / "adapter"


def _mode(fmt):
    """Modo di decodifica e di calcolatrice: A1 e A2 usano quelli di A (stesso formato di risposta)."""
    return "A" if fmt in ("A", "A1", "A2") else fmt


def _test_rows():
    return [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in _jsonl(SFT / "test.jsonl")]


def build_rows(a_rows, chosen, select):
    """Righe SFT: A di training con la soluzione scelta. Con select == "strict" restano solo le soluzioni strette."""
    rows = []
    for r in a_rows:
        if r["id"] not in chosen:
            continue
        text, strict = chosen[r["id"]]
        if select == "strict" and not strict:
            continue
        rows.append({**r, "solution": text})
    return rows


def reuse():
    """Condizioni 1, 2, 3, 5 (small, large) e 2, 8 (xl) di v6: tutti i seed, output e latenze se presenti."""
    OUT7.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS6 / "runs.csv")
    lat = pd.read_csv(RESULTS6 / "latency.csv")
    for model, which in REUSED_V6.items():
        for r in runs[(runs.model == model) & runs.cond.isin(which)].to_dict("records"):
            rid = r["run_id"]
            if _done(rid):
                continue
            for name in (f"{rid}_test.jsonl", f"{rid}_latency.jsonl"):
                if (RESULTS6 / "outputs" / name).exists():
                    shutil.copyfile(RESULTS6 / "outputs" / name, OUT7 / name)
            _append(RESULTS7 / "runs.csv", {k: r.get(k) for k in RUN6_COLUMNS}, RUN6_COLUMNS)
            lr = lat[lat.run_id == rid]
            if len(lr):
                _append(RESULTS7 / "latency.csv", {**lr.iloc[0].to_dict()}, LATENCY6_COLUMNS)
    print("reuse ok", flush=True)


def _words_per_line(lines):
    """Parole per riga (solo [A-Za-z']+), escluso l'ultima riga; media su tutte le righe."""
    body = lines[:-1]
    counts = [len(WORD_RE.findall(line)) for line in body]
    return sum(counts), len(counts)


def _relation_lines(lines):
    """Righe che contengono almeno una parola di RELATION_WORDS."""
    hit = sum(bool(set(w.lower() for w in WORD_RE.findall(line)) & set(RELATION_WORDS)) for line in lines)
    return hit, len(lines)


def build7():
    """Scelta delle riscritture per fmt, SFT, statistiche (ids comuni A1 / A2 / C_train) ed esempi."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["small"])
    ntok = lambda s: len(tok(s).input_ids)  # noqa: E731
    a_all = _jsonl(SFT / "A_train.jsonl")
    a_by = {r["id"]: r for r in a_all}
    a_train = a_all[:-DEV_SIZE]
    train_ids = [r["id"] for r in a_train]
    c5_all = {r["id"]: r for r in _jsonl(SFT / "C_train.jsonl")}
    attempts, chosen, kept, data = {}, {}, {}, {}
    for fmt in FORMATS7:
        attempts[fmt] = load_attempts(REWRITE7[fmt])
        chosen[fmt] = select7(attempts[fmt])
        kept[fmt] = build_rows(a_train, chosen[fmt], SELECT7)
        with open(SFT / f"{fmt}_train.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept[fmt])
        strict_ids = {i for i, (_, s) in chosen[fmt].items() if s}
        non_strict = [i for i in train_ids if i in attempts[fmt] and i not in strict_ids]
        data[fmt] = {"n_train_ids": len(train_ids), "generated": sum(i in attempts[fmt] for i in train_ids),
                     "strict": sum(i in strict_ids for i in train_ids),
                     "soft_only": sum(i in chosen[fmt] and i not in strict_ids for i in train_ids),
                     "kept": len(kept[fmt]),
                     "non_strict_last_reason": dict(Counter(attempts[fmt][i][-1]["reason"] for i in non_strict))}
        kept_ids = {r["id"] for r in kept[fmt]}
        rows = []
        for i in train_ids:
            if i in kept_ids:
                continue
            last = attempts[fmt].get(i, [None])[-1]
            reason = (last["reason"] or "ok") if last else "no_attempt"
            ratio = last["ratio"] if last else None
            rows.append({"id": i, "reason": reason, "ratio": ratio, "a_tokens": ntok(a_by[i]["solution"])})
        pd.DataFrame(rows, columns=["id", "reason", "ratio", "a_tokens"]).to_csv(
            RESULTS7 / f"dropped_{fmt}.csv", index=False)
    by_fmt = {fmt: {r["id"]: r for r in kept[fmt]} for fmt in FORMATS7}
    common = [i for i in train_ids if i in c5_all and all(i in by_fmt[f] for f in FORMATS7)]
    texts = {"A": lambda i: a_by[i]["solution"], "A1": lambda i: by_fmt["A1"][i]["solution"],
             "A2": lambda i: by_fmt["A2"][i]["solution"], "C5": lambda i: c5_all[i]["solution"]}
    common_stats = {"n": len(common), "tok_mean": {}, "words_per_line": {}, "relation_share": {}}
    for name, get in texts.items():
        toks, words, words_n, rel, rel_n = [], 0, 0, 0, 0
        for i in common:
            lines = [ln for ln in get(i).split("\n") if ln.strip()]
            toks.append(ntok(get(i)))
            w, wn = _words_per_line(lines)
            words, words_n = words + w, words_n + wn
            h, hn = _relation_lines(lines)
            rel, rel_n = rel + h, rel_n + hn
        common_stats["tok_mean"][name] = statistics.mean(toks)
        common_stats["words_per_line"][name] = words / words_n
        common_stats["relation_share"][name] = rel / rel_n
    data["common"] = common_stats
    (RESULTS7 / "data_a7.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    ex = ["# Esempi v7: stesso problema in A / A1 / A2 / C5", ""]
    for i in [common[k] for k in (5, 400, 3000)]:
        ex += [f"## {i}", "", f"**Domanda.** {a_by[i]['question']}", ""]
        for name in ("A", "A1", "A2", "C5"):
            ex += [f"**{name}**", "", "```", texts[name](i), "```", ""]
    (RESULTS7 / "examples_A_A1_A2_C5.md").write_text("\n".join(ex), encoding="utf-8")
    print({k: v for k, v in data.items() if k != "common"}, "common", common_stats["n"], flush=True)


def audit7():
    """Calcolatrice di A su 200 soluzioni A1 / A2 di training: dove scatta e quanto spesso."""
    for fmt in FORMATS7:
        rows = _jsonl(SFT / f"{fmt}_train.jsonl")[:200]
        res = constrained.audit_a_trigger([r["solution"] for r in rows])
        res["fired_rate"] = res["fired"] / res["equals"] if res["equals"] else 0.0
        (RESULTS7 / f"a_calc_audit_{fmt}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                           encoding="utf-8")
        shown = {k: v for k, v in res.items() if not isinstance(v, list)}
        print(fmt, {**shown, "pass": res["fired_rate"] >= CALC_AUDIT_MIN}, flush=True)


def run_cond(model, cond, seed, test, latency=False):
    rid = f"{model}_c{cond}_s{seed}"
    if _done(rid):
        return
    c = CONDITIONS[cond]
    mode = _mode(c["fmt"])
    m, tok = eval5.load_model(model, _adapter(model, cond, seed))
    factory = constrained.make_processor(tok, mode, c["grammar"], c["calc"]) if (c["grammar"] or c["calc"]) else None
    met = eval6.evaluate6(m, tok, mode, test, processor_factory=factory, out_path=OUT7 / f"{rid}_test.jsonl")
    _append(RESULTS7 / "runs.csv", {"run_id": rid, "model": model, "cond": cond, "format": c["fmt"],
                                    "grammar": c["grammar"], "calc": c["calc"], "seed": seed, "split": "test", **met},
            RUN6_COLUMNS)
    print(f"eval {rid} ({COND_LABELS[cond]}): acc {met['accuracy']:.3f} conf {met['conform_rate']:.3f} "
          f"tok {met['tok_mean']:.1f} forced {met['forced_mean']:.1f} {met['throughput_time_s']:.0f}s", flush=True)
    if latency:
        lat, items = eval6.latency6(m, tok, mode, test, processor_factory=factory)
        _append(RESULTS7 / "latency.csv", {"run_id": rid, "model": model, "cond": cond, "format": c["fmt"], **lat},
                LATENCY6_COLUMNS)
        with open(OUT7 / f"{rid}_latency.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in items)
        print(f"latency {rid}: {lat['lat_mean_s']:.3f}s/risposta, {lat['ms_per_token_mean']:.1f} ms/token", flush=True)
    del m
    gc.collect(); torch.cuda.empty_cache()


def conds(models, which, seeds=None):
    test = _test_rows()
    for model in models:
        for cond in which:
            for seed in (seeds if seeds is not None else SEEDS6[model]):
                run_cond(model, cond, seed, test, latency=(seed == 0))
            report(f"{model} x condizione {cond}")


def j7(models, which=None, seeds=None):
    """Condizioni nuove (9-12) per ogni modello (o solo which / seeds: un processo per adapter)."""
    for model in models:
        conds([model], which or NEW_CONDS[model], seeds)


def train7(models):
    for model in models:
        for fmt in FORMATS7:
            for seed in SEEDS6[model]:
                out = RUNS / f"{model}_{fmt}_lr{LR6:g}_s{seed}" / "adapter"
                if (out.parent / "train.json").exists():
                    continue
                rows = _jsonl(SFT / f"{fmt}_train.jsonl")
                info = train5.train(model, fmt, LR6, seed, rows=rows, out_dir=out)
                print(f"train {model} {fmt} s{seed}: {info['steps']} step, {info['train_time_s']:.0f}s, "
                      f"loss {info['final_loss']:.3f}", flush=True)
                return True  # un adapter per processo: nello stesso processo la memoria non si liberava (spill in RAM)
    return False


def xl_train(fmt):
    """Fase opzionale: Qwen3-4B-Base QLoRA, adapter xl C (se manca) e xl del formato richiesto (seed 0)."""
    for name, rows in (("C", train5.load_split("C")[0]), (fmt, _jsonl(SFT / f"{fmt}_train.jsonl"))):
        out = RUNS / f"xl_{name}_lr{LR6:g}_s0" / "adapter"
        if not (out.parent / "train.json").exists():
            info = train5.train("xl", name, LR6, 0, rows=rows, out_dir=out)
            print(f"train xl {name} s0: {info['steps']} step, {info['train_time_s']:.0f}s, "
                  f"loss {info['final_loss']:.3f}", flush=True)
            return True
    return False


def xl_eval(fmt):
    conds(["xl"], (5, 10 if fmt == "A1" else 12))


def report(blocco):
    from src.analysis7 import write_all7
    md = write_all7(RESULTS7)
    if REPORT.exists() and FINALE in REPORT.read_text(encoding="utf-8"):
        return
    notes = "".join(p.read_text(encoding="utf-8") + "\n\n" for p in sorted(RESULTS7.glob("notes_*.md")))
    REPORT.write_text(f"# Esperimento v7: prosa concisa\n\n**Stato: in corso, ultimo blocco completato: {blocco}.** "
                      f"Tabelle generate da `results/v7/`.\n\n{notes}{md}\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["reuse", "build7", "audit7", "train7", "J", "xl_train", "xl_eval", "report"])
    ap.add_argument("--models", nargs="+", default=list(NEW_CONDS))
    ap.add_argument("--fmt", choices=FORMATS7)
    ap.add_argument("--cond", nargs="+", type=int)
    ap.add_argument("--seed", nargs="+", type=int)
    a = ap.parse_args()
    if a.stage in ("xl_train", "xl_eval") and a.fmt is None:
        ap.error("--fmt richiesto per xl_train e xl_eval")
    RESULTS7.mkdir(parents=True, exist_ok=True)
    {"reuse": reuse, "build7": build7, "audit7": audit7, "report": lambda: report("report"),
     "train7": lambda: sys.exit(0 if train7(a.models) else 3),
     "J": lambda: j7(a.models, a.cond, a.seed),
     "xl_train": lambda: sys.exit(0 if xl_train(a.fmt) else 3),
     "xl_eval": lambda: xl_eval(a.fmt)}[a.stage]()
