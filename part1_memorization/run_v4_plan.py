"""Piano v4 a blocchi riprendibili: allineamento del vocabolario (0bis), forme, poi i 4 formati del 2x2.

Dopo ogni blocco: CSV in results/v4, summary_v4.md e REPORT_v4.md rigenerati (finche' il report non e' finale).
Uso: python run_v4_plan.py [--from BLOCCO] [--only]
"""
import argparse
import functools
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from src.config import ATTRS
from src.v2 import corpus, runner, train
from src.v2.model2 import GPT2
from src.v2.spec import JOB_KEYS, SIZES_V2, V2_RESULTS, Job
from src.v3 import train3, vocab
from src.v3.fastgen import evaluate_fast
from src.v3.spec3 import V3_RESULTS, V3_RUNS
from src.v4 import corpus4, eval4, forms4, summary4
from src.v4.spec4 import (F_COLUMNS, F_FORMATS, F_ORDER, F_SEEDS, F_SIZES, LR_CHECK_FORMAT, LR_CHECK_MULTS,
                          LR_CHECK_SIZE, V4_RESULTS, V4_RUNS, VOCAB, VOCAB_PAD)

PERSONS = None
BEST = runner.best_lr(pd.read_csv(V2_RESULTS / "A_lr.csv"))
BASE = {"italiano_pieno": "italiano_pieno", "italiano_canonico": "italiano_pieno",
        "telegrafico_lista": "telegrafico", "telegrafico_canonico": "telegrafico"}
REPORT = Path("REPORT_v4.md")
FINALE = "<!-- FINALE -->"


def v2_lr(fmt, size):
    return runner._lookup_lr(BEST, BASE[fmt], size)


def key(fmt):
    return f"{fmt}@{VOCAB}"


def _riga_v4(row, fmt, n_embd):
    """Riga F_COLUMNS: aggiunge formato e conteggi con padding a 64."""
    v = int(row["vocab"])
    vp = -(-v // VOCAB_PAD) * VOCAB_PAD
    return {**row, "format": fmt, "vocab_padded": vp, "params_total_padded": int(row["params_total"]) + (vp - v) * n_embd}


def _append(csv, row):
    pd.DataFrame([row], columns=list(F_COLUMNS)).to_csv(csv, mode="a", header=not Path(csv).exists(), index=False)


def run_jobs(jobs, fmt, csv, reuse=()):
    """Esegue i job mancanti in csv (chiave JOB_KEYS senza stage); riusa righe con stessa chiave da reuse."""
    done = {runner._key(r)[1:] for r in runner._read_rows(csv)}
    donors = {runner._key(r)[1:]: r for p in reuse for r in runner._read_rows(p)}
    valuta = eval4.make_evaluate(fmt)
    for k, job in enumerate(jobs, 1):
        kk = runner._key(vars(job))[1:]
        if kk in done:
            continue
        if kk in donors:
            row = {**donors[kk], "stage": job.stage}
        else:
            row = _riga_v4(train3.run_job(job, PERSONS, evaluate=valuta, runs_dir=V4_RUNS, vocab_pad=VOCAB_PAD),
                           fmt, SIZES_V2[job.size].n_embd)
        _append(csv, row)
        done.add(kk)
        print(f"[{k}/{len(jobs)}] {job.run_id} acc_mean={row['acc_mean']:.3f} train {row['train_time_s']:.0f}s", flush=True)


def _carica(path, size):
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    z = SIZES_V2[size]
    m = GPT2(state["tok_emb.weight"].shape[0], state["pos_emb.weight"].shape[0], z.n_layer, z.n_embd, z.n_head)
    m.load_state_dict(state)
    return m.cuda().eval()


def reuse_pieno(csv):
    """Run pieno@bpe2048 di v3 (D_main, V=2048 gia' allineato): rivalutati con la domanda v4, stage D_main_reeval."""
    done = {runner._key(r)[1:] for r in runner._read_rows(csv)}
    d = pd.read_csv(V3_RESULTS / "D_main.csv")
    d = d[d.corpus == key("italiano_pieno")]
    spec = train.CORPORA[key("italiano_pieno")]
    tok = train.get_tokenizer(spec, PERSONS)
    valuta = eval4.make_evaluate("italiano_pieno")
    for r in d.to_dict("records"):
        if runner._key(r)[1:] in done:
            continue
        job = Job(*[r[k] for k in JOB_KEYS])
        model = _carica(V3_RUNS / job.run_id / "ckpt.pt", r["size"])
        m = valuta(model, tok, spec, PERSONS, int(r["eval_persons"]))
        row = _riga_v4({**r, **m, "stage": "D_main_reeval"}, "italiano_pieno", SIZES_V2[r["size"]].n_embd)
        _append(csv, row)
        print(f"riuso {job.run_id}: acc v3 {r['acc_mean']:.3f} -> v4 {m['acc_mean']:.3f}", flush=True)


def f_block(fmt):
    main_csv, lr_csv = V4_RESULTS / "F_main.csv", V4_RESULTS / "F_lr.csv"
    scale = 1.0
    if fmt == "italiano_pieno":
        reuse_pieno(main_csv)
    if fmt == LR_CHECK_FORMAT:
        run_jobs([Job("F_lr", key(fmt), "passi", 200, LR_CHECK_SIZE, m, 0) for m in LR_CHECK_MULTS], fmt, lr_csv)
        best = runner.best_lr(pd.read_csv(lr_csv))[(key(fmt), LR_CHECK_SIZE)]
        scale = best / v2_lr(fmt, LR_CHECK_SIZE)
        print(f"verifica lr {fmt}: migliore {best}x a {LR_CHECK_SIZE} -> scala {scale}", flush=True)
    jobs = [Job("F_main", key(fmt), "passi", 200, s, v2_lr(fmt, s) * scale, sd) for sd in F_SEEDS for s in F_SIZES]
    run_jobs(jobs, fmt, main_csv, reuse=(lr_csv,) if fmt == LR_CHECK_FORMAT else ())


def align():
    """Fase 0bis: micro-benchmark e due run di controllo allineati (0,38M pieno seed 0)."""
    if not (V4_RESULTS / "align_bench.csv").exists():
        subprocess.run([sys.executable, "scripts/v4_align_bench.py"], check=True,
                       env={**__import__("os").environ, "PYTHONPATH": "."})
    out = V4_RESULTS / "align_control.csv"
    if out.exists():
        return
    a = pd.read_csv(V2_RESULTS / "A_main.csv")
    d = pd.read_csv(V3_RESULTS / "D_main.csv")
    righe = []
    for corp, ref, fonte in (("italiano_pieno@bpe2048", d, "v3 D_main"), ("italiano_pieno", a, "v2 A_main")):
        r = ref[(ref.corpus == corp) & (ref["size"] == "0.38M") & (ref.seed == 0)].iloc[0]
        spec = train.CORPORA[corp]
        k = vocab.max_answer_tokens(spec, train.get_tokenizer(spec, PERSONS))
        job = Job("F_ctrl", corp, "passi", 200, "0.38M", float(r["lr_mult"]), 0)
        row = train3.run_job(job, PERSONS, evaluate=functools.partial(evaluate_fast, max_new_tokens=max(k, 24)),
                             runs_dir=V4_RUNS, vocab_pad=VOCAB_PAD)
        v = int(row["vocab"])
        righe.append({"corpus": corp, "riferimento": fonte, "V": v, "V_padded": -(-v // VOCAB_PAD) * VOCAB_PAD,
                      "acc_ref": r["acc_mean"], "acc_aligned": row["acc_mean"], "delta": row["acc_mean"] - r["acc_mean"],
                      "train_time_ref_s": r["train_time_s"], "train_time_aligned_s": row["train_time_s"],
                      "params_total": row["params_total"]})
        print(righe[-1], flush=True)
    pd.DataFrame(righe).to_csv(out, index=False)


def forms():
    """F_forms.csv, controllo delle domande (si ferma se una non e' univoca) ed esempi."""
    stats, esempi = [], ["# Esempi v4: la stessa biografia, 2 rese per formato", ""]
    p = PERSONS[3]
    for fmt in F_FORMATS:
        conflitti = forms4.prompt_conflicts(fmt, PERSONS[:3000])
        if conflitti:
            raise SystemExit(f"domande non univoche in {fmt}: {conflitti[:5]}")
        stats.append(forms4.form_stats(fmt, PERSONS))
        esempi += [f"## {fmt}", ""] + [f"{i + 1}. {t}" for i, t in enumerate(corpus4.examples(fmt, p))]
        esempi += ["", f"Domanda (universita): `{corpus4.prompt(fmt, p, 'universita')}` -> "
                       f"`{corpus4.expected(fmt, p, 'universita')}`", ""]
    pd.concat(stats).to_csv(V4_RESULTS / "F_forms.csv", index=False)
    (V4_RESULTS / "F_examples.md").write_text("\n".join(esempi), encoding="utf-8")


def aggiorna_report(blocco):
    testo = summary4.write_all(V4_RESULTS)
    if REPORT.exists() and FINALE in REPORT.read_text(encoding="utf-8"):
        return
    REPORT.write_text(f"# Esperimento v4: regolarità vs lunghezza (disegno 2×2)\n\n"
                      f"**Stato: in corso, ultimo blocco completato: {blocco}.** "
                      f"Tabelle generate automaticamente da `results/v4/`.\n\n{testo}\n", encoding="utf-8")


BLOCKS = {"align": align, "forms": forms, **{f"F_{f}": functools.partial(f_block, f) for f in F_ORDER}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="align", choices=list(BLOCKS))
    ap.add_argument("--only", action="store_true")
    a = ap.parse_args()
    PERSONS = corpus.load_persons(20000)
    corpus4.register_all()
    V4_RESULTS.mkdir(parents=True, exist_ok=True)
    names = list(BLOCKS)
    for name in names[names.index(a.start):][: 1 if a.only else None]:
        print(f"=== BLOCCO {name} INIZIO", flush=True)
        BLOCKS[name]()
        aggiorna_report(name)
        print(f"=== BLOCCO {name} COMPLETATO", flush=True)
