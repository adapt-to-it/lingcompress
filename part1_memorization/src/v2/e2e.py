"""Fase C v2: risposta end-to-end (stripper, core compresso, traduttore) confrontata con il core pieno."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

from src.config import ATTRS, EOS
from src.data import Person
from src.v2 import corpus, infer, kvgen, translate
from src.v2.model2 import GPT2
from src.v2.spec import C_BATCHES, C_QUESTIONS, C_SIZES, E2E_COLUMNS, N_PERSONS, V2_RESULTS

E2E_FORMATS = ("telegrafico", "interlingua")


def _domande(persons: list[Person], fmt: str) -> list[tuple[str, str, str, str]]:
    """(P, C, attr, truth) delle prime C_QUESTIONS domande, come in translate.run_translate."""
    n_persone = math.ceil(C_QUESTIONS / len(ATTRS))
    return translate.pairs(persons, range(n_persone), fmt, "strip")[:C_QUESTIONS]


def _core(model: GPT2, tok: Tokenizer, C_hat: list[str], ks: list[int], batch: int) -> tuple[list[str], dict]:
    """Risposta del core compresso dopo [EOS] C_hat; prime ks[i] token generati per ogni domanda.

    Gruppi di stessa lunghezza di prompt, a blocchi di batch. Il prompt viene ritagliato a sinistra
    se non lascia spazio per max(ks) token nel blocco (ctx_len del modello).
    """
    device = next(model.parameters()).device
    n_total = model.count_params()[1]
    n_layer = len(model.blocks)
    n_embd = model.tok_emb.embedding_dim
    eos = tok.token_to_id(EOS)
    ctx = model.ctx_len

    ids = [[eos, *tok.encode(c).ids] for c in C_hat]
    gruppi: dict[int, list[int]] = {}
    for i, riga in enumerate(ids):
        gruppi.setdefault(len(riga), []).append(i)
    unita = [g[s:s + batch] for g in gruppi.values() for s in range(0, len(g), batch)]

    outs = [""] * len(C_hat)
    latenze, flops = [], []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for unit in unita:
            K = max(ks[i] for i in unit)
            limite = ctx - K + 1
            if limite < 1:
                raise ValueError(f"{K} token di risposta non stanno in ctx_len={ctx}")
            righe = [ids[i][-limite:] for i in unit]
            idx = torch.tensor(righe, dtype=torch.long, device=device)
            nuovi, st = kvgen.generate(model, idx, max_new_tokens=K)
            latenze.append((st["prefill_s"] + st["decode_s"]) * 1000.0)
            n_nuovi = nuovi.shape[1]
            for r, i in enumerate(unit):
                outs[i] = tok.decode(nuovi[r, :ks[i]].tolist())
                pre, dec = kvgen.flops(n_total, n_layer, n_embd, len(righe[r]), n_nuovi)
                flops.append(pre + dec)

    stats = {
        "latency_ms_mean": float(np.mean(latenze)),
        "flops_per_request": float(np.mean(flops)),
    }
    return outs, stats


def _riga(fmt: str, size: str, batch: int, domande, base: dict,
          st_s: dict, st_c: dict, st_t: dict, outs: list[str]) -> dict:
    valori = [out.startswith(P) and corpus.parse_answer(out[len(P):], attr, "pieno", "it") == truth
              for out, (P, _, attr, truth) in zip(outs, domande)]
    stripper_ms = st_s["latency_ms_mean"]
    core_ms = st_c["latency_ms_mean"]
    translator_ms = st_t["latency_ms_mean"]
    pipeline_ms = stripper_ms + core_ms + translator_ms
    return {
        "format": fmt, "size": size, "batch": batch, "n_questions": len(domande),
        "stripper_ms": stripper_ms, "core_ms": core_ms, "translator_ms": translator_ms,
        "pipeline_ms": pipeline_ms, "pieno_core_ms": base["latency_ms_mean"],
        "speedup_vs_pieno": base["latency_ms_mean"] / pipeline_ms,
        "pipeline_flops": st_s["flops_per_request"] + st_c["flops_per_request"] + st_t["flops_per_request"],
        "pieno_flops": base["flops_per_answer"],
        "pipeline_acc": float(np.mean(valori)), "pieno_acc": base["acc"],
    }


def _append(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row], columns=list(E2E_COLUMNS)).to_csv(
        csv_path, mode="a", header=write_header, index=False)


def run_e2e(out_csv: Path) -> None:
    """Misura E2E_FORMATS x C_SIZES x C_BATCHES; salta le combinazioni gia' in out_csv; scrive subito."""
    out_csv = Path(out_csv)
    fatti = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        fatti = {(str(r["format"]), str(r["size"]), int(r["batch"]))
                 for r in pd.read_csv(out_csv).to_dict("records")}

    persons: list[Person] | None = None
    for fmt in E2E_FORMATS:
        for size in C_SIZES:
            in_sospeso = [b for b in C_BATCHES if (fmt, size, b) not in fatti]
            if not in_sospeso:
                continue
            if persons is None:
                persons = corpus.load_persons(N_PERSONS)

            stripper, tok_s, _ = translate.train(fmt, "strip", persons)
            core, tok_c, _ = infer.load(infer.core_job(fmt, size), persons)
            translator, tok_t, _ = translate.train(fmt, "trans", persons)

            domande = _domande(persons, fmt)
            P_list = [P for P, _, _, _ in domande]
            ks = [infer.answer_tokens(tok_c, C, corpus.surface(attr, truth, fmt, "it"))
                  for _, C, attr, truth in domande]

            for batch in in_sospeso:
                C_hat, st_s = translate.run(stripper, tok_s, P_list, batch)
                risposte, st_c = _core(core, tok_c, C_hat, ks, batch)
                srcs = [c + " " + r.strip() for c, r in zip(C_hat, risposte)]
                outs, st_t = translate.run(translator, tok_t, srcs, batch)
                base = infer.measure_c(infer.core_job("italiano_pieno", size), persons, batch)
                riga = _riga(fmt, size, batch, domande, base, st_s, st_c, st_t, outs)
                _append(out_csv, riga)
                fatti.add((fmt, size, batch))
                print(f"e2e {fmt} {size} batch {batch}: pipeline {riga['pipeline_ms']:.2f} ms "
                      f"pieno {riga['pieno_core_ms']:.2f} ms acc {riga['pipeline_acc']:.3f}")


if __name__ == "__main__":
    run_e2e(V2_RESULTS / "e2e_c.csv")
