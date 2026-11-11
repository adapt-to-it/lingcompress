"""Misura di inferenza della fase C (latenza, throughput, FLOPs, accuratezza) dei modelli core di A_main."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

from src.config import EOS
from src.data import Person
from src.v2 import corpus, kvgen
from src.v2.model2 import GPT2
from src.v2.spec import (
    C_BATCHES, C_FORMATS, C_QUESTIONS, C_SIZES, CORPORA, EVAL_PERSONS, INFER_COLUMNS, SIZES_V2,
    V2_RESULTS, V2_RUNS, CorpusSpec, EvalCase, Job,
)
from src.v2.train import get_tokenizer

WARMUP_CALLS = 20           # chiamate di riscaldamento (batch 1) non cronometrate
FALLBACK_STAGE = "A_lr"     # A_main riusa il checkpoint di A_lr


def answer_tokens(tok: Tokenizer, prompt: str, surf: str) -> int:
    """Token minimi per scrivere la risposta: differenza tra prompt+risposta e prompt."""
    completo = len(tok.encode(prompt + " " + surf).ids)
    return max(1, completo - len(tok.encode(prompt).ids))


def load(job: Job, persons: list[Person]) -> tuple[GPT2, Tokenizer, CorpusSpec]:
    """Modello in eval mode dal checkpoint del Job; se manca, usa quello di FALLBACK_STAGE."""
    percorso = V2_RUNS / job.run_id / "ckpt.pt"
    if not percorso.exists():
        job = dataclasses.replace(job, stage=FALLBACK_STAGE)
        percorso = V2_RUNS / job.run_id / "ckpt.pt"
    if not percorso.exists():
        raise FileNotFoundError(f"checkpoint mancante: {percorso}")

    spec = CORPORA[job.corpus]
    tok = get_tokenizer(spec, persons)
    stato = torch.load(percorso, map_location="cpu")["model"]
    size = SIZES_V2[job.size]
    # ctx_len non e' salvato nel checkpoint: e' il numero di righe di pos_emb
    ctx_len = stato["pos_emb.weight"].shape[0]
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=ctx_len, n_layer=size.n_layer,
                 n_embd=size.n_embd, n_head=size.n_head)
    model.load_state_dict(stato)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tok, spec


def core_job(corpus: str, size: str) -> Job:
    """Job della riga seed 0 di A_main per corpus e taglia: il modello core misurato in fase C."""
    for rec in pd.read_csv(V2_RESULTS / "A_main.csv").to_dict("records"):
        if str(rec["corpus"]) == corpus and str(rec["size"]) == size and int(rec["seed"]) == 0:
            return Job(stage=str(rec["stage"]), corpus=str(rec["corpus"]), protocol=str(rec["protocol"]),
                       exposures=int(rec["exposures"]), size=str(rec["size"]),
                       lr_mult=float(rec["lr_mult"]), seed=int(rec["seed"]),
                       n_persons=int(rec["n_persons"]))
    raise ValueError(f"nessuna riga seed 0 in A_main per {corpus} {size}")


def c_cases(spec: CorpusSpec, persons: list[Person]) -> list[EvalCase]:
    """Prime C_QUESTIONS domande "in" (persona, poi ATTRS) tra le EVAL_PERSONS persone valutate."""
    casi = [c for c in corpus.eval_cases(spec, persons, EVAL_PERSONS) if c.kind == "in"]
    return casi[:C_QUESTIONS]


def _gruppi(batch: int, ids: list[list[int]]) -> list[list[int]]:
    """Indici raggruppati per lunghezza del prompt (senza padding), a blocchi di batch."""
    per_lunghezza: dict[int, list[int]] = {}
    for i, riga in enumerate(ids):
        per_lunghezza.setdefault(len(riga), []).append(i)
    return [indici[s:s + batch] for indici in per_lunghezza.values()
            for s in range(0, len(indici), batch)]


def _genera(model: GPT2, device: torch.device, righe: list[list[int]], k: int) -> tuple[torch.Tensor, dict]:
    """Generazione greedy con cache KV in bf16 autocast; righe di uguale lunghezza."""
    idx = torch.tensor(righe, dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        return kvgen.generate(model, idx, max_new_tokens=k)


def measure_c(job: Job, persons: list[Person], batch: int) -> dict:
    """Riga con esattamente INFER_COLUMNS: tempi per richiesta (batch 1) o per batch, FLOPs, accuratezza."""
    model, tok, spec = load(job, persons)
    device = next(model.parameters()).device
    eos = tok.token_to_id(EOS)
    casi = c_cases(spec, persons)
    ids = [[eos] + tok.encode(c.prompt).ids for c in casi]
    ks = [answer_tokens(tok, c.prompt, corpus.surface(c.attr, c.truth, c.style, c.lang)) for c in casi]
    size = SIZES_V2[job.size]
    n_nonemb, n_total = model.count_params()

    if batch == 1:
        # riscaldamento non cronometrato
        for i in range(WARMUP_CALLS):
            j = i % len(casi)
            _genera(model, device, [ids[j]], ks[j])

    tempi = []      # statistiche di ogni richiesta (batch 1) o di ogni batch
    esiti = []      # (corretto, flops_prefill, flops_decode) per caso
    for indici in _gruppi(batch, ids):
        nuovi, stats = _genera(model, device, [ids[i] for i in indici], max(ks[i] for i in indici))
        n_nuovi = nuovi.shape[1]
        tempi.append(stats)
        for r, i in enumerate(indici):
            c = casi[i]
            testo = tok.decode(nuovi[r, :ks[i]].tolist())
            corretto = corpus.parse_answer(testo, c.attr, c.style, c.lang) == c.truth
            f_pre, f_dec = kvgen.flops(n_total, size.n_layer, size.n_embd, len(ids[i]), n_nuovi)
            esiti.append((corretto, f_pre, f_dec))

    prefill = np.array([s["prefill_s"] for s in tempi]) * 1000.0
    ttft = np.array([s["ttft_s"] for s in tempi]) * 1000.0
    decode = np.array([s["decode_s"] for s in tempi]) * 1000.0
    latenza = prefill + decode
    passi = sum(s["n_steps"] for s in tempi)
    n = len(casi)
    f_pre = np.array([e[1] for e in esiti], dtype=float)
    f_dec = np.array([e[2] for e in esiti], dtype=float)
    valori = {
        "corpus": job.corpus, "size": job.size, "seed": job.seed, "batch": batch,
        "n_questions": n,
        "params_nonemb": n_nonemb, "params_total": n_total,
        "prompt_tokens_mean": float(np.mean([len(r) for r in ids])),
        "answer_tokens_mean": float(np.mean(ks)),
        "prefill_ms": float(prefill.mean()),
        "ttft_ms": float(ttft.mean()),
        "decode_ms": float(decode.mean()),
        "latency_ms_mean": float(latenza.mean()),
        "latency_ms_p50": float(np.percentile(latenza, 50)),
        "latency_ms_p90": float(np.percentile(latenza, 90)),
        "ms_per_decode_token": float(decode.sum() / passi) if passi > 0 else float("nan"),
        "answers_per_s": float(n / (latenza.sum() / 1000.0)),
        "flops_prefill": float(f_pre.mean()),
        "flops_decode": float(f_dec.mean()),
        "flops_per_answer": float((f_pre + f_dec).mean()),
        "acc": float(np.mean([e[0] for e in esiti])),
    }
    return {col: valori[col] for col in INFER_COLUMNS}


def _append(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row], columns=list(INFER_COLUMNS)).to_csv(
        csv_path, mode="a", header=write_header, index=False)


def run_infer_c(out_csv: Path) -> None:
    """Misura C_FORMATS x C_SIZES x C_BATCHES; salta le combinazioni gia' in out_csv; scrive subito."""
    out_csv = Path(out_csv)
    fatti = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        fatti = {(str(r["corpus"]), str(r["size"]), int(r["batch"]))
                 for r in pd.read_csv(out_csv).to_dict("records")}

    persone_per_n: dict[int, list[Person]] = {}
    for formato in C_FORMATS:
        for taglia in C_SIZES:
            job = core_job(formato, taglia)
            for batch in C_BATCHES:
                if (formato, taglia, batch) in fatti:
                    continue
                if job.n_persons not in persone_per_n:
                    persone_per_n[job.n_persons] = corpus.load_persons(job.n_persons)
                riga = measure_c(job, persone_per_n[job.n_persons], batch)
                _append(out_csv, riga)
                fatti.add((formato, taglia, batch))
                print(f"{job.run_id} batch {batch}: p50 {riga['latency_ms_p50']:.2f} ms "
                      f"answers/s {riga['answers_per_s']:.2f} acc {riga['acc']:.3f}")


if __name__ == "__main__":
    run_infer_c(V2_RESULTS / "inference_c.csv")
