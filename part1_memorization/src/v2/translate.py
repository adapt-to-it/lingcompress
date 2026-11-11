"""Fase C v2: strippers e traduttori decoder-only tra italiano_pieno e un formato compresso."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, EOS, PAD, ModelSize
from src.data import Person
from src.v2 import corpus, kvgen
from src.v2.model2 import GPT2
from src.v2.spec import (
    C_BATCHES, C_FORMATS, C_MAX_OUT_TOKENS, C_QUESTIONS, C_TRANSLATE_SIZE,
    C_TRANSLATE_TRAIN_PERSONS, N_PERSONS, SIZES_V2, TRANSLATE_COLUMNS, V2_DATA, V2_RUNS,
    VOCAB_SIZE,
)
from src.v2.train import (
    BETAS, GRAD_CLIP, MIN_LR_FRAC, WARMUP_FRAC, WEIGHT_DECAY, _lr_at, _salva, _sincronizza,
)

SEP = "<sep>"
EPOCHS = 20
SEQS_PER_STEP = 256
DIREZIONI = ("strip", "trans")


def pairs(persons: list[Person], idxs, fmt: str, direction: str) -> list[tuple[str, str, str, str]]:
    """(src, tgt, attr, truth) per persona in idxs e attributo in ATTRS, nell'ordine idxs poi ATTRS."""
    coppie = []
    for i in idxs:
        persona = persons[i]
        for attr in ATTRS:
            P = corpus.eval_prompt(persona, attr, "pieno", "it")
            C = corpus.eval_prompt(persona, attr, fmt, "it")
            v = persona.attrs[attr]
            if direction == "strip":
                coppie.append((P, C, attr, v))
            elif direction == "trans":
                s = corpus.surface(attr, v, fmt, "it")
                coppie.append((C + " " + s, P + " " + v, attr, v))
            else:
                raise ValueError(f"direzione sconosciuta: {direction}")
    return coppie


def _tok_path(fmt: str) -> Path:
    return V2_DATA / f"tok_translate_{fmt}" / "tokenizer.json"


def get_tok(fmt: str, persons: list[Person]) -> Tokenizer:
    """BPE byte-level VOCAB_SIZE con cifre singole e token <sep>; addestrato su src/tgt di entrambe le direzioni."""
    path = _tok_path(fmt)
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True),
         pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOS, PAD, SEP],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    lo, hi = C_TRANSLATE_TRAIN_PERSONS
    testi = []
    for direzione in DIREZIONI:
        for src, tgt, _, _ in pairs(persons, range(lo, hi), fmt, direzione):
            testi.extend((src, tgt))
    tok.train_from_iterator(testi, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def _prefisso(tok: Tokenizer, src: str) -> list[int]:
    """[EOS] src <sep>: prompt dato al modello."""
    return [tok.token_to_id(EOS), *tok.encode(src).ids, tok.token_to_id(SEP)]


def _sequenze(tok: Tokenizer, coppie) -> list[tuple[list[int], int]]:
    """(ids, lunghezza del prefisso) per coppia: [EOS] src <sep> tgt [EOS]."""
    eos = tok.token_to_id(EOS)
    sequenze = []
    for src, tgt, _, _ in coppie:
        pref = _prefisso(tok, src)
        sequenze.append((pref + tok.encode(tgt).ids + [eos], len(pref)))
    return sequenze


def _batch(righe: list[tuple[list[int], int]], pad: int, device: torch.device):
    """x = seq[:-1] con PAD; y = target solo su tgt e EOS finale, -100 altrove."""
    T = max(len(ids) for ids, _ in righe) - 1
    x = np.full((len(righe), T), pad, dtype=np.int64)
    y = np.full((len(righe), T), -100, dtype=np.int64)
    for i, (ids, pre) in enumerate(righe):
        x[i, :len(ids) - 1] = ids[:-1]
        y[i, pre - 1:len(ids) - 1] = ids[pre:]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def _modello(tok: Tokenizer, ctx_len: int) -> GPT2:
    size: ModelSize = SIZES_V2[C_TRANSLATE_SIZE]
    return GPT2(vocab_size=tok.get_vocab_size(), ctx_len=ctx_len, n_layer=size.n_layer,
                n_embd=size.n_embd, n_head=size.n_head)


def _ottimizzatore(model: nn.Module, lr: float, device: torch.device) -> torch.optim.Optimizer:
    # niente weight decay su bias, LayerNorm ed embedding (come train.run_job)
    senza_decay_ids = set()
    for modulo in model.modules():
        if isinstance(modulo, (nn.LayerNorm, nn.Embedding)):
            senza_decay_ids.update(id(p) for p in modulo.parameters(recurse=False))
    decay, senza_decay = [], []
    for nome, p in model.named_parameters():
        if p.ndim < 2 or nome.endswith("bias") or id(p) in senza_decay_ids:
            senza_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": senza_decay, "weight_decay": 0.0}],
        lr=lr, betas=BETAS, fused=device.type == "cuda",
    )


def train(fmt: str, direction: str, persons: list[Person]) -> tuple[GPT2, Tokenizer, dict]:
    """Addestra (o carica dal checkpoint completo) il traduttore fmt/direction. Ritorna (model, tok, info)."""
    run_dir = V2_RUNS / f"C_{direction}_{fmt}"
    ckpt_path = run_dir / "ckpt.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tok(fmt, persons)

    if ckpt_path.exists():
        stato = torch.load(ckpt_path, map_location="cpu")
        if stato.get("done"):
            model = _modello(tok, stato["ctx_len"]).to(device)
            model.load_state_dict(stato["model"])
            model.eval()
            return model, tok, stato["info"]

    lo, hi = C_TRANSLATE_TRAIN_PERSONS
    sequenze = _sequenze(tok, pairs(persons, range(lo, hi), fmt, direction))
    ctx_len = max(len(ids) for ids, _ in sequenze) + C_MAX_OUT_TOKENS
    pad = tok.token_to_id(PAD)

    torch.manual_seed(0)
    size: ModelSize = SIZES_V2[C_TRANSLATE_SIZE]
    model = _modello(tok, ctx_len).to(device)
    n_nonemb, n_total = model.count_params()
    optim = _ottimizzatore(model, size.lr, device)

    n = len(sequenze)
    steps = EPOCHS * math.ceil(n / SEQS_PER_STEP)
    warm = max(1, int(WARMUP_FRAC * steps))
    tokens = 0
    step = 0
    model.train()
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        perm = torch.randperm(n).tolist()
        for k in range(0, n, SEQS_PER_STEP):
            righe = [sequenze[i] for i in perm[k:k + SEQS_PER_STEP]]
            x, y = _batch(righe, pad, device)
            lr_ora = _lr_at(step, steps, warm, size.lr, MIN_LR_FRAC)
            for gruppo in optim.param_groups:
                gruppo["lr"] = lr_ora
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model.loss(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            step += 1
            tokens += sum(len(ids) - 1 for ids, _ in righe)
    _sincronizza(device)
    train_time = time.perf_counter() - t0

    info = {"train_steps": steps, "train_time_s": train_time,
            "train_flops": 6 * n_total * tokens}
    run_dir.mkdir(parents=True, exist_ok=True)
    _salva(ckpt_path, {"model": model.state_dict(), "ctx_len": ctx_len, "info": info, "done": True})
    model.eval()
    return model, tok, info


def _media(valori: list[float]) -> float:
    return float(np.mean(valori)) if valori else float("nan")


def run(model: GPT2, tok: Tokenizer, srcs: list[str], batch: int) -> tuple[list[str], dict]:
    """Generazione greedy per ogni src (gruppi di stessa lunghezza di prefisso, a blocchi di batch).

    Statistiche per richiesta (batch 1) o per blocco (batch > 1); tempi in ms.
    """
    device = next(model.parameters()).device
    n_total = model.count_params()[1]
    n_layer = len(model.blocks)
    n_embd = model.tok_emb.embedding_dim
    eos = tok.token_to_id(EOS)

    prefissi = [_prefisso(tok, s) for s in srcs]
    gruppi: dict[int, list[int]] = {}
    for i, pref in enumerate(prefissi):
        gruppi.setdefault(len(pref), []).append(i)
    unita = [g[k:k + batch] for g in gruppi.values() for k in range(0, len(g), batch)]

    outs = [""] * len(srcs)
    in_lens, out_lens, flops = [], [], []
    prefill_s, decode_s = [], []
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for gruppo in unita:
            idx = torch.tensor([prefissi[i] for i in gruppo], dtype=torch.long, device=device)
            nuovi, st = kvgen.generate(model, idx, max_new_tokens=C_MAX_OUT_TOKENS, stop_id=eos)
            k_new = nuovi.shape[1]
            for i, riga in zip(gruppo, nuovi.tolist()):
                if eos in riga:
                    riga = riga[:riga.index(eos)]
                outs[i] = tok.decode(riga)
                in_lens.append(len(prefissi[i]))
                out_lens.append(len(riga))
                pre, dec = kvgen.flops(n_total, n_layer, n_embd, len(prefissi[i]), k_new)
                flops.append(pre + dec)
            prefill_s.append(st["prefill_s"])
            decode_s.append(st["decode_s"])

    latenze_s = [p + d for p, d in zip(prefill_s, decode_s)]
    tempo_totale = sum(latenze_s)
    stats = {
        "prefill_ms": _media([p * 1000 for p in prefill_s]),
        "decode_ms": _media([d * 1000 for d in decode_s]),
        "latency_ms_mean": _media([t * 1000 for t in latenze_s]),
        "answers_per_s": len(srcs) / tempo_totale if tempo_totale > 0 else float("nan"),
        "flops_per_request": _media(flops),
        "in_tokens_mean": _media(in_lens),
        "out_tokens_mean": _media(out_lens),
    }
    return outs, stats


def _riga(modello: str, fmt: str, batch: int, direction: str, model: GPT2, info: dict,
          domande, outs: list[str], st: dict) -> dict:
    n_nonemb, n_total = model.count_params()
    sentence_em = _media([out.strip() == tgt for out, (_, tgt, _, _) in zip(outs, domande)])
    values_em = float("nan")
    if direction == "trans":
        valori = []
        for out, (_, tgt, attr, truth) in zip(outs, domande):
            P = tgt[:-(len(truth) + 1)]  # tgt = P + " " + truth
            valori.append(out.startswith(P)
                          and corpus.parse_answer(out[len(P):], attr, "pieno", "it") == truth)
        values_em = _media(valori)
    return {
        "model": modello, "format": fmt, "batch": batch, "n_questions": len(domande),
        "params_nonemb": n_nonemb, "params_total": n_total,
        "train_steps": info["train_steps"], "train_time_s": info["train_time_s"],
        "train_flops": info["train_flops"],
        "in_tokens_mean": st["in_tokens_mean"], "out_tokens_mean": st["out_tokens_mean"],
        "sentence_em": sentence_em, "values_em": values_em,
        "prefill_ms": st["prefill_ms"], "decode_ms": st["decode_ms"],
        "latency_ms_mean": st["latency_ms_mean"], "answers_per_s": st["answers_per_s"],
        "flops_per_request": st["flops_per_request"],
    }


def run_translate(out_csv: Path) -> None:
    """Valuta ogni traduttore su C_QUESTIONS domande (persone 0..) e aggiunge una riga per batch; riprendibile."""
    out_csv = Path(out_csv)
    fatti: set[tuple[str, int]] = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        with open(out_csv, newline="", encoding="utf-8") as f:
            fatti = {(r["model"], int(r["batch"])) for r in csv.DictReader(f)}
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TRANSLATE_COLUMNS)

    persons = corpus.load_persons(N_PERSONS)
    n_persone = math.ceil(C_QUESTIONS / len(ATTRS))
    for fmt in (f for f in C_FORMATS if f != "italiano_pieno"):
        for direction in DIREZIONI:
            modello = f"{direction}_{fmt}"
            in_sospeso = [b for b in C_BATCHES if (modello, b) not in fatti]
            if not in_sospeso:
                continue
            model, tok, info = train(fmt, direction, persons)
            domande = pairs(persons, range(n_persone), fmt, direction)[:C_QUESTIONS]
            srcs = [src for src, _, _, _ in domande]
            for batch in in_sospeso:
                outs, st = run(model, tok, srcs, batch)
                riga = _riga(modello, fmt, batch, direction, model, info, domande, outs, st)
                with open(out_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=TRANSLATE_COLUMNS).writerow(riga)
