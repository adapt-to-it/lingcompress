"""Training ed evaluation v2 di un singolo Job."""
from __future__ import annotations

import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, EOS, PAD
from src.data import Person
from src.v2 import corpus
from src.v2.model2 import GPT2
from src.v2.spec import (
    BIOS_PER_STEP, CORPORA, JOB_KEYS, MAX_NEW_TOKENS, RESULT_COLUMNS_V2, SIZES_V2,
    TOKENIZER_SOURCES, V2_DATA, V2_RUNS, VOCAB_SIZE, CorpusSpec, Job,
)

REF_CORPUS = "italiano_pieno"   # riferimento del protocollo "esposizioni"
STEADY_FROM = 50                # throughput stazionario: dal passo 50 alla fine
WARMUP_FRAC = 0.02
MIN_LR_FRAC = 0.1
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
GRAD_CLIP = 1.0
CKPT_EVERY_S = 300.0
LOG_EVERY = 100
EVAL_BATCH = 2048
POOL_MAX = 12


def _tok_path(spec: CorpusSpec, n: int) -> Path:
    return V2_DATA / f"tok_{spec.tokenizer}_n{n}" / "tokenizer.json"


def _epoch_path(spec: CorpusSpec, n: int, epoch: int) -> Path:
    return V2_DATA / f"{spec.name}_n{n}" / f"epoch{epoch}.npz"


def get_tokenizer(spec: CorpusSpec, persons: list[Person]) -> Tokenizer:
    """BPE byte-level VOCAB_SIZE per chiave spec.tokenizer (+ cifre singole se split_digits); cache su disco."""
    path = _tok_path(spec, len(persons))
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if spec.split_digits:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level])
    else:
        tok.pre_tokenizer = byte_level
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOS, PAD],
        # alfabeto completo dei 256 byte, cosi' ogni testo e' codificabile
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    fonti = TOKENIZER_SOURCES.get(spec.tokenizer, (spec.name,))
    testi = [t for nome in fonti for t in corpus.epoch_texts(CORPORA[nome], persons, 0)]
    tok.train_from_iterator(testi, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def _genera_epoca(args: tuple) -> None:
    """Worker del Pool (top-level, spawn-safe): tokenizza un'epoca e la salva in out_path."""
    nome, n, epoca, tok_path, out_path = args
    spec = CORPORA[nome]
    persone = corpus.load_persons(n)
    tok = Tokenizer.from_file(str(tok_path))
    eos = tok.token_to_id(EOS)
    codificate = tok.encode_batch(corpus.epoch_texts(spec, persone, epoca))
    biografie = [[eos, *enc.ids, eos] for enc in codificate]
    lunghezze = np.array([len(b) for b in biografie], dtype=np.int64)
    offsets = np.zeros(len(biografie) + 1, dtype=np.int64)
    np.cumsum(lunghezze, out=offsets[1:])
    tokens = np.fromiter((t for b in biografie for t in b), dtype=np.uint16, count=int(offsets[-1]))
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, tokens=tokens, offsets=offsets,
                 tokens_per_bio=np.float64(lunghezze.mean() - 2),  # senza EOS
                 max_len=np.int64(lunghezze.max()))                 # con EOS
    os.replace(tmp, out_path)


def prepare(spec: CorpusSpec, persons: list[Person], n_epochs: int) -> None:
    """Genera e tokenizza in parallelo (multiprocessing) le epoche mancanti nella cache del corpus."""
    n = len(persons)
    get_tokenizer(spec, persons)  # il worker ricarica il tokenizer da disco
    mancanti = [e for e in range(n_epochs) if not _epoch_path(spec, n, e).exists()]
    if not mancanti:
        return
    _epoch_path(spec, n, 0).parent.mkdir(parents=True, exist_ok=True)
    argomenti = [(spec.name, n, e, _tok_path(spec, n), _epoch_path(spec, n, e)) for e in mancanti]
    with multiprocessing.Pool(min(POOL_MAX, len(argomenti))) as pool:
        pool.map(_genera_epoca, argomenti)


def _leggi_epoca(spec: CorpusSpec, n: int, epoca: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(tokens uint16, offsets int64, tokens per bio senza EOS) di un'epoca in cache."""
    with np.load(_epoch_path(spec, n, epoca)) as z:
        return z["tokens"], z["offsets"], float(z["tokens_per_bio"])


def _max_len(spec: CorpusSpec, n: int, epoca: int) -> int:
    with np.load(_epoch_path(spec, n, epoca)) as z:
        return int(z["max_len"])


def _bio_stream(spec: CorpusSpec, n: int, seed: int, start: int, n_epoca: int):
    """Biografie (array uint16 con EOS) in ordine di stream, dalla posizione start."""
    epoca, salto = divmod(start, n_epoca)
    while True:
        tokens, offsets, _ = _leggi_epoca(spec, n, epoca)
        gen = torch.Generator().manual_seed(seed * 100003 + epoca)
        perm = torch.randperm(n_epoca, generator=gen).tolist()
        for i in perm[salto:]:
            yield tokens[offsets[i]:offsets[i + 1]]
        salto = 0
        epoca += 1


def _lr_at(step: int, total: int, warm: int, base: float, min_frac: float) -> float:
    if step < warm:
        return base * (step + 1) / warm
    progresso = min(1.0, (step - warm) / max(1, total - warm))
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progresso)))


def _salva(path: Path, stato: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(stato, tmp)
    os.replace(tmp, path)


def _stato(model, optim, step: int, tokens: int, elapsed: float) -> dict:
    return {"model": model.state_dict(), "optim": optim.state_dict(),
            "step": step, "tokens": tokens, "elapsed": elapsed}


def _sincronizza(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _evaluate(model, tok: Tokenizer, spec: CorpusSpec, persons: list[Person], eval_persons: int) -> dict:
    """Accuratezza greedy per attributo (in e cross) e cross_lenient_mean."""
    device = next(model.parameters()).device
    eos = tok.token_to_id(EOS)
    casi = corpus.eval_cases(spec, persons, eval_persons)
    ids = [[eos] + tok.encode(c.prompt).ids for c in casi]

    # prompt della stessa lunghezza insieme: generate_greedy non usa padding
    per_lunghezza: dict[int, list[int]] = {}
    for i, riga in enumerate(ids):
        per_lunghezza.setdefault(len(riga), []).append(i)

    testi = [""] * len(casi)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for indici in per_lunghezza.values():
            for k in range(0, len(indici), EVAL_BATCH):
                blocco = indici[k:k + EVAL_BATCH]
                idx = torch.tensor([ids[i] for i in blocco], dtype=torch.long, device=device)
                nuovi = model.generate_greedy(idx, max_new_tokens=MAX_NEW_TOKENS, stop_ids={eos})
                for i, riga in zip(blocco, nuovi.tolist()):
                    testi[i] = tok.decode(riga, skip_special_tokens=True)

    in_ok = {a: [] for a in ATTRS}
    cross_ok = {a: [] for a in ATTRS}
    lenient = []
    for caso, testo in zip(casi, testi):
        giusta = corpus.parse_answer(testo, caso.attr, caso.style, caso.lang) == caso.truth
        if caso.kind == "in":
            in_ok[caso.attr].append(giusta)
        else:
            cross_ok[caso.attr].append(giusta)
            lenient.append(
                corpus.parse_answer(testo, caso.attr, "pieno", "it") == caso.truth
                or corpus.parse_answer(testo, caso.attr, "pieno", "en") == caso.truth)

    risultato = {}
    for a in ATTRS:
        risultato[f"acc_{a}"] = float(np.mean(in_ok[a])) if in_ok[a] else 0.0
        risultato[f"cross_{a}"] = float(np.mean(cross_ok[a])) if cross_ok[a] else float("nan")
    risultato["acc_mean"] = float(np.mean([risultato[f"acc_{a}"] for a in ATTRS]))
    risultato["cross_mean"] = float(np.mean([risultato[f"cross_{a}"] for a in ATTRS]))
    risultato["cross_lenient_mean"] = float(np.mean(lenient)) if lenient else float("nan")
    return risultato


def run_job(job: Job, persons: list[Person]) -> dict:
    """Train (riprendibile) + eval. Ritorna la riga con esattamente RESULT_COLUMNS_V2."""
    run_dir = V2_RUNS / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    log_path = run_dir / "log.csv"

    stato = torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
    if stato is not None and stato.get("done"):
        return stato["result"]

    spec = CORPORA[job.corpus]
    size = SIZES_V2[job.size]
    n = len(persons)
    torch.manual_seed(job.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = get_tokenizer(spec, persons)
    pad = tok.token_to_id(PAD)
    prepare(spec, persons, 1)
    _, offsets0, tokens_per_bio = _leggi_epoca(spec, n, 0)
    n_epoca = len(offsets0) - 1

    # passi: stesso numero di biografie per passo; esposizioni: stessi token per passo
    if job.protocol == "passi":
        bps = BIOS_PER_STEP
    elif job.protocol == "esposizioni":
        riferimento = CORPORA[REF_CORPUS]
        prepare(riferimento, persons, 1)
        _, _, tokens_rif = _leggi_epoca(riferimento, n, 0)
        bps = max(1, round(BIOS_PER_STEP * tokens_rif / tokens_per_bio))
    else:
        raise ValueError(f"protocollo sconosciuto: {job.protocol}")

    steps = max(1, round(job.exposures * n / bps))
    epoche = math.ceil(steps * bps / n_epoca)
    prepare(spec, persons, epoche)
    ctx_len = max(_max_len(spec, n, e) for e in range(epoche))

    lr = size.lr * job.lr_mult
    warm = max(1, int(WARMUP_FRAC * steps))
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=ctx_len, n_layer=size.n_layer,
                 n_embd=size.n_embd, n_head=size.n_head).to(device)
    n_nonemb, n_total = model.count_params()

    # niente weight decay su bias, LayerNorm ed embedding
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
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": senza_decay, "weight_decay": 0.0}],
        lr=lr, betas=BETAS, fused=device.type == "cuda",
    )

    if stato is not None:
        model.load_state_dict(stato["model"])
        optim.load_state_dict(stato["optim"])
        step, tokens, elapsed0 = stato["step"], stato["tokens"], stato["elapsed"]
        print(f"{job.run_id}: ripreso da step {step}")
    else:
        step, tokens, elapsed0 = 0, 0, 0.0
        log_path.write_text("step,loss,lr,tok_s\n", encoding="utf-8")

    # loss media sull'ultimo 1% dei passi
    n_ultimi = max(1, int(0.01 * steps))
    loss_finali = []
    sessione_t0 = time.perf_counter()
    ultimo_ckpt = sessione_t0
    ultimo_log_t, ultimo_log_tok = sessione_t0, tokens
    steady_t0, steady_tok0 = None, None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    stream = _bio_stream(spec, n, job.seed, step * bps, n_epoca)
    while step < steps:
        bios = [next(stream) for _ in range(bps)]
        T = max(len(b) for b in bios)
        righe = np.full((bps, T), pad, dtype=np.int64)
        for i, b in enumerate(bios):
            righe[i, :len(b)] = b
        reali = int(np.count_nonzero(righe[:, 1:] != pad))
        seq = torch.from_numpy(righe).to(device)
        x = seq[:, :-1]
        y = seq[:, 1:].masked_fill(seq[:, 1:] == pad, -100)

        lr_ora = _lr_at(step, steps, warm, lr, MIN_LR_FRAC)
        for gruppo in optim.param_groups:
            gruppo["lr"] = lr_ora
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model.loss(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        step += 1
        tokens += reali

        if step == STEADY_FROM:
            _sincronizza(device)
            steady_t0, steady_tok0 = time.perf_counter(), tokens
        if step > steps - n_ultimi:
            loss_finali.append(loss.item())

        if step % LOG_EVERY == 0:
            agora = time.perf_counter()
            tok_s = (tokens - ultimo_log_tok) / max(agora - ultimo_log_t, 1e-9)
            loss_v = loss.item()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss_v:.6f},{lr_ora:.6e},{tok_s:.1f}\n")
            print(f"{job.run_id} step {step}/{steps} loss {loss_v:.4f} "
                  f"lr {lr_ora:.2e} tok/s {tok_s:.0f}")
            ultimo_log_t, ultimo_log_tok = agora, tokens

        if time.perf_counter() - ultimo_ckpt >= CKPT_EVERY_S:
            elapsed = elapsed0 + time.perf_counter() - sessione_t0
            _salva(ckpt_path, _stato(model, optim, step, tokens, elapsed))
            ultimo_ckpt = time.perf_counter()

    _sincronizza(device)
    train_time = elapsed0 + time.perf_counter() - sessione_t0
    if steady_t0 is not None and steps > STEADY_FROM:
        steady_s = time.perf_counter() - steady_t0
        steady_tok_s = (tokens - steady_tok0) / steady_s
        steady_facts_s = len(ATTRS) * (steps - STEADY_FROM) * bps / steady_s
    else:
        steady_tok_s = steady_facts_s = float("nan")
    _salva(ckpt_path, _stato(model, optim, step, tokens, train_time))

    peak_vram = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    final_loss = float(np.mean(loss_finali)) if loss_finali else float("nan")

    t_eval = time.perf_counter()
    metriche = _evaluate(model, tok, spec, persons, job.eval_persons)
    eval_time = time.perf_counter() - t_eval

    entropia = corpus.entropy_bits(n)
    bits = float(sum(metriche[f"acc_{a}"] * entropia[a] for a in ATTRS) * n)

    valori = {
        **{k: getattr(job, k) for k in JOB_KEYS},
        "eval_persons": job.eval_persons,
        "n_layer": size.n_layer, "n_embd": size.n_embd, "n_head": size.n_head,
        "lr": lr, "params_nonemb": n_nonemb, "params_total": n_total,
        "vocab": tok.get_vocab_size(),
        "tokens_per_bio": tokens_per_bio, "bios_per_step": bps, "steps": steps,
        "train_tokens": tokens, "flops": 6 * n_total * tokens, "train_time_s": train_time,
        "steady_tokens_per_s": steady_tok_s, "steady_facts_per_s": steady_facts_s,
        "peak_vram_mb": peak_vram, "final_loss": final_loss, "eval_time_s": eval_time,
        **metriche,
        "bits_memorized": bits, "bits_per_param": bits / n_nonemb,
    }
    riga = {col: valori[col] for col in RESULT_COLUMNS_V2}
    # run completo: un rilancio ritorna questa riga senza ripetere training ed eval
    _salva(ckpt_path, {**_stato(model, optim, step, tokens, train_time),
                       "done": True, "result": riga})
    return riga
