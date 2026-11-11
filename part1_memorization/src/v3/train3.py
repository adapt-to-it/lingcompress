"""Training ed evaluation v3 di un singolo Job: come v2, con batch vettorizzati e loss compilata."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ATTRS, PAD
from src.data import Person
from src.v2 import corpus
from src.v2.model2 import GPT2
from src.v2.spec import BIOS_PER_STEP, CORPORA, JOB_KEYS, RESULT_COLUMNS_V2, SIZES_V2, Job
from src.v2.train import (
    CKPT_EVERY_S, GRAD_CLIP, MIN_LR_FRAC, REF_CORPUS, STEADY_FROM, WARMUP_FRAC, WEIGHT_DECAY, BETAS,
    _leggi_epoca, _lr_at, _max_len, _salva, _stato, _sincronizza, get_tokenizer, prepare,
)
from src.v3.fastgen import evaluate_fast
from src.v3.spec3 import LOG_EVERY, V3_RUNS


def build_batch(tokens: np.ndarray, offsets: np.ndarray, idx: np.ndarray, T: int, pad: int) -> np.ndarray:
    """Biografie idx (da tokens/offsets) a destra con pad fino a T; (len(idx), T) int64, senza cicli."""
    inizio = offsets[idx]
    lunghezze = offsets[idx + 1] - inizio
    colonne = np.arange(T)
    valide = colonne[None, :] < lunghezze[:, None]
    pos = np.where(valide, inizio[:, None] + colonne[None, :], 0)
    return np.where(valide, tokens[pos], pad).astype(np.int64)


def _perm_epoche(seed: int, n_epoca: int, start: int):
    """(epoca, permutazione delle biografie) in ordine di stream, dalla posizione start: come _bio_stream."""
    epoca, salto = divmod(start, n_epoca)
    while True:
        gen = torch.Generator().manual_seed(seed * 100003 + epoca)
        perm = torch.randperm(n_epoca, generator=gen).numpy()
        yield epoca, perm[salto:]
        salto = 0
        epoca += 1


def _batch_stream(spec, n: int, seed: int, start: int, n_epoca: int, bps: int, T: int, pad: int):
    """Batch (bps, T) con le biografie di _bio_stream dalla posizione start; un batch puo' coprire due epoche."""
    flusso = _perm_epoche(seed, n_epoca, start)
    coda_epoca = np.empty(0, dtype=np.int64)
    coda_bio = np.empty(0, dtype=np.int64)
    caricate: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    while True:
        while len(coda_bio) < bps:
            epoca, perm = next(flusso)
            coda_epoca = np.concatenate([coda_epoca, np.full(len(perm), epoca, dtype=np.int64)])
            coda_bio = np.concatenate([coda_bio, perm.astype(np.int64)])
        epoche, bios = coda_epoca[:bps], coda_bio[:bps]
        coda_epoca, coda_bio = coda_epoca[bps:], coda_bio[bps:]
        batch = np.empty((bps, T), dtype=np.int64)
        for epoca in map(int, np.unique(epoche)):
            if epoca not in caricate:
                caricate.clear()  # le epoche precedenti non servono piu': lo stream e' monotono
                tokens, offsets, _ = _leggi_epoca(spec, n, epoca)
                caricate[epoca] = (tokens, offsets)
            tokens, offsets = caricate[epoca]
            sel = epoche == epoca
            batch[sel] = build_batch(tokens, offsets, bios[sel], T, pad)
        yield batch


def _loss_batch(model: GPT2, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy media sui target != -100: logits su tutte le posizioni, senza indicizzazione booleana."""
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        pos = torch.arange(x.shape[1], device=x.device)
        h = model.tok_emb(x) + model.pos_emb(pos)
        for block in model.blocks:
            h = block(h)
        logits = model.lm_head(model.ln_f(h))
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)


def run_job(job: Job, persons: list[Person], compile: bool = True, evaluate=None, runs_dir=None,
            vocab_pad: int = 1) -> dict:
    """Train (riprendibile) + eval. Ritorna la riga con esattamente RESULT_COLUMNS_V2.

    vocab_pad > 1 (v4): embedding e testa con V arrotondato al multiplo di vocab_pad; vocab e params_total
    restano quelli del V reale (le righe di padding non compaiono mai nei dati).
    """
    run_dir = (runs_dir or V3_RUNS) / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    log_path = run_dir / "log.csv"

    stato = torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
    if stato is not None and stato.get("done"):
        return stato["result"]

    spec = CORPORA[job.corpus]
    size = SIZES_V2[job.size]
    n = len(persons)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    epoche = -(-steps * bps // n_epoca)  # ceil
    prepare(spec, persons, epoche)
    ctx_len = max(_max_len(spec, n, e) for e in range(epoche))

    lr = size.lr * job.lr_mult
    warm = max(1, int(WARMUP_FRAC * steps))
    v_reale = tok.get_vocab_size()
    v_modello = -(-v_reale // vocab_pad) * vocab_pad
    model = GPT2(vocab_size=v_modello, ctx_len=ctx_len, n_layer=size.n_layer,
                 n_embd=size.n_embd, n_head=size.n_head).to(device)
    n_nonemb, n_total = model.count_params()
    n_total -= (v_modello - v_reale) * size.n_embd

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
    torch._dynamo.reset()  # un grafo per run: oltre cache_size_limit (8) dynamo tornerebbe eager
    loss_fn = torch.compile(_loss_batch, dynamic=False) if compile else _loss_batch

    if stato is not None:
        model.load_state_dict(stato["model"])
        optim.load_state_dict(stato["optim"])
        step, tokens, elapsed0 = stato["step"], stato["tokens"], stato["elapsed"]
        print(f"{job.run_id}: ripreso da step {step}")
    else:
        step, tokens, elapsed0 = 0, 0, 0.0
        log_path.write_text("step,loss,lr,tok_s\n", encoding="utf-8")

    # loss media sull'ultimo 1% dei passi, accumulata sulla GPU
    n_ultimi = max(1, int(0.01 * steps))
    somma_finale = torch.zeros((), device=device)
    n_finali = 0
    sessione_t0 = time.perf_counter()
    ultimo_ckpt = sessione_t0
    ultimo_log_t, ultimo_log_tok = sessione_t0, tokens
    steady_t0, steady_tok0 = None, None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    stream = _batch_stream(spec, n, job.seed, step * bps, n_epoca, bps, ctx_len, pad)
    while step < steps:
        batch = next(stream)
        reali = int(np.count_nonzero(batch[:, 1:] != pad))
        lotto = torch.from_numpy(batch)
        if device.type == "cuda":
            lotto = lotto.pin_memory()
        seq = lotto.to(device, non_blocking=device.type == "cuda")
        x = seq[:, :-1]
        y = seq[:, 1:].masked_fill(seq[:, 1:] == pad, -100)

        lr_ora = _lr_at(step, steps, warm, lr, MIN_LR_FRAC)
        for gruppo in optim.param_groups:
            gruppo["lr"] = lr_ora
        loss = loss_fn(model, x, y)
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
            somma_finale += loss.detach()
            n_finali += 1

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
    final_loss = somma_finale.item() / n_finali if n_finali else float("nan")

    t_eval = time.perf_counter()
    metriche = (evaluate or evaluate_fast)(model, tok, spec, persons, job.eval_persons)
    eval_time = time.perf_counter() - t_eval

    entropia = corpus.entropy_bits(n)
    bits = float(sum(metriche[f"acc_{a}"] * entropia[a] for a in ATTRS) * n)

    valori = {
        **{k: getattr(job, k) for k in JOB_KEYS},
        "eval_persons": job.eval_persons,
        "n_layer": size.n_layer, "n_embd": size.n_embd, "n_head": size.n_head,
        "lr": lr, "params_nonemb": n_nonemb, "params_total": n_total,
        "vocab": v_reale,
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
