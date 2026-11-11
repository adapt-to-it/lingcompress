"""Tokenizer, cache dei token, training ed evaluation di un singolo run."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, DATA_DIR, EOS, PAD, RESULT_COLUMNS, RUNS_DIR, ExpConfig, ModelSize
from src.data import Person, epoch_texts, eval_prompt, extract_answer
from src.model import GPT


def _cache_dir(persons: list[Person], fmt: str, cfg: ExpConfig) -> Path:
    return DATA_DIR / f"{fmt}_n{len(persons)}_s{cfg.data_seed}"


def train_tokenizer(persons: list[Person], fmt: str, cfg: ExpConfig) -> Tokenizer:
    """BPE byte-level per formato, vocab cfg.vocab_size; salvato/ricaricato da DATA_DIR."""
    path = _cache_dir(persons, fmt, cfg) / "tokenizer.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=[EOS, PAD],
        # alfabeto completo dei 256 byte, cosi' ogni testo e' codificabile
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(epoch_texts(persons, fmt, 0, cfg.data_seed), trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def _encode_epoch(persons: list[Person], fmt: str, epoch: int, tok: Tokenizer, cfg: ExpConfig) -> list[list[int]]:
    texts = epoch_texts(persons, fmt, epoch, cfg.data_seed)
    return [enc.ids for enc in tok.encode_batch(texts)]


def epoch_tokens(persons: list[Person], fmt: str, epoch: int, tok: Tokenizer, ctx_len: int, cfg: ExpConfig):
    """Array uint16 (n_seq, ctx_len) dell'epoca: biografie intere impacchettate, PAD in coda; cache su disco."""
    path = _cache_dir(persons, fmt, cfg) / f"epoch{epoch}_ctx{ctx_len}.npy"
    if path.exists():
        return np.load(path)
    eos, pad = tok.token_to_id(EOS), tok.token_to_id(PAD)
    # ogni biografia e' preceduta da EOS; la sequenza termina con EOS dopo l'ultima
    righe, cur = [], []
    for ids in _encode_epoch(persons, fmt, epoch, tok, cfg):
        if len(ids) + 2 > ctx_len:
            raise ValueError(f"biografia di {len(ids)} token oltre ctx_len={ctx_len}")
        if len(cur) + len(ids) + 2 > ctx_len:
            righe.append(cur + [eos])
            cur = []
        cur.append(eos)
        cur.extend(ids)
    if cur:
        righe.append(cur + [eos])
    out = np.full((len(righe), ctx_len), pad, dtype=np.uint16)
    for i, riga in enumerate(righe):
        out[i, :len(riga)] = riga
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def _lr_at(step: int, total: int, warm: int, base: float, min_frac: float) -> float:
    if step < warm:
        return base * (step + 1) / warm
    progresso = min(1.0, (step - warm) / max(1, total - warm))
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progresso)))


def _salva(path: Path, stato: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(stato, tmp)
    os.replace(tmp, path)


def _stato(model, optim, step: int, epoch: int, batch: int, elapsed: float, tokens: int) -> dict:
    return {
        "model": model.state_dict(), "optim": optim.state_dict(), "step": step,
        "epoch": epoch, "batch": batch, "elapsed": elapsed, "tokens": tokens,
    }


def run_one(persons: list[Person], fmt: str, size: ModelSize, seed: int, cfg: ExpConfig) -> dict:
    """Train (riprendibile da checkpoint) + eval. Ritorna la riga per results.csv."""
    if not persons:
        raise ValueError("nessuna persona")
    n = len(persons)
    run_id = f"{cfg.tag}_{fmt}_{size.name}_s{seed}_n{n}_e{cfg.exposures}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    log_path = run_dir / "log.csv"

    stato = torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
    if stato is not None and stato.get("done"):
        return stato["result"]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = train_tokenizer(persons, fmt, cfg)
    pad = tok.token_to_id(PAD)
    lunghezze = [len(ids) for ids in _encode_epoch(persons, fmt, 0, tok, cfg)]
    tokens_per_bio = sum(lunghezze) / len(lunghezze)
    ctx_len = cfg.ctx_len or ((2 * (max(lunghezze) + 1) + 63) // 64) * 64

    bs = cfg.batch_size
    rows0 = epoch_tokens(persons, fmt, 0, tok, ctx_len, cfg)
    # il numero di sequenze varia poco tra epoche: il totale dei passi usa l'epoca 0
    total = cfg.exposures * math.ceil(len(rows0) / bs)
    warm = max(1, int(cfg.warmup_frac * total))

    model = GPT(vocab_size=tok.get_vocab_size(), ctx_len=ctx_len, n_layer=size.n_layer,
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
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": senza_decay, "weight_decay": 0.0}],
        lr=size.lr, betas=tuple(cfg.betas), fused=device.type == "cuda",
    )

    if stato is not None:
        model.load_state_dict(stato["model"])
        optim.load_state_dict(stato["optim"])
        step, ep0, b0 = stato["step"], stato["epoch"], stato["batch"]
        elapsed0, tokens = stato["elapsed"], stato["tokens"]
        print(f"{run_id}: ripreso da step {step}")
    else:
        step, ep0, b0, elapsed0, tokens = 0, 0, 0, 0.0, 0
        log_path.write_text("step,epoch,loss,lr,tok_s\n", encoding="utf-8")

    # loss media sull'ultimo 1% dei passi
    n_ultimi = max(1, int(0.01 * total))
    loss_finali = []
    pos_ep, pos_b = ep0, b0
    sessione_t0 = time.perf_counter()
    ultimo_ckpt = sessione_t0
    ultimo_log_t, ultimo_log_tok = sessione_t0, tokens

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch in range(ep0, cfg.exposures):
        rows = epoch_tokens(persons, fmt, epoch, tok, ctx_len, cfg)
        gen = torch.Generator().manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(len(rows), generator=gen).numpy()
        n_batch = math.ceil(len(rows) / bs)
        for b in range(b0 if epoch == ep0 else 0, n_batch):
            righe = rows[perm[b * bs:(b + 1) * bs]]
            tokens += int(np.count_nonzero(righe != pad))
            seq = torch.from_numpy(righe.astype(np.int64)).to(device)
            x = seq[:, :-1]
            y = seq[:, 1:].masked_fill(seq[:, 1:] == pad, -100)

            lr = _lr_at(step, total, warm, size.lr, cfg.min_lr_frac)
            for gruppo in optim.param_groups:
                gruppo["lr"] = lr
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            step += 1
            pos_ep, pos_b = epoch, b + 1
            if step > total - n_ultimi:
                loss_finali.append(loss.item())

            if step % cfg.log_every == 0:
                agora = time.perf_counter()
                tok_s = (tokens - ultimo_log_tok) / max(agora - ultimo_log_t, 1e-9)
                loss_v = loss.item()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{epoch},{loss_v:.6f},{lr:.6e},{tok_s:.1f}\n")
                print(f"{run_id} step {step}/{total} ep {epoch} loss {loss_v:.4f} "
                      f"lr {lr:.2e} tok/s {tok_s:.0f}")
                ultimo_log_t, ultimo_log_tok = agora, tokens

            if time.perf_counter() - ultimo_ckpt >= cfg.ckpt_every_s:
                elapsed = elapsed0 + time.perf_counter() - sessione_t0
                _salva(ckpt_path, _stato(model, optim, step, pos_ep, pos_b, elapsed, tokens))
                ultimo_ckpt = time.perf_counter()

    train_time = elapsed0 + time.perf_counter() - sessione_t0
    _salva(ckpt_path, _stato(model, optim, step, cfg.exposures, 0, train_time, tokens))

    peak_vram = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    t_eval = time.perf_counter()
    metriche = evaluate(model, tok, persons, fmt, cfg)
    eval_time = time.perf_counter() - t_eval
    final_loss = float(np.mean(loss_finali)) if loss_finali else float("nan")

    risultato = {
        "tag": cfg.tag, "format": fmt, "size": size.name, "seed": seed,
        "n_persons": n, "exposures": cfg.exposures,
        "n_layer": size.n_layer, "n_embd": size.n_embd, "n_head": size.n_head,
        "params_nonemb": n_nonemb, "params_total": n_total, "ctx_len": ctx_len,
        "tokens_per_bio": tokens_per_bio, "train_tokens": tokens, "steps": step,
        "flops": 6 * n_total * tokens, "train_time_s": train_time,
        "tokens_per_s": tokens / train_time if train_time > 0 else 0.0,
        "peak_vram_mb": peak_vram, "final_loss": final_loss, "eval_time_s": eval_time,
        **metriche,
    }
    riga = {col: risultato[col] for col in RESULT_COLUMNS}
    # run completo: un rilancio ritorna questa riga senza ripetere training ed eval
    _salva(ckpt_path, {**_stato(model, optim, step, cfg.exposures, 0, train_time, tokens),
                       "done": True, "result": riga})
    return riga


def evaluate(model, tok: Tokenizer, persons: list[Person], fmt: str, cfg: ExpConfig) -> dict:
    """Accuratezza exact-match greedy: {"acc_<attr>": float, ..., "acc_mean": float}."""
    campione = persons[:cfg.eval_persons] if cfg.eval_persons else persons
    device = next(model.parameters()).device
    eos = tok.token_to_id(EOS)

    # un caso per persona e attributo: (attributo, valore atteso, ids del prompt)
    casi = []
    for p in campione:
        for attr in ATTRS:
            ids = [eos] + tok.encode(eval_prompt(p, attr, fmt)).ids
            casi.append((attr, p.attrs[attr], ids))

    # prompt della stessa lunghezza insieme: generate_greedy non usa padding
    per_lunghezza: dict[int, list[int]] = {}
    for i, (_, _, ids) in enumerate(casi):
        per_lunghezza.setdefault(len(ids), []).append(i)

    corrette = {attr: 0 for attr in ATTRS}
    totali = {attr: 0 for attr in ATTRS}
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for indici in per_lunghezza.values():
            for k in range(0, len(indici), cfg.eval_batch):
                blocco = indici[k:k + cfg.eval_batch]
                idx = torch.tensor([casi[i][2] for i in blocco], dtype=torch.long, device=device)
                nuovi = model.generate_greedy(idx, max_new_tokens=cfg.max_new_tokens, stop_ids={eos})
                for i, riga in zip(blocco, nuovi.tolist()):
                    attr, atteso, _ = casi[i]
                    generato = tok.decode(riga, skip_special_tokens=False)
                    totali[attr] += 1
                    if extract_answer(generato, fmt) == atteso:
                        corrette[attr] += 1

    risultato = {f"acc_{a}": corrette[a] / totali[a] if totali[a] else 0.0 for a in ATTRS}
    risultato["acc_mean"] = sum(risultato[f"acc_{a}"] for a in ATTRS) / len(ATTRS)
    return risultato
