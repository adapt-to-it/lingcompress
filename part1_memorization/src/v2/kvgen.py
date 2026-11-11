"""Generazione greedy con cache KV sui pesi di src.model.GPT (senza modificarlo)."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.model import GPT, Block


def _sincronizza(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _blocco(block: Block, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None):
    """Un blocco su x (B,T,C) con la cache past (B,H,S,hs) delle posizioni precedenti."""
    attn = block.attn
    B, T, C = x.shape
    q, k, v = attn.c_attn(block.ln_1(x)).split(C, dim=2)
    hs = C // attn.n_head
    q = q.view(B, T, attn.n_head, hs).transpose(1, 2)
    k = k.view(B, T, attn.n_head, hs).transpose(1, 2)
    v = v.view(B, T, attn.n_head, hs).transpose(1, 2)
    if past is not None:
        k = torch.cat([past[0], k], dim=2)
        v = torch.cat([past[1], v], dim=2)
    # con la cache il token nuovo vede tutte le posizioni salvate: maschera causale solo al prefill
    y = F.scaled_dot_product_attention(q, k, v, is_causal=(past is None))
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    x = x + attn.c_proj(y)
    x = x + block.mlp(block.ln_2(x))
    return x, (k, v)


def _passo(model: GPT, idx: torch.Tensor, start: int, past):
    """Logits dell'ultimo token di idx (B,T) che occupano le posizioni da start, e nuove cache."""
    B, T = idx.shape
    pos = torch.arange(start, start + T, device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    nuove = []
    for i, block in enumerate(model.blocks):
        x, kv = _blocco(block, x, None if past is None else past[i])
        nuove.append(kv)
    x = model.ln_f(x[:, -1:, :])
    return model.lm_head(x)[:, -1, :], nuove


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, stop_id: int | None = None):
    """idx (B,T) senza padding; ritorna (token nuovi (B,K) con K <= max_new_tokens, statistiche)."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens deve essere almeno 1")
    B, P = idx.shape
    # la cache non scorre come il ritaglio di generate_greedy: tutte le posizioni devono stare in ctx_len
    if P + max_new_tokens - 1 > model.ctx_len:
        raise ValueError(f"prompt di {P} token e {max_new_tokens} nuovi oltre ctx_len={model.ctx_len}")
    device = idx.device

    t0 = time.perf_counter()
    logits, past = _passo(model, idx, 0, None)
    _sincronizza(device)
    t1 = time.perf_counter()
    nxt = logits.argmax(dim=-1)
    _sincronizza(device)
    t2 = time.perf_counter()

    tokens = [nxt]
    # ogni riga che ha emesso stop_id ripete quello come padding, come generate_greedy
    done = torch.zeros(B, dtype=torch.bool, device=device)
    if stop_id is not None:
        done = nxt == stop_id

    passi = 0
    while len(tokens) < max_new_tokens and not (stop_id is not None and bool(done.all())):
        logits, past = _passo(model, nxt.unsqueeze(1), P + passi, past)
        nxt = logits.argmax(dim=-1)
        if stop_id is not None:
            nxt = torch.where(done, torch.full_like(nxt, stop_id), nxt)
            done = done | (nxt == stop_id)
        tokens.append(nxt)
        passi += 1
    _sincronizza(device)
    t3 = time.perf_counter()

    new = torch.stack(tokens, dim=1)
    stats = {"prefill_s": t1 - t0, "ttft_s": t2 - t0, "decode_s": t3 - t2, "n_steps": passi}
    return new, stats


def flops(params_total: int, n_layer: int, n_embd: int, prompt_len: int, n_new: int) -> tuple[int, int]:
    """FLOPs stimati (prefill, decode) per un prompt di prompt_len token e n_new token generati."""
    P = prompt_len
    prefill = 2 * params_total * P + 2 * n_layer * n_embd * P ** 2
    decode = sum(2 * params_total + 4 * n_layer * n_embd * (P + i) for i in range(n_new))
    return prefill, decode
