"""Transformer decoder-only stile GPT."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd deve essere multiplo di n_head"
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-LayerNorm con connessioni residue
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, ctx_len: int, n_layer: int, n_embd: int, n_head: int):
        super().__init__()
        self.ctx_len = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(ctx_len, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # peso condiviso tra embedding token e lm_head
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # proiezioni residue con std scalata per la profondita
        std_res = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=std_res)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=std_res)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Ritorna (logits, loss). loss=None senza targets; ignore_index=-100."""
        B, T = idx.shape
        if T > self.ctx_len:
            raise ValueError(f"sequenza di lunghezza {T} oltre ctx_len={self.ctx_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def count_params(self) -> tuple[int, int]:
        """(non_embedding, totale). Non-embedding esclude embedding token e posizione."""
        # parameters() non conta due volte i tensori condivisi
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return total - emb, total

    @torch.no_grad()
    def generate_greedy(self, idx: torch.Tensor, max_new_tokens: int, stop_ids: set[int] | None = None) -> torch.Tensor:
        """idx (B,T) senza padding; ritorna solo i token nuovi (B, max_new_tokens)."""
        B = idx.size(0)
        stop_t = None
        if stop_ids:
            stop_t = torch.tensor(sorted(stop_ids), dtype=torch.long, device=idx.device)
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        # ogni riga che ha emesso uno stop id ripete quello come padding
        pad = torch.zeros(B, dtype=torch.long, device=idx.device)
        new = idx.new_zeros((B, 0))
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx_len:])
            nxt = logits[:, -1, :].argmax(dim=-1)
            nxt = torch.where(done, pad, nxt)
            if stop_t is not None:
                hit = torch.isin(nxt, stop_t) & ~done
                pad = torch.where(hit, nxt, pad)
                done = done | hit
            new = torch.cat([new, nxt.unsqueeze(1)], dim=1)
            idx = torch.cat([idx, nxt.unsqueeze(1)], dim=1)
            if stop_t is not None and bool(done.all()):
                break
        # output sempre di lunghezza max_new_tokens
        missing = max_new_tokens - new.size(1)
        if missing > 0:
            fill = pad.unsqueeze(1).expand(B, missing)
            new = torch.cat([new, fill], dim=1)
        return new
