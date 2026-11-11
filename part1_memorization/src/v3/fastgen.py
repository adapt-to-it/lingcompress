"""Generazione greedy con cache KV preallocata e CUDA graph (v3): stessi token di src.v2.kvgen."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.config import ATTRS, EOS
from src.data import Person
from src.model import GPT
from src.v2 import corpus
from src.v2.spec import MAX_NEW_TOKENS, CorpusSpec
from src.v2.train import EVAL_BATCH
from src.v3.spec3 import DONE_CHECK_EVERY


def _sincronizza(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class StaticGen:
    """Cache (max_batch, H, ctx_len, hs) per layer; ogni passo di decodifica legge buffer statici."""

    def __init__(self, model: GPT, max_batch: int, dtype: torch.dtype = torch.bfloat16, use_graphs: bool = True):
        self.model = model
        self.max_batch = max_batch
        self.dtype = dtype
        self.device = next(model.parameters()).device
        self.use_graphs = use_graphs and self.device.type == "cuda"
        n_head = model.blocks[0].attn.n_head
        hs = model.tok_emb.embedding_dim // n_head
        forma = (max_batch, n_head, model.ctx_len, hs)
        self.k = [torch.zeros(forma, dtype=dtype, device=self.device) for _ in model.blocks]
        self.v = [torch.zeros(forma, dtype=dtype, device=self.device) for _ in model.blocks]
        self.tok = torch.zeros(max_batch, dtype=torch.long, device=self.device)   # token di ingresso
        self.posbuf = torch.zeros(1, dtype=torch.long, device=self.device)        # posizione del passo
        self.ar = torch.arange(model.ctx_len, device=self.device)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _prefill(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits dell'ultimo token del prompt (B,V); scrive la cache in [0,P)."""
        m = self.model
        B, P = idx.shape
        pos = torch.arange(P, device=self.device)
        x = m.tok_emb(idx) + m.pos_emb(pos)
        for i, block in enumerate(m.blocks):
            attn = block.attn
            C = x.shape[-1]
            q, k, v = attn.c_attn(block.ln_1(x)).split(C, dim=2)
            hs = C // attn.n_head
            q = q.view(B, P, attn.n_head, hs).transpose(1, 2)
            k = k.view(B, P, attn.n_head, hs).transpose(1, 2)
            v = v.view(B, P, attn.n_head, hs).transpose(1, 2)
            self.k[i][:B, :, :P].copy_(k)
            self.v[i][:B, :, :P].copy_(v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, P, C)
            x = x + attn.c_proj(y)
            x = x + block.mlp(block.ln_2(x))
        x = m.ln_f(x[:, -1:, :])
        return m.lm_head(x)[:, -1, :]

    def _decode(self, B: int) -> torch.Tensor:
        """Un passo sui token tok[:B] alla posizione posbuf; attende su tutto ctx_len con maschera."""
        m = self.model
        x = m.tok_emb(self.tok[:B].unsqueeze(1)) + m.pos_emb(self.posbuf)
        mask = (self.ar <= self.posbuf).view(1, 1, 1, -1)
        for i, block in enumerate(m.blocks):
            attn = block.attn
            C = x.shape[-1]
            q, k, v = attn.c_attn(block.ln_1(x)).split(C, dim=2)
            hs = C // attn.n_head
            q = q.view(B, 1, attn.n_head, hs).transpose(1, 2)
            k = k.view(B, 1, attn.n_head, hs).transpose(1, 2)
            v = v.view(B, 1, attn.n_head, hs).transpose(1, 2)
            ck = self.k[i][:B]
            cv = self.v[i][:B]
            ck.index_copy_(2, self.posbuf, k.to(ck.dtype))
            cv.index_copy_(2, self.posbuf, v.to(cv.dtype))
            y = F.scaled_dot_product_attention(q, ck.to(q.dtype), cv.to(q.dtype), attn_mask=mask)
            y = y.transpose(1, 2).contiguous().view(B, 1, C)
            x = x + attn.c_proj(y)
            x = x + block.mlp(block.ln_2(x))
        x = m.ln_f(x)
        return m.lm_head(x)[:, -1, :].argmax(dim=-1)

    def _capture(self, B: int) -> None:
        # riscaldamento con gli stessi ingressi del passo corrente: riscrive la stessa cache, quindi e' innocuo
        flusso = torch.cuda.Stream()
        flusso.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(flusso):
            for _ in range(2):
                self._decode(B)
        torch.cuda.current_stream().wait_stream(flusso)
        grafo = torch.cuda.CUDAGraph()
        with torch.cuda.graph(grafo):
            out = self._decode(B)
        self.graphs[B] = (grafo, out)

    def _passo(self, B: int) -> torch.Tensor:
        if not self.use_graphs:
            return self._decode(B)
        if B not in self.graphs:
            self._capture(B)
        grafo, out = self.graphs[B]
        grafo.replay()
        return out.clone()  # il buffer di uscita del grafo viene riscritto al replay successivo

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, stop_id: int | None = None):
        """idx (B,P) senza padding; ritorna (token nuovi (B,K) con K <= max_new_tokens, statistiche)."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens deve essere almeno 1")
        B, P = idx.shape
        if P + max_new_tokens - 1 > self.model.ctx_len:
            raise ValueError(f"prompt di {P} token e {max_new_tokens} nuovi oltre ctx_len={self.model.ctx_len}")
        if B > self.max_batch:
            raise ValueError(f"batch di {B} righe oltre max_batch={self.max_batch}")
        device = self.device

        t0 = time.perf_counter()
        logits = self._prefill(idx)
        _sincronizza(device)
        t1 = time.perf_counter()
        nxt = logits.argmax(dim=-1)
        _sincronizza(device)
        t2 = time.perf_counter()

        self.tok[:B].copy_(nxt)
        tokens = [nxt]
        # flags[t]: tutte le righe hanno emesso stop_id entro il token t (monotono)
        flags = []
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if stop_id is not None:
            done = nxt == stop_id
            flags.append(done.all())

        passi = 0
        while len(tokens) < max_new_tokens:
            if stop_id is not None and passi % DONE_CHECK_EVERY == 0 and bool(flags[-1]):
                break
            self.posbuf.fill_(P + passi)
            nxt = self._passo(B)
            if stop_id is not None:
                # ogni riga che ha emesso stop_id ripete quello come padding, come kvgen
                nxt = torch.where(done, torch.full_like(nxt, stop_id), nxt)
                done = done | (nxt == stop_id)
                flags.append(done.all())
            self.tok[:B].copy_(nxt)
            tokens.append(nxt)
            passi += 1
        _sincronizza(device)
        t3 = time.perf_counter()

        # tronca al primo passo in cui tutte le righe sono finite, come kvgen
        K = len(tokens)
        if stop_id is not None:
            fatti = torch.stack(flags).tolist()
            if True in fatti:
                K = fatti.index(True) + 1
        new = torch.stack(tokens[:K], dim=1)
        stats = {"prefill_s": t1 - t0, "ttft_s": t2 - t0, "decode_s": t3 - t2, "n_steps": K - 1}
        return new, stats


def evaluate_fast(model: GPT, tok: Tokenizer, spec: CorpusSpec, persons: list[Person], eval_persons: int,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Come train._evaluate, con generazione statica bf16 e stop su eos."""
    device = next(model.parameters()).device
    eos = tok.token_to_id(EOS)
    casi = corpus.eval_cases(spec, persons, eval_persons)
    ids = [[eos] + tok.encode(c.prompt).ids for c in casi]

    # prompt della stessa lunghezza insieme: la generazione non usa padding
    per_lunghezza: dict[int, list[int]] = {}
    for i, riga in enumerate(ids):
        per_lunghezza.setdefault(len(riga), []).append(i)

    testi = [""] * len(casi)
    model.eval()
    # cache_enabled=False: i CUDA graph non devono riusare cast dei pesi catturati durante la capture
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False):
        max_batch = min(EVAL_BATCH, max((len(v) for v in per_lunghezza.values()), default=1))
        gen = StaticGen(model, max_batch=max_batch, dtype=torch.bfloat16)
        for indici in per_lunghezza.values():
            for k in range(0, len(indici), EVAL_BATCH):
                blocco = indici[k:k + EVAL_BATCH]
                idx = torch.tensor([ids[i] for i in blocco], dtype=torch.long, device=device)
                nuovi, _ = gen.generate(idx, max_new_tokens, stop_id=eos)
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
