"""Fase 0 v3: tempo GPU puro dello step 0,38M (forma fissa 512 x ctx) eager vs CUDA graph."""
import json
import sys
import time

import torch
import torch.nn.functional as F

from src.v2.model2 import GPT2
from src.v2.spec import SIZES_V2

size = SIZES_V2[sys.argv[1] if len(sys.argv) > 1 else "0.38M"]
B, T, V = 512, 66, 3814
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
model = GPT2(V, T, size.n_layer, size.n_embd, size.n_head).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(2e-3, device=dev), fused=True, capturable=True)
x = torch.randint(0, V, (B, T), device=dev)
y = x.clone()
y[:, 53:] = -100   # ~20% padding, come nei dati reali


def loss_full():
    pos = torch.arange(T, device=dev)
    h = model.tok_emb(x) + model.pos_emb(pos)
    for blk in model.blocks:
        h = blk(h)
    logits = model.lm_head(model.ln_f(h))
    return F.cross_entropy(logits.view(-1, V).float(), y.view(-1), ignore_index=-100)


def step(fn):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fn()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    opt.step()
    opt.zero_grad(set_to_none=False)
    return loss


def bench(f, n=200):
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n


yv = y.clone()
masked = lambda: model.loss(x, yv)          # v2: indicizzazione booleana (sync)
out = {"eager_v2_loss_ms": bench(lambda: step(masked)), "eager_full_loss_ms": bench(lambda: step(loss_full))}
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step(loss_full)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_loss = step(loss_full)
out["cuda_graph_ms"] = bench(g.replay)
comp = torch.compile(loss_full)
out["compiled_full_loss_ms"] = bench(lambda: step(comp))
comp2 = torch.compile(loss_full, mode="max-autotune-no-cudagraphs")
out["compiled_autotune_ms"] = bench(lambda: step(comp2))
out["tokens_step"] = int((y != -100).sum())
print(json.dumps(out, indent=1))
