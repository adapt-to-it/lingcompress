"""Fase 0 v3: profilo del training v2 (0,38M, E200, italiano_pieno) e della valutazione v2."""
import json
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from src.config import PAD
from src.v2 import corpus, train
from src.v2.model2 import GPT2
from src.v2.spec import BIOS_PER_STEP, CORPORA, SIZES_V2

corpus_name = sys.argv[1] if len(sys.argv) > 1 else "italiano_pieno"
spec, size = CORPORA[corpus_name], SIZES_V2["0.38M"]
persons = corpus.load_persons(20000)
tok = train.get_tokenizer(spec, persons)
pad = tok.token_to_id(PAD)
dev = torch.device("cuda")
ctx = max(train._max_len(spec, 20000, e) for e in range(201))
model = GPT2(tok.get_vocab_size(), ctx, size.n_layer, size.n_embd, size.n_head).to(dev)
optim = torch.optim.AdamW(model.parameters(), lr=2e-3, fused=True)
stream = train._bio_stream(spec, 20000, 0, 0, 20000)
t_data = t_step = 0.0


def step():
    global t_data, t_step
    t0 = time.perf_counter()
    bios = [next(stream) for _ in range(BIOS_PER_STEP)]
    T = max(len(b) for b in bios)
    righe = np.full((BIOS_PER_STEP, T), pad, dtype=np.int64)
    for i, b in enumerate(bios):
        righe[i, :len(b)] = b
    seq = torch.from_numpy(righe).to(dev)
    x, y = seq[:, :-1], seq[:, 1:].masked_fill(seq[:, 1:] == pad, -100)
    t1 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.loss(x, y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    t_data += t1 - t0
    t_step += time.perf_counter() - t1


for _ in range(100):
    step()
torch.cuda.synchronize()
N = 300
u0 = time.perf_counter()
for _ in range(N):
    step()
torch.cuda.synchronize()
unprof_ms = 1e3 * (time.perf_counter() - u0) / N
t_data = t_step = 0.0
w0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        step()
    torch.cuda.synchronize()
wall = time.perf_counter() - w0
ev = prof.key_averages()
from torch.autograd import DeviceType
kern = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
gpu_us = sum(e.time_range.elapsed_us() for e in kern)
n_kernels = len(kern)
#print(ev.table(sort_by="self_cpu_time_total", row_limit=15))
#print(ev.table(sort_by="self_cuda_time_total", row_limit=10))

# valutazione v2 completa (non profilata: tempo totale e quota GPU)
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p2:
    train._evaluate(model, tok, spec, persons, 5000)
    torch.cuda.synchronize()
eval_wall = time.perf_counter() - t0
eval_gpu = sum(e.time_range.elapsed_us() for e in p2.events() if e.device_type == DeviceType.CUDA)
out = dict(corpus=corpus_name, steps=N, unprofiled_ms_step=unprof_ms, wall_ms_step=1e3 * wall / N,
           data_ms_step=1e3 * t_data / N, launch_ms_step=1e3 * t_step / N,
           gpu_ms_step=gpu_us / 1e3 / N, kernels_per_step=n_kernels / N,
           eval_wall_s=eval_wall, eval_gpu_s=eval_gpu / 1e6)
nz = [e for e in ev if e.key == "aten::nonzero"]
out["nonzero_cpu_ms_step"] = sum(e.self_cpu_time_total for e in nz) / 1e3 / N
print(json.dumps(out, indent=1))
