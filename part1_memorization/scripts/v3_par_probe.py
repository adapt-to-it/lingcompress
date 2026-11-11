"""Fase 0 v3: step compilato (0,38M) per misurare il throughput con N processi in parallelo."""
import sys, time, torch
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "0.38M"]
import importlib.util
src = open("scripts/v3_graph_probe.py", encoding="utf-8").read().split("yv = y.clone()")[0]
exec(src)
comp = torch.compile(loss_full)
for _ in range(30): step(comp)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(400): step(comp)
torch.cuda.synchronize(); print("ms_step", 1e3 * (time.perf_counter() - t0) / 400, flush=True)
