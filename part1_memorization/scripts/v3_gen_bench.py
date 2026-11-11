"""Fase 0 v3: latenza di generazione v2 (kvgen) vs v3 (StaticGen + CUDA graph), pesi casuali."""
import time, torch, pandas as pd
from src.v2 import kvgen
from src.v2.model2 import GPT2
from src.v2.spec import SIZES_V2
from src.v3.fastgen import StaticGen
from src.v3.spec3 import V3_RESULTS
torch.backends.cuda.matmul.allow_tf32 = True
P, NEW, REP = 6, 4, 200
rows = []
for s in ("0.60M", "1.49M", "4M"):
    z = SIZES_V2[s]
    m = GPT2(3814, 66, z.n_layer, z.n_embd, z.n_head).cuda().eval()
    sg = StaticGen(m, 64)
    for B in (1, 64):
        idx = torch.randint(0, 3814, (B, P), device="cuda")
        for name, f in (("v2_kvgen", lambda: kvgen.generate(m, idx, NEW)), ("v3_static_graph", lambda: sg.generate(idx, NEW))):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                for _ in range(20): f()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(REP): _, st = f()
                torch.cuda.synchronize()
            ms = 1e3 * (time.perf_counter() - t0) / REP
            rows.append(dict(size=s, batch=B, impl=name, latency_ms=ms, ms_per_new_token=ms / NEW))
df = pd.DataFrame(rows)
piv = df.pivot_table(index=["size", "batch"], columns="impl", values="latency_ms")
piv["speedup"] = piv["v2_kvgen"] / piv["v3_static_graph"]
print(piv.round(3).to_string())
V3_RESULTS.mkdir(parents=True, exist_ok=True); piv.reset_index().to_csv(V3_RESULTS / "phase0_generation.csv", index=False)
