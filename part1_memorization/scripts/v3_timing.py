"""Fase D v3: scomposizione del tempo per step (0,38M) in head (softmax + cross-entropy) e corpo del modello.

head_eq_ms:   forward+backward del solo head (lm_head legato + CE) a parità di token (27 136, come pieno@bpe4096)
head_real_ms: lo stesso con i token reali per step del formato/vocabolario
body_ms:      forward+backward di embedding + blocchi + ln_f su (512, T) con T = lunghezza massima reale
"""
import time

import pandas as pd
import torch
import torch.nn.functional as F

from src.v2.model2 import GPT2
from src.v2.spec import SIZES_V2
from src.v3.spec3 import V3_RESULTS

torch.backends.cuda.matmul.allow_tf32 = True
z = SIZES_V2["0.38M"]
B, N_EQ = 512, 27136
probe = pd.read_csv(V3_RESULTS / "vocab_probe.csv")


def bench(f, n=150):
    for _ in range(40):
        f()
    tempi = []
    for _ in range(3):   # mediana di 3 ripetizioni: clock della GPU stabile
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n // 3):
            f()
        torch.cuda.synchronize()
        tempi.append(1e3 * (time.perf_counter() - t0) / (n // 3))
    return sorted(tempi)[1]


rows = []
for r in probe.itertuples():
    V, T = int(r.vocab_eff), int(r.max_len)
    n_real = int(B * (r.tok_bio + 1))
    m = GPT2(V, T, z.n_layer, z.n_embd, z.n_head).cuda()
    w = m.lm_head.weight

    def head_fn(h, y):
        return F.cross_entropy((h @ w.t()).float(), y)
    head_c = torch.compile(head_fn, dynamic=False)

    def head(n):
        h = torch.randn(n, z.n_embd, device="cuda", requires_grad=True)
        y = torch.randint(0, V, (n,), device="cuda")
        def f():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                head_c(h, y).backward()
        return f

    x = torch.randint(0, V, (B, T), device="cuda")

    def body_fn():
        hh = m.tok_emb(x) + m.pos_emb(torch.arange(T, device="cuda"))
        for blk in m.blocks:
            hh = blk(hh)
        return m.ln_f(hh).float().square().mean()
    body_c = torch.compile(body_fn, dynamic=False)

    def body():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            body_c().backward()

    rows.append(dict(format=r.format, vocab=("char" if r.vocab == "char" else f"bpe{r.vocab}"), V=V, T=T,
                     tokens_step=n_real, head_eq_ms=bench(head(N_EQ)), head_real_ms=bench(head(n_real)),
                     body_ms=bench(body)))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df["head_ms"] = df["head_real_ms"]
ref = df[df.vocab == "bpe4096"].set_index("format")
df["save_softmax_ms"] = df["format"].map(ref["head_eq_ms"]) - df["head_eq_ms"]
df["loss_length_ms"] = (df["body_ms"] - df["format"].map(ref["body_ms"])) + \
                       (df["head_real_ms"] - df["head_eq_ms"]) - (df["format"].map(ref["head_real_ms"] - ref["head_eq_ms"]))
df.to_csv(V3_RESULTS / "D_timing.csv", index=False)
print(df.round(2).to_string())
