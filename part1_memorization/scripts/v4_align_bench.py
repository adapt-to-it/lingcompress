"""Fase 0bis v4: costo dello step di training (0,38M, pieno) con V allineato a 64 o no.

Step completo come train3.run_job: loss compilata (forward), backward, clip, AdamW fused; batch 512 x T reali
del pieno (T = max_len - 1 di vocab_probe). head_ms: solo lm_head + cross-entropy (27 136 token, come D.5).
"""
import time

import pandas as pd
import torch
import torch.nn.functional as F

from src.v2.model2 import GPT2
from src.v2.spec import SIZES_V2
from src.v3.spec3 import V3_RESULTS
from src.v3.train3 import _loss_batch
from src.v4.spec4 import V4_RESULTS

torch.backends.cuda.matmul.allow_tf32 = True
z = SIZES_V2["0.38M"]
B, N_HEAD = 512, 27136
probe = pd.read_csv(V3_RESULTS / "vocab_probe.csv").query("format == 'italiano_pieno'").set_index("vocab")
# (vocabolario, V, allineato): 4096 reale = 3814; 2048 e' gia' multiplo di 64, i non allineati sono artificiali
CASI = [("bpe4096", int(probe.loc["4096", "vocab_eff"]), False), ("bpe4096", 3816, False), ("bpe4096", 3840, True),
        ("bpe2048", 2047, False), ("bpe2048", 2040, False), ("bpe2048", 2048, True)]


def bench(f, n=150):
    for _ in range(40):
        f()
    tempi = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n // 3):
            f()
        torch.cuda.synchronize()
        tempi.append(1e3 * (time.perf_counter() - t0) / (n // 3))
    return sorted(tempi)[1]


rows = []
for vocab, V, allineato in CASI:
    T = int(probe.loc[vocab[3:], "max_len"])
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = GPT2(V, T, z.n_layer, z.n_embd, z.n_head).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    loss_fn = torch.compile(_loss_batch, dynamic=False)
    seq = torch.randint(0, min(V, 2000), (B, T), device="cuda")
    x, y = seq[:, :-1], seq[:, 1:]

    def step():
        loss = loss_fn(m, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    w = m.lm_head.weight
    h = torch.randn(N_HEAD, z.n_embd, device="cuda", requires_grad=True)
    yy = torch.randint(0, V, (N_HEAD,), device="cuda")
    head_c = torch.compile(lambda: F.cross_entropy((h @ w.t()).float(), yy), dynamic=False)

    def head():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            head_c().backward()

    rows.append(dict(vocab=vocab, V=V, aligned=allineato, V_mod64=V % 64, T=T,
                     step_ms=bench(step), head_ms=bench(head)))
    print(rows[-1], flush=True)

V4_RESULTS.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(V4_RESULTS / "align_bench.csv", index=False)
print(df.round(2).to_string())
