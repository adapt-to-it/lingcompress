import sys, time, torch, torch.nn.functional as F
from src.v2.model2 import GPT2
from src.v2.spec import SIZES_V2
s, T, V, used = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
z = SIZES_V2[s]; B = 512; dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
m = GPT2(V, T, z.n_layer, z.n_embd, z.n_head).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
x = torch.randint(0, V, (B, T), device=dev); y = x.clone(); y[:, int(used):] = -100
def lf():
    h = m.tok_emb(x) + m.pos_emb(torch.arange(T, device=dev))
    for b in m.blocks: h = b(h)
    return F.cross_entropy(m.lm_head(m.ln_f(h)).view(-1, V).float(), y.view(-1), ignore_index=-100)
c = torch.compile(lf, dynamic=False)
def st():
    with torch.autocast("cuda", dtype=torch.bfloat16): l = c()
    l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
for _ in range(20): st()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): st()
torch.cuda.synchronize(); ms = 1e3 * (time.perf_counter() - t0) / 100
print(s, T, V, "ms_step %.1f" % ms, "Mtok_s %.2f" % (B * used / ms / 1e3), "vram %d" % (torch.cuda.max_memory_allocated() >> 20))
