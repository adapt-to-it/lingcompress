"""v7 data generation: concise rewrites A1 / A2 of the A solutions, up to 3 attempts each.

Needs a running llama-server (scripts/llama_server.sh). Run from part2_reasoning/:
    python scripts/gen7.py [A1] [A2]
Writes data/rewrites/A1.jsonl and A2.jsonl (resumable). About 34 min (A1) and 43 min (A2) on an RTX 4070 SUPER.
"""
import json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from src.concise7 import run7
from src.rewrite import llama_generator, load_gsm8k
from src.spec7 import FORMATS7, REWRITE7

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')
cache = {}
def ntok(s):
    if s not in cache:
        cache[s] = len(tok(s).input_ids)
    return cache[s]
rows = [json.loads(l) for l in open('data/sft/A_train.jsonl', encoding='utf-8') if l.strip()]
official = {x['id']: x['answer'] for x in load_gsm8k('train')}
gen = llama_generator()
for fmt in (sys.argv[1:] or FORMATS7):
    t = time.time()
    done = run7(fmt, rows, official, gen, REWRITE7[fmt], ntok)
    c = Counter(rs[-1]["reason"] for rs in done.values())
    print(fmt, f"{time.time() - t:.0f}s", dict(c), flush=True)
