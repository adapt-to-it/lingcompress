"""Scarica GSM8K (in data/gsm8k/*.jsonl) e i modelli usati da v5 nella cache HF."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.net  # noqa: F401,E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

MODELLI = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Instruct-2507"]
out = Path(__file__).resolve().parents[1] / "data" / "gsm8k"
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(ds):
            f.write(json.dumps({"id": f"{split}-{i}", **r}, ensure_ascii=False) + "\n")
    print(split, len(ds), flush=True)
for m in MODELLI:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
    print(m, p, flush=True)
