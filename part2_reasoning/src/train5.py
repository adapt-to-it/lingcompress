"""SFT LoRA di v5: un modello base causale su un formato (A, B, C, D); loss solo su solution + EOS."""
import argparse
import contextlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.spec5 import DEV_SIZE, FORMATS, LORA, LR_DEFAULT, MICRO_BATCH, MODELS, PROMPT, QLORA, RUNS, SFT, TRAIN


def load_split(fmt):
    """(train_rows, dev_rows) da SFT/{fmt}_train.jsonl; il dev sono le ultime DEV_SIZE righe."""
    lines = (SFT / f"{fmt}_train.jsonl").read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line.strip()]
    cut = len(rows) - DEV_SIZE
    return rows[:cut], rows[cut:]


def encode(tok, row, max_len):
    """input_ids = prompt + solution + EOS; labels = -100 sul prompt, troncati a max_len."""
    prompt_ids = tok(PROMPT.format(question=row["question"]), add_special_tokens=False)["input_ids"]
    sol_ids = tok(row["solution"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + sol_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + sol_ids)[:max_len]
    return input_ids, labels


def run_id(model_key, fmt, lr, seed):
    return f"{model_key}_{fmt}_lr{lr:g}_s{seed}"


def _collate(encoded, pad_id):
    """Padding a destra: (input_ids, labels, attention_mask) come tensori (n, T)."""
    width = max(len(ids) for ids, _ in encoded)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, (a, b) in enumerate(encoded):
        ids[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        labels[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        mask[i, :len(a)] = 1
    return ids, labels, mask


def _n_targets(labels):
    """Token bersaglio predetti da una riga: le label dalla seconda in poi (shift come nel forward)."""
    return sum(1 for x in labels[1:] if x != -100)


def _batches(n, batch_size):
    """Indici dei passi: una permutazione per epoca, ogni epoca spezzata in blocchi da batch_size."""
    while True:
        perm = torch.randperm(n).tolist()
        for i in range(0, n, batch_size):
            yield perm[i:i + batch_size]


def train(model_key, fmt, lr, seed, rows=None, device="cuda", model=None, tok=None, out_dir=None, max_steps=None):
    """LoRA SFT su rows (default: train split del formato). Salva l'adapter e train.json; ritorna il log."""
    name = run_id(model_key, fmt, lr, seed)
    out_dir = Path(out_dir) if out_dir is not None else RUNS / name / "adapter"
    log_path = out_dir.parent / "train.json"
    if rows is None:
        rows = load_split(fmt)[0]

    if tok is None:
        tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa", **({"quantization_config": __import__("transformers").BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), "device_map": {"": 0}} if model_key in QLORA else {}))
    model.config.use_cache = False
    if model_key in QLORA:  # QLoRA: base in 4 bit, layernorm in fp32, input con gradiente
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = model.to(device)

    torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    if model_key not in QLORA:
        model = model.to(device)
    for p in model.parameters():  # adapter LoRA in fp32 (il modello base resta bf16), autocast bf16 su cuda
        if p.requires_grad:
            p.data = p.data.float()
    model.train()

    encoded = [encode(tok, row, TRAIN["max_len"]) for row in rows]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch_size = TRAIN["batch_size"]
    micro = MICRO_BATCH[model_key]
    steps = max_steps if max_steps is not None else TRAIN["epochs"] * math.ceil(len(encoded) / batch_size)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=TRAIN["weight_decay"])
    warm = int(steps * TRAIN["warmup_frac"])

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = device.startswith("cuda")

    losses = []
    start = time.perf_counter()
    for _, idx in zip(range(steps), _batches(len(encoded), batch_size)):
        n_tok = max(1, sum(_n_targets(encoded[i][1]) for i in idx))
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(0, len(idx), micro):
            ids, labels, mask = _collate([encoded[i] for i in idx[k:k + micro]], pad_id)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            target = labels[:, 1:].to(device)
            sel = target != -100
            with ctx:
                # logit solo sulle posizioni con target: con un vocabolario da 152k i logit pieni non stanno in memoria
                hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                               output_hidden_states=True, logits_to_keep=1).hidden_states[-1]
                logits = model.get_output_embeddings()(hidden[:, :-1][sel])
            part = F.cross_entropy(logits.float(), target[sel], reduction="sum")
            (part / n_tok).backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_(params, TRAIN["grad_clip"])
        optimizer.step()
        scheduler.step()
        losses.append(total / n_tok)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    result = {
        "run_id": name,
        "model": model_key,
        "format": fmt,
        "lr": lr,
        "seed": seed,
        "n_train": len(rows),
        "steps": steps,
        "train_time_s": train_time,
        "final_loss": statistics.mean(losses[-20:]),
        "losses": losses,
    }
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description="LoRA SFT di v5 su un formato")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fmt", required=True, choices=FORMATS)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    rows, _ = load_split(args.fmt)
    res = train(args.model, args.fmt, args.lr, args.seed, rows=rows, device="cuda", max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in res.items() if k != "losses"}))


if __name__ == "__main__":
    main()
