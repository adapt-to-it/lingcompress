"""Valutazione di v5: generazione greedy in batch, metriche di accuratezza e latenza per problema."""
import json
import random
import statistics
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.formats import conforms, extract_answer, same_number
from src.spec5 import EVAL_BATCH, LATENCY_N, LATENCY_SEED, MAX_NEW_TOKENS, MODELS, OUTPUT_FIELDS, QLORA, PROMPT


def load_model(model_key, adapter_dir=None, device="cuda"):
    """Modello base bf16 con sdpa; se adapter_dir, adapter LoRA fuso nei pesi. Tokenizer con padding a sinistra."""
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model_key_ = model_key
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 **({"quantization_config": __import__("transformers").BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), "device_map": {"": 0}} if model_key_ in QLORA else {}))
    if model_key in QLORA:  # 4 bit: adapter non fuso
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        return model.eval(), tok
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model = model.to(device).eval()
    return model, tok


def _now(model):
    """Istante di orologio; con CUDA attende la fine dei kernel in coda, cosi' il tempo e' quello reale."""
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return time.perf_counter()


def generate(model, tok, questions, fmt, batch):
    """Risposte greedy al prompt PROMPT, nell'ordine dato. Ritorna [(testo, n_tokens)]; n_tokens include l'EOS."""
    prompts = [PROMPT.format(question=q) for q in questions]
    eos = tok.eos_token_id
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, pad_to_multiple_of=16).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS[fmt],
                cache_implementation="static",
                disable_compile=True,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        for ids in new:
            n = ids.index(eos) + 1 if eos in ids else len(ids)
            text = tok.decode(ids[:n], skip_special_tokens=True).strip()
            out.append((text, n))
    return out


def evaluate(model, tok, fmt, rows, batch=EVAL_BATCH, out_path=None):
    """Metriche sul set rows ({id, question, answer}); se out_path, scrive una riga jsonl per problema."""
    if not rows:
        raise ValueError("nessuna riga da valutare")
    questions = [r["question"] for r in rows]
    start = _now(model)
    results = generate(model, tok, questions, fmt, batch)
    elapsed = _now(model) - start

    records = []
    for row, (text, n) in zip(rows, results):
        pred = extract_answer(fmt, text)
        records.append({
            "id": row["id"],
            "output": text,
            "n_tokens": n,
            "pred": pred,
            "gold": row["answer"],
            "correct": pred is not None and same_number(pred, row["answer"]),
            "conform": conforms(fmt, text),
        })

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps({k: rec[k] for k in OUTPUT_FIELDS}, ensure_ascii=False) + "\n")

    n_eval = len(records)
    tokens = [rec["n_tokens"] for rec in records]
    return {
        "n_eval": n_eval,
        "accuracy": sum(rec["correct"] for rec in records) / n_eval,
        "conform_rate": sum(rec["conform"] for rec in records) / n_eval,
        "tok_mean": statistics.mean(tokens),
        "tok_median": statistics.median(tokens),
        "throughput_time_s": elapsed,
        "answers_per_s": n_eval / elapsed,
        "tokens_per_s": sum(tokens) / elapsed,
    }


def latency(model, tok, fmt, rows):
    """Latenza per problema, batch 1, su un campione fisso di LATENCY_N righe (seme LATENCY_SEED)."""
    if not rows:
        raise ValueError("nessuna riga per la latenza")
    ordered = sorted(rows, key=lambda r: r["id"])
    rng = random.Random(LATENCY_SEED)
    sample = rng.sample(ordered, min(LATENCY_N, len(ordered)))

    generate(model, tok, [sample[0]["question"]], fmt, 1)    # warm-up, non misurato
    items = []
    for row in sample:
        start = _now(model)
        ((_, n),) = generate(model, tok, [row["question"]], fmt, 1)
        seconds = _now(model) - start
        items.append({"id": row["id"], "seconds": seconds, "n_tokens": n})

    seconds = [it["seconds"] for it in items]
    tokens = [it["n_tokens"] for it in items]
    summary = {
        "n": len(items),
        "lat_mean_s": statistics.mean(seconds),
        "lat_median_s": statistics.median(seconds),
        "ms_per_token_mean": statistics.mean(it["seconds"] / it["n_tokens"] * 1000 for it in items),
        "tok_mean": statistics.mean(tokens),
    }
    return summary, items
