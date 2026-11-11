"""Valutazione di v6: generazione greedy come eval5, con logits processor opzionale e conteggio dei token forzati."""
import json
import random
import statistics
from pathlib import Path

import torch
from transformers import LogitsProcessorList

from src.eval5 import _now, load_model
from src.formats import conforms, extract_answer, same_number
from src.grammar6 import check_c6, stats_c6
from src.spec5 import EVAL_BATCH, LATENCY_N, LATENCY_SEED, MAX_NEW_TOKENS, PROMPT
from src.spec6 import OUTPUT6_FIELDS

# Formato di v6 -> formato di formats.py per risposta e conformita' (C6 usa il calcolo di grammar6).
ANSWER_FMT = {"A": "A", "A1": "A", "A2": "A", "C5": "C", "C6": "C"}


def _check_fmt(fmt):
    if fmt not in ANSWER_FMT:
        raise ValueError(f"formato sconosciuto: {fmt}")


def _max_new(fmt):
    return MAX_NEW_TOKENS["A"] if ANSWER_FMT[fmt] == "A" else MAX_NEW_TOKENS["C"]


def _pred(fmt, text):
    return extract_answer(ANSWER_FMT[fmt], text)


def _conform(fmt, text):
    if fmt == "C6":
        return check_c6(text, arithmetic=False)[0]
    return conforms(ANSWER_FMT[fmt], text)


def generate6(model, tok, questions, fmt, batch, processor_factory=None):
    """Risposte greedy al prompt PROMPT, nell'ordine dato. Ritorna [(testo, n_tokens, n_forced)].

    processor_factory(lunghezza_prompt, n_righe) costruisce un logits processor per ogni batch; n_forced e' il
    numero di token forzati dal processor per quella risposta (0 senza processor)."""
    _check_fmt(fmt)
    prompts = [PROMPT.format(question=q) for q in questions]
    eos = tok.eos_token_id
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, pad_to_multiple_of=16).to(model.device)
        kwargs = {}
        proc = None
        if processor_factory is not None:
            proc = processor_factory(enc["input_ids"].shape[1], enc["input_ids"].shape[0])
            kwargs["logits_processor"] = LogitsProcessorList([proc])
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=_max_new(fmt),
                cache_implementation="static",
                disable_compile=True,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
                **kwargs,
            )
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        for j, ids in enumerate(new):
            n = ids.index(eos) + 1 if eos in ids else len(ids)
            text = tok.decode(ids[:n], skip_special_tokens=True).strip()
            n_forced = proc.n_forced[j] if proc is not None else 0
            out.append((text, n, n_forced))
    return out


def evaluate6(model, tok, fmt, rows, batch=EVAL_BATCH, processor_factory=None, out_path=None):
    """Metriche sul set rows ({id, question, answer}); se out_path, scrive una riga jsonl per problema."""
    if not rows:
        raise ValueError("nessuna riga da valutare")
    questions = [r["question"] for r in rows]
    start = _now(model)
    results = generate6(model, tok, questions, fmt, batch, processor_factory)
    elapsed = _now(model) - start

    records = []
    for row, (text, n, n_forced) in zip(rows, results):
        pred = _pred(fmt, text)
        records.append({
            "id": row["id"],
            "output": text,
            "n_tokens": n,
            "n_forced": n_forced,
            "pred": pred,
            "gold": row["answer"],
            "correct": pred is not None and same_number(pred, row["answer"]),
            "conform": _conform(fmt, text),
        })

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps({k: rec[k] for k in OUTPUT6_FIELDS}, ensure_ascii=False) + "\n")

    n_eval = len(records)
    tokens = [rec["n_tokens"] for rec in records]
    if fmt == "C6":
        stats = [stats_c6(rec["output"]) for rec in records]
        note_rate = sum(s["n_note"] >= 1 for s in stats) / n_eval
        solve_rate = sum(s["n_solve"] >= 1 for s in stats) / n_eval
    else:
        note_rate = 0.0
        solve_rate = 0.0
    return {
        "n_eval": n_eval,
        "accuracy": sum(rec["correct"] for rec in records) / n_eval,
        "conform_rate": sum(rec["conform"] for rec in records) / n_eval,
        "tok_mean": statistics.mean(tokens),
        "tok_median": statistics.median(tokens),
        "forced_mean": statistics.mean(rec["n_forced"] for rec in records),
        "throughput_time_s": elapsed,
        "answers_per_s": n_eval / elapsed,
        "tokens_per_s": sum(tokens) / elapsed,
        "note_rate": note_rate,
        "solve_rate": solve_rate,
    }


def latency6(model, tok, fmt, rows, processor_factory=None):
    """Latenza per problema, batch 1, su un campione fisso di LATENCY_N righe (seme LATENCY_SEED)."""
    if not rows:
        raise ValueError("nessuna riga per la latenza")
    ordered = sorted(rows, key=lambda r: r["id"])
    rng = random.Random(LATENCY_SEED)
    sample = rng.sample(ordered, min(LATENCY_N, len(ordered)))

    generate6(model, tok, [sample[0]["question"]], fmt, 1, processor_factory)    # warm-up, non misurato
    items = []
    for row in sample:
        start = _now(model)
        ((_, n, _),) = generate6(model, tok, [row["question"]], fmt, 1, processor_factory)
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
