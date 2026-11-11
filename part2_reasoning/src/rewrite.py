"""Riscrittura locale delle soluzioni GSM8K nei formati A e C, verifica, file di SFT e CLI."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.formats import (normalize, 
    check_c,
    conforms,
    extract_answer,
    norm_num,
    official_answer,
    official_steps,
    render_d,
    same_number,
    strip_b,
)
from src.spec5 import FEWSHOT, FEWSHOT_IDS, FORMATS, GEN_SYSTEM, GEN_USER, GENERATOR, GSM8K, REWRITES, SFT

# Numero nel testo per il controllo dei passaggi: il '-' attaccato a una cifra precedente non fa parte del numero.
_NUM_RE = re.compile(r"(?<![\w.])-?[0-9][0-9,]*(?:\.[0-9]+)?")
# Prompt per ogni chiamata di generate: ordinati per lunghezza, il padding resta piccolo.
_SUB_BATCH = int(__import__("os").environ.get("V5_GEN_BATCH", "10"))


def load_gsm8k(split):
    """Righe di GSM8K/{split}.jsonl: {"id", "question", "answer"}."""
    with (GSM8K / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(fmt, item, by_id):
    """Chat di riscrittura: system, esempi few-shot (domanda, soluzione ufficiale, testo atteso), poi l'item."""
    if fmt == "C6":
        from src.spec6 import FEWSHOT_C6, FEWSHOT_IDS_C6, GEN_SYSTEM_C6
        system, ids, shots = GEN_SYSTEM_C6, FEWSHOT_IDS_C6, FEWSHOT_C6
    else:
        system, ids, shots = GEN_SYSTEM[fmt], FEWSHOT_IDS, FEWSHOT[fmt]
    msgs = [{"role": "system", "content": system}]
    for fid in ids:
        shot = by_id[fid]
        msgs.append({"role": "user", "content": GEN_USER.format(question=shot["question"], answer=shot["answer"])})
        msgs.append({"role": "assistant", "content": shots[fid]})
    msgs.append({"role": "user", "content": GEN_USER.format(question=item["question"], answer=item["answer"])})
    return msgs


def _numbers(text):
    """Numeri del testo normalizzati."""
    return [n for n in (norm_num(m) for m in _NUM_RE.findall(text)) if n is not None]


def validate(fmt, item, text, require_steps=True):
    """Verifica una riscrittura: formato, aritmetica (solo C), risposta finale, passaggi ufficiali.
    Ritorna (ok, motivo) con motivo "" se ok; "format", "arith", "answer" o "steps" altrimenti."""
    if fmt == "A":
        if not conforms("A", text):
            return False, "format"
    elif fmt == "C6":
        from src.grammar6 import check_c6
        if not check_c6(text, arithmetic=False)[0]:
            return False, "format"
        if not check_c6(text)[0]:
            return False, "arith"
    elif fmt == "C":
        if not check_c(text, arithmetic=False)[0]:
            return False, "format"
        if not check_c(text)[0]:
            return False, "arith"
    else:
        raise ValueError(f"formato non riscrivibile: {fmt}")
    if extract_answer("C" if fmt == "C6" else fmt, text) != official_answer(item["answer"]):
        return False, "answer"
    if require_steps:
        nums = _numbers(text)
        for step in official_steps(item["answer"]):
            if not any(same_number(step, n) for n in nums):
                return False, "steps"
    return True, ""


def _load_rows(path):
    """Ultima riga per id di un file di riscritture (vuoto se il file non esiste)."""
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["id"]] = row
    return rows


def _pending(row, attempt):
    """L'item va generato al tentativo dato: nessuna riga, oppure ultima riga non ok e tentativo non ancora fatto."""
    return row is None or (not row["ok"] and row["attempt"] < attempt)


def run(fmt, items, gen_fn, out_path, attempts=3, batch=64, require_steps=True):
    """Genera e valida le riscritture, una riga JSON per tentativo in out_path; ripartibile.
    Tentativo 1 a temperatura 0, i successivi a 0.7 e solo per gli item non ancora ok.
    Ritorna il conteggio per motivo finale ("" = ok)."""
    by_id = {x["id"]: x for x in load_gsm8k("train")}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last = _load_rows(out_path)
    for attempt in range(1, attempts + 1):
        temperature = 0 if attempt == 1 else 0.7
        pending = [it for it in items if _pending(last.get(it["id"]), attempt)]
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            texts = gen_fn([build_messages(fmt, it, by_id) for it in chunk], temperature)
            if len(texts) != len(chunk):
                raise ValueError(f"gen_fn ha restituito {len(texts)} testi per {len(chunk)} richieste")
            with out_path.open("a", encoding="utf-8") as f:
                for it, text in zip(chunk, texts):
                    if fmt == "C6":
                        from src.grammar6 import normalize_c6
                        text = normalize_c6(text)
                    else:
                        text = normalize(fmt, text)
                    ok, reason = validate(fmt, it, text, require_steps)
                    row = {"id": it["id"], "text": text, "ok": ok, "reason": reason, "attempt": attempt}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    last[it["id"]] = row
    counts = Counter(last[it["id"]]["reason"] if it["id"] in last else "none" for it in items)
    return dict(counts)


def hf_generator(model_id, max_new_tokens=320):
    """gen_fn(messages_list, temperature) -> testi nell'ordine d'ingresso, con transformers su CUDA."""
    import src.net  # noqa: F401  IPv4 prima della prima connessione a Hugging Face
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()

    def gen_fn(messages_list, temperature):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [""] * len(prompts)
        if temperature == 0:
            sampling = {"do_sample": False}
        else:
            sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95}
        for s in range(0, len(order), _SUB_BATCH):
            idx = order[s:s + _SUB_BATCH]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **sampling)
            new = ids[:, enc["input_ids"].shape[1]:]
            for i, text in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
                out[i] = text.strip()
        return out

    return gen_fn


def llama_generator(url="http://127.0.0.1:8090", concurrency=24, max_new_tokens=320):
    """gen_fn che usa un llama-server locale (/v1/chat/completions) con richieste concorrenti."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def one(messages, temperature):
        body = {"messages": messages, "max_tokens": max_new_tokens, "temperature": temperature, "seed": 0,
                "cache_prompt": True}
        if temperature > 0:
            body["top_p"] = 0.95
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def gen_fn(messages_list, temperature):
        with ThreadPoolExecutor(concurrency) as ex:
            return list(ex.map(lambda m: one(m, temperature), messages_list))

    return gen_fn


def _is_ok(rows, item_id):
    return item_id in rows and rows[item_id]["ok"]


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft():
    """File SFT: train con id ok sia in A sia in C (ultima riga per id); test con solution vuota."""
    a_rows = _load_rows(REWRITES / "A.jsonl")
    c_rows = _load_rows(REWRITES / "C.jsonl")
    keep = [it for it in load_gsm8k("train") if _is_ok(a_rows, it["id"]) and _is_ok(c_rows, it["id"])]
    SFT.mkdir(parents=True, exist_ok=True)
    out = {fmt: [] for fmt in FORMATS}
    for it in keep:
        ans = official_answer(it["answer"])
        head = {"id": it["id"], "question": it["question"]}
        a = a_rows[it["id"]]["text"]
        out["A"].append({**head, "solution": a, "answer": ans})
        out["B"].append({**head, "solution": strip_b(a), "answer": ans})
        out["C"].append({**head, "solution": c_rows[it["id"]]["text"], "answer": ans})
        out["D"].append({**head, "solution": render_d(ans), "answer": ans})
    counts = {}
    for fmt in FORMATS:
        _write_jsonl(SFT / f"{fmt}_train.jsonl", out[fmt])
        counts[fmt] = len(out[fmt])
    test = [{"id": it["id"], "question": it["question"], "solution": "", "answer": official_answer(it["answer"])}
            for it in load_gsm8k("test")]
    _write_jsonl(SFT / "test.jsonl", test)
    counts["test"] = len(test)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.rewrite")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="genera e valida le riscritture A o C")
    g.add_argument("--fmt", choices=["A", "C", "C6"], required=True)
    g.add_argument("--ids-from", help="jsonl con campo id: genera solo quegli id (C6: i problemi di v5)")
    g.add_argument("--split", default="train")
    g.add_argument("--limit", type=int)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--no-steps", action="store_true", help="non richiede i passaggi ufficiali tra i numeri")
    g.add_argument("--engine", choices=["llama", "hf"], default="llama", help="llama-server locale o transformers")
    sub.add_parser("build", help="scrive i file SFT")
    args = p.parse_args(argv)
    if args.cmd == "gen":
        items = load_gsm8k(args.split)
        if args.ids_from:
            keep = {json.loads(line)["id"] for line in open(args.ids_from, encoding="utf-8") if line.strip()}
            items = [it for it in items if it["id"] in keep]
        if args.limit is not None:
            items = items[:args.limit]
        counts = run(args.fmt, items, llama_generator() if args.engine == "llama" else hf_generator(GENERATOR), REWRITES / f"{args.fmt}.jsonl",
                     batch=args.batch, require_steps=not args.no_steps)
    else:
        counts = build_sft()
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
