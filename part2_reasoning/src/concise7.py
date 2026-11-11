"""v7: riscrittura delle soluzioni A in prosa concisa (A1, A2) con il generatore locale; verifica e selezione."""
import json
from pathlib import Path

from src.formats import conforms, extract_answer, normalize, official_answer, official_steps, same_number
from src.rewrite import _numbers
from src.spec7 import FEWSHOT7, FEWSHOT_IDS7, GEN_SYSTEM7, GEN_USER7, MAX_RATIO

RETRY_MSG = ("Too long: {n} tokens, the limit is {m}. Rewrite it shorter: fewer words, the same computations and "
             "relation words, the same last line.")


def messages7(fmt, row, by_a, prev=None):
    """Chat di riscrittura di una soluzione A; prev = (testo, n_token, limite) aggiunge la richiesta di accorciare."""
    msgs = [{"role": "system", "content": GEN_SYSTEM7[fmt]}]
    for fid in FEWSHOT_IDS7:
        shot = by_a[fid]
        msgs.append({"role": "user", "content": GEN_USER7.format(question=shot["question"], solution=shot["solution"])})
        msgs.append({"role": "assistant", "content": FEWSHOT7[fmt][fid]})
    msgs.append({"role": "user", "content": GEN_USER7.format(question=row["question"], solution=row["solution"])})
    if prev is not None:
        msgs.append({"role": "assistant", "content": prev[0]})
        msgs.append({"role": "user", "content": RETRY_MSG.format(n=prev[1], m=prev[2])})
    return msgs


def check7(fmt, row, official, text, ntok):
    """(ok, motivo, rapporto, n_token): formato A, risposta e passaggi ufficiali, poi lunghezza <= MAX_RATIO * A."""
    n, n_a = ntok(text), ntok(row["solution"])
    ratio = n / n_a
    if not conforms("A", text):
        return False, "format", ratio, n
    if extract_answer("A", text) != official_answer(official):
        return False, "answer", ratio, n
    nums = _numbers(text)
    if not all(any(same_number(s, x) for x in nums) for s in official_steps(official)):
        return False, "steps", ratio, n
    if ratio > MAX_RATIO[fmt]:
        return False, "length", ratio, n
    return True, "", ratio, n


def load_attempts(path):
    """Tutte le righe del file di riscritture, per id e in ordine di tentativo."""
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["id"], []).append(r)
    return out


def run7(fmt, rows, official, gen_fn, out_path, ntok, attempts=3, batch=480):
    """Genera e verifica (riprendibile). Tentativo 1 a T=0; i successivi a T=0.7, con la richiesta di accorciare se
    l'ultimo era solo troppo lungo. official: id -> soluzione ufficiale GSM8K."""
    by_a = {r["id"]: r for r in rows}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_attempts(out_path)
    for attempt in range(1, attempts + 1):
        todo = [r for r in rows if not any(x["ok"] for x in done.get(r["id"], []))
                and len(done.get(r["id"], [])) < attempt]
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            msgs = []
            for r in chunk:
                last = done.get(r["id"], [None])[-1]
                prev = None
                if last is not None and last["reason"] == "length":
                    prev = (last["text"], last["n_tok"], int(MAX_RATIO[fmt] * ntok(r["solution"])))
                msgs.append(messages7(fmt, r, by_a, prev))
            texts = gen_fn(msgs, 0 if attempt == 1 else 0.7)
            with out_path.open("a", encoding="utf-8") as f:
                for r, text in zip(chunk, texts):
                    text = normalize("A", text)
                    ok, reason, ratio, n = check7(fmt, r, official[r["id"]], text, ntok)
                    row = {"id": r["id"], "text": text, "ok": ok, "reason": reason, "attempt": attempt,
                           "ratio": ratio, "n_tok": n}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.setdefault(r["id"], []).append(row)
    return done


def select7(attempts_by_id):
    """id -> (testo, stretto): il primo tentativo valido; altrimenti il piu' corto bocciato solo per lunghezza."""
    out = {}
    for i, rs in attempts_by_id.items():
        ok = [r for r in rs if r["ok"]]
        if ok:
            out[i] = (ok[0]["text"], True)
            continue
        soft = [r for r in rs if r["reason"] == "length"]
        if soft:
            out[i] = (min(soft, key=lambda r: r["n_tok"])["text"], False)
    return out
