"""Pipeline v3 misurata: stripper, core e realizzatore a regole, con tempi, FLOPs e accuratezza per formato."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

from src.config import ATTRS, EOS
from src.data import Person
from src.v2 import corpus, kvgen
from src.v2.infer import answer_tokens
from src.v2.model2 import GPT2
from src.v2.spec import CORPORA, SIZES_V2, V2_RESULTS, V2_RUNS, CorpusSpec, Job
from src.v2.train import get_tokenizer
from src.v3.fastgen import StaticGen
from src.v3.realize import InterRealizer, TeleRealizer, question
from src.v3.spec3 import (E_FORMATS, PIPE_BATCHES, PIPE_COLUMNS, PIPE_QUESTIONS, PIPE_SIZES, V3_RESULTS,
                          V3_RUNS)
from src.v3.strip import ListStripper, SpacyStripper, _Stripper, register

WARMUP_CALLS = 20      # chiamate di riscaldamento non cronometrate
LOTTO_MAX = 2048       # righe per chiamata di generazione in evaluate_stripped
BASE_PIENO = "italiano_pieno"
FORMATI = ("telegrafico_auto", "telegrafico_lista", "interlingua")
STRIPPER_DI = {"telegrafico_auto": SpacyStripper, "telegrafico_lista": ListStripper}
_CASI: dict[tuple[str, int], list[tuple]] = {}
for _nome in E_FORMATS:
    # i corpora telegrafici si registrano in CORPORA all'import, se non ci sono gia'
    if _nome not in CORPORA:
        register(_nome)


@dataclass
class _Voce:
    attr: str
    domanda: str
    canonico: str
    core: str       # prompt del core (non cronometrato, usato per raggruppare per lunghezza)
    atteso: str     # telegrafico: token attesi dopo la domanda; interlingua: superficie della risposta


def _sincronizza(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _canonico(person: Person, attr: str) -> str:
    """Valore canonico dell'attributo della persona."""
    return person.attrs[attr]


def _domande(persons: list[Person], n_persons: int) -> list[tuple[int, str, str, str]]:
    """(persona, attributo, domanda, canonico) per le persone 0..n-1 e ogni attributo."""
    return [(i, attr, question(persons[i], attr), _canonico(persons[i], attr))
            for i in range(n_persons) for attr in ATTRS]


def _dopo(stripper: _Stripper, prima: str, coda: str) -> str:
    """Token di `coda` che seguono `prima` in stripper.spans; il punto finale va via."""
    inizio = len(prima) + 1
    tokens = [tok for tok, start, _ in stripper.spans(f"{prima} {coda}") if start >= inizio]
    if tokens and tokens[-1] == ".":
        tokens.pop()
    return " ".join(tokens)


def cases(stripper: _Stripper, persons: list[Person], n_persons: int) -> list[tuple]:
    """(persona, attributo, domanda, prompt core, atteso, canonico) per le persone 0..n-1 e ogni attributo."""
    chiave = (stripper.name, n_persons)
    if chiave not in _CASI:
        _CASI[chiave] = [
            (i, attr, domanda, stripper.strip(domanda), _dopo(stripper, domanda, f"{canonico}."), canonico)
            for i, attr, domanda, canonico in _domande(persons, n_persons)]
    return _CASI[chiave]


def _gruppi(batch: int, lunghezze: list[int]) -> list[list[int]]:
    """Indici raggruppati per lunghezza del prompt (senza padding), a blocchi di batch."""
    per_lunghezza: dict[int, list[int]] = {}
    for i, n in enumerate(lunghezze):
        per_lunghezza.setdefault(n, []).append(i)
    return [indici[s:s + batch] for indici in per_lunghezza.values()
            for s in range(0, len(indici), batch)]


def _genera(gen: StaticGen, tok: Tokenizer, prompt: list[str], max_new: int,
            stop_id: int | None) -> list[list[int]]:
    """Tokenizza (eos + prompt) e genera greedy in bf16 autocast; righe di uguale lunghezza."""
    eos = tok.token_to_id(EOS)
    idx = torch.tensor([[eos] + tok.encode(p).ids for p in prompt], dtype=torch.long, device=gen.device)
    # cache_enabled=False: i CUDA graph non devono riusare cast dei pesi catturati durante la capture
    with torch.autocast(device_type=gen.device.type, dtype=torch.bfloat16, cache_enabled=False):
        nuovi, _ = gen.generate(idx, max_new, stop_id=stop_id)
    return nuovi.tolist()


def _flops(model: GPT2, lunghezze: list[int], nuovi: list[int]) -> float:
    """FLOPs (prefill + decode) sui valori medi di lunghezza del prompt e della risposta."""
    n_total = model.count_params()[1]
    pre, dec = kvgen.flops(n_total, len(model.blocks), model.tok_emb.embedding_dim,
                           int(round(float(np.mean(lunghezze)))), int(round(float(np.mean(nuovi)))))
    return float(pre + dec)


def _misura_core(stripper: _Stripper | None, inter: InterRealizer, model: GPT2, tok: Tokenizer,
                 voci: list[_Voce], batch: int) -> dict:
    """Strip (o prompt interlingua), core e realizzazione per batch: tempi in ms, esiti per domanda."""
    tele = stripper is not None
    gen = StaticGen(model, max_batch=batch)
    device = gen.device
    punto = tok.token_to_id(".")
    realizzatore = TeleRealizer(stripper) if tele else inter
    lunghezze = [len(tok.encode(v.core).ids) + 1 for v in voci]
    ks = [answer_tokens(tok, v.core, v.atteso) for v in voci]
    if tele:
        max_new = max(len(tok.encode(v.atteso).ids) for v in voci) + 4
    stop = punto if tele else None

    def massimo(chunk: list[int]) -> int:
        # interlingua: come Fase C, si genera fino alla lunghezza oracolo del batch
        return max_new if tele else max(ks[i] for i in chunk)

    gruppi = _gruppi(batch, lunghezze)
    for _ in range(WARMUP_CALLS):
        _genera(gen, tok, [voci[i].core for i in gruppi[0]], massimo(gruppi[0]), stop)
    for g in {len(g): g for g in gruppi}.values():   # cattura i CUDA graph di ogni dimensione di batch
        _genera(gen, tok, [voci[i].core for i in g], massimo(g), stop)

    strip_ms, core_ms, real_ms = [], [], []
    nuovi_per = [0] * len(voci)
    ok_core = [False] * len(voci)
    ok_e2e = [False] * len(voci)
    for chunk in gruppi:
        t0 = time.perf_counter()
        prompt = [stripper.strip(voci[i].domanda) if tele else inter.to_prompt(voci[i].domanda)
                  for i in chunk]
        strip_ms.append(_ms(t0))

        _sincronizza(device)
        t0 = time.perf_counter()
        righe = _genera(gen, tok, prompt, massimo(chunk), stop)
        _sincronizza(device)
        core_ms.append(_ms(t0))
        n_nuovi = len(righe[0])

        t0 = time.perf_counter()
        for r, i in enumerate(chunk):
            v = voci[i]
            riga = righe[r] if tele else righe[r][:ks[i]]
            testo = tok.decode(riga, skip_special_tokens=True)
            frase, canonico = realizzatore.realize(prompt[r], testo)
            if tele:
                ok_core[i] = testo.split(".", 1)[0].strip() == v.atteso.strip()
            else:
                ok_core[i] = corpus.parse_answer(testo, v.attr, "interlingua", "xx") == v.canonico
            ok_e2e[i] = frase == f"{v.domanda} {v.canonico}."
            nuovi_per[i] = n_nuovi
        real_ms.append(_ms(t0))

    return {"strip": strip_ms, "core": core_ms, "real": real_ms, "K": nuovi_per, "lung": lunghezze,
            "ks": ks, "ok_core": ok_core, "ok_e2e": ok_e2e}


def _misura_pieno(model: GPT2, tok: Tokenizer, voci: list[_Voce], batch: int) -> dict:
    """Core pieno di riferimento: domanda -> valore, stop su "."; tempi in ms per richiesta o batch."""
    gen = StaticGen(model, max_batch=batch)
    device = gen.device
    punto = tok.token_to_id(".")
    prompt = [v.domanda for v in voci]
    lunghezze = [len(tok.encode(p).ids) + 1 for p in prompt]
    gruppi = _gruppi(batch, lunghezze)
    # come per il telegrafico: la risposta piu' lunga + 4 token
    max_new = max(len(tok.encode(corpus.surface(v.attr, v.canonico, "pieno", "it")).ids) for v in voci) + 4
    for _ in range(WARMUP_CALLS):
        _genera(gen, tok, [prompt[i] for i in gruppi[0]], max_new, punto)
    for g in {len(g): g for g in gruppi}.values():   # cattura i CUDA graph di ogni dimensione di batch
        _genera(gen, tok, [prompt[i] for i in g], max_new, punto)

    ms = []
    nuovi_per = [0] * len(voci)
    ok = [False] * len(voci)
    for chunk in gruppi:
        _sincronizza(device)
        t0 = time.perf_counter()
        righe = _genera(gen, tok, [prompt[i] for i in chunk], max_new, punto)
        _sincronizza(device)
        ms.append(_ms(t0))
        for r, i in enumerate(chunk):
            nuovi_per[i] = len(righe[r])
            testo = tok.decode(righe[r], skip_special_tokens=True)
            ok[i] = corpus.parse_answer(testo, voci[i].attr, "pieno", "it") == voci[i].canonico
    return {"ms": ms, "lung": lunghezze, "K": nuovi_per, "ok": ok}


def _riga(formato: str, taglia: str, batch: int, voci: list[_Voce], core: dict, model: GPT2,
          pieno: dict, model_pieno: GPT2) -> dict:
    """Riga di PIPE_COLUMNS: medie per richiesta (batch 1) o per batch (batch 64)."""
    strip_m = float(np.mean(core["strip"]))
    core_m = float(np.mean(core["core"]))
    real_m = float(np.mean(core["real"]))
    totale = strip_m + core_m + real_m
    pieno_m = float(np.mean(pieno["ms"]))
    return {
        "format": formato, "size": taglia, "batch": batch, "n_questions": len(voci),
        "prompt_tokens_mean": float(np.mean(core["lung"])),
        "answer_tokens_mean": float(np.mean(core["ks"])),
        "strip_cpu_ms": strip_m, "core_gpu_ms": core_m, "realize_cpu_ms": real_m,
        "total_ms": totale, "pieno_ms": pieno_m, "ratio_vs_pieno": totale / pieno_m,
        "core_flops": _flops(model, core["lung"], core["K"]),
        "pieno_flops": _flops(model_pieno, pieno["lung"], pieno["K"]),
        "core_acc": float(np.mean(core["ok_core"])),
        "e2e_acc": float(np.mean(core["ok_e2e"])),
        "pieno_acc": float(np.mean(pieno["ok"])),
    }


def evaluate_stripped(stripper: _Stripper):
    """Come fastgen.evaluate_fast, con il core addestrato sul telegrafico strippato da `stripper`."""

    def valuta(model: GPT2, tok: Tokenizer, spec: CorpusSpec, persons: list[Person], eval_persons: int) -> dict:
        casi = cases(stripper, persons, eval_persons)
        device = next(model.parameters()).device
        eos = tok.token_to_id(EOS)
        punto = tok.token_to_id(".")
        ids = [[eos] + tok.encode(c[3]).ids for c in casi]
        max_new = max(len(tok.encode(c[4]).ids) for c in casi) + 4

        # prompt della stessa lunghezza insieme: la generazione non usa padding
        per_lunghezza: dict[int, list[int]] = {}
        for i, riga in enumerate(ids):
            per_lunghezza.setdefault(len(riga), []).append(i)

        testi = [""] * len(casi)
        model.eval()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False):
            max_batch = min(LOTTO_MAX, max((len(v) for v in per_lunghezza.values()), default=1))
            gen = StaticGen(model, max_batch=max_batch, dtype=torch.bfloat16)
            for indici in per_lunghezza.values():
                for k in range(0, len(indici), LOTTO_MAX):
                    blocco = indici[k:k + LOTTO_MAX]
                    idx = torch.tensor([ids[i] for i in blocco], dtype=torch.long, device=device)
                    nuovi, _ = gen.generate(idx, max_new, stop_id=punto)
                    for i, riga in zip(blocco, nuovi.tolist()):
                        testi[i] = tok.decode(riga, skip_special_tokens=True)

        in_ok = {a: [] for a in ATTRS}
        for caso, testo in zip(casi, testi):
            in_ok[caso[1]].append(testo.split(".", 1)[0].strip() == caso[4].strip())

        risultato = {}
        for a in ATTRS:
            risultato[f"acc_{a}"] = float(np.mean(in_ok[a])) if in_ok[a] else 0.0
            risultato[f"cross_{a}"] = float("nan")
        risultato["acc_mean"] = float(np.mean([risultato[f"acc_{a}"] for a in ATTRS]))
        risultato["cross_mean"] = float("nan")
        risultato["cross_lenient_mean"] = float("nan")
        return risultato

    return valuta


def _job(csv: Path, corpus_name: str, size: str) -> Job:
    """Job della riga seed 0 del CSV per corpus e taglia."""
    for rec in pd.read_csv(csv).to_dict("records"):
        if str(rec["corpus"]) == corpus_name and str(rec["size"]) == size and int(rec["seed"]) == 0:
            return Job(stage=str(rec["stage"]), corpus=str(rec["corpus"]), protocol=str(rec["protocol"]),
                       exposures=int(rec["exposures"]), size=str(rec["size"]),
                       lr_mult=float(rec["lr_mult"]), seed=int(rec["seed"]),
                       n_persons=int(rec["n_persons"]))
    raise ValueError(f"nessuna riga seed 0 in {csv} per {corpus_name} {size}")


def _carica(job: Job, radice: Path, persone: list[Person]) -> tuple[GPT2, Tokenizer]:
    """Modello in eval mode dal checkpoint radice/run_id; ctx_len e' il numero di righe di pos_emb."""
    percorso = radice / job.run_id / "ckpt.pt"
    if not percorso.exists():
        raise FileNotFoundError(f"checkpoint mancante: {percorso}")
    spec = CORPORA[job.corpus]
    tok = get_tokenizer(spec, persone)
    stato = torch.load(percorso, map_location="cpu")["model"]
    size = SIZES_V2[job.size]
    ctx_len = stato["pos_emb.weight"].shape[0]
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=ctx_len, n_layer=size.n_layer,
                 n_embd=size.n_embd, n_head=size.n_head)
    model.load_state_dict(stato)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tok


def _append(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row], columns=list(PIPE_COLUMNS)).to_csv(
        csv_path, mode="a", header=write_header, index=False)


def run_pipeline(out_csv: Path) -> None:
    """Misura FORMATI x PIPE_SIZES x PIPE_BATCHES contro il baseline pieno; salta le combinazioni gia' in out_csv."""
    out_csv = Path(out_csv)
    fatti = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        fatti = {(str(r["format"]), str(r["size"]), int(r["batch"]))
                 for r in pd.read_csv(out_csv).to_dict("records")}

    n_persone = -(-PIPE_QUESTIONS // len(ATTRS))   # persone che coprono le prime PIPE_QUESTIONS domande
    inter = InterRealizer()
    for formato in FORMATI:
        for taglia in PIPE_SIZES:
            da_fare = [b for b in PIPE_BATCHES if (formato, taglia, b) not in fatti]
            if not da_fare:
                continue
            tele = formato in STRIPPER_DI
            if tele:
                job = _job(V3_RESULTS / "E_main.csv", formato, taglia)
                radice = V3_RUNS
                stripper = STRIPPER_DI[formato]()
            else:
                job = _job(V2_RESULTS / "A_main.csv", formato, taglia)
                radice = V2_RUNS
                stripper = None
            persone = corpus.load_persons(job.n_persons)
            model, tok = _carica(job, radice, persone)
            if tele:
                voci = [_Voce(attr, domanda, canonico, core, atteso) for _, attr, domanda, core, atteso, canonico
                        in cases(stripper, persone, n_persone)[:PIPE_QUESTIONS]]
            else:
                voci = []
                for _, attr, domanda, canonico in _domande(persone, n_persone)[:PIPE_QUESTIONS]:
                    superficie = corpus.surface(attr, canonico, "interlingua", "xx")
                    voci.append(_Voce(attr, domanda, canonico, inter.to_prompt(domanda), superficie))

            # baseline pieno: stesse domande, prompt = domanda
            job_p = _job(V2_RESULTS / "A_main.csv", BASE_PIENO, taglia)
            model_p, tok_p = _carica(job_p, V2_RUNS, corpus.load_persons(job_p.n_persons))
            voci_p = [_Voce(v.attr, v.domanda, v.canonico, v.domanda, "") for v in voci]

            for batch in da_fare:
                core = _misura_core(stripper, inter, model, tok, voci, batch)
                pieno = _misura_pieno(model_p, tok_p, voci_p, batch)
                riga = _riga(formato, taglia, batch, voci, core, model, pieno, model_p)
                _append(out_csv, riga)
                fatti.add((formato, taglia, batch))
                print(f"{formato} {taglia} batch {batch}: total {riga['total_ms']:.2f} ms "
                      f"pieno {riga['pieno_ms']:.2f} ms core_acc {riga['core_acc']:.3f}")


if __name__ == "__main__":
    run_pipeline(V3_RESULTS / "pipeline3.csv")
