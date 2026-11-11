"""Valutazione v4: come evaluate_stripped, con prompt e atteso di corpus4 per il formato dato."""
from __future__ import annotations

import numpy as np
import torch
from tokenizers import Tokenizer

from src.config import ATTRS, EOS
from src.data import Person
from src.v2.model2 import GPT2
from src.v2.spec import CorpusSpec
from src.v3.fastgen import StaticGen
from src.v3.pipeline3 import LOTTO_MAX
from src.v4 import corpus4


def make_evaluate(fmt: str):
    """Funzione di valutazione per il formato: persone 0..eval_persons-1 e ogni attributo."""

    def evaluate(model: GPT2, tok: Tokenizer, spec: CorpusSpec, persons: list[Person], eval_persons: int) -> dict:
        casi = [(attr, corpus4.prompt(fmt, persona, attr), corpus4.expected(fmt, persona, attr))
                for persona in persons[:eval_persons] for attr in ATTRS]
        device = next(model.parameters()).device
        eos = tok.token_to_id(EOS)
        punto = tok.token_to_id(".")
        ids = [[eos] + tok.encode(prompt).ids for _, prompt, _ in casi]
        max_new = max((len(tok.encode(" " + atteso).ids) for _, _, atteso in casi), default=0) + 4

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
        for (attr, _, atteso), testo in zip(casi, testi):
            in_ok[attr].append(testo.split(".", 1)[0].strip() == atteso)

        risultato = {}
        for a in ATTRS:
            risultato[f"acc_{a}"] = float(np.mean(in_ok[a])) if in_ok[a] else 0.0
            risultato[f"cross_{a}"] = float("nan")
        risultato["acc_mean"] = float(np.mean([risultato[f"acc_{a}"] for a in ATTRS]))
        risultato["cross_mean"] = float("nan")
        risultato["cross_lenient_mean"] = float("nan")
        return risultato

    return evaluate
