"""Forme delle frasi v4: frasi con valore e nome sostituiti, statistiche per attributo e conflitti dei prompt."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import ATTRS
from src.data import Person
from src.v4 import corpus4
from src.v4.spec4 import FORMS_COLUMNS

NOME = "NOME"
GENERI = ("F", "M")
_VALORE = re.compile(r"VAL_([a-z_]+)_([FM])")


def _segnaposto(persons: list[Person]) -> list[Person]:
    """Persone con nome NOME e valori VAL_<attr>_<genere>, nel genere delle persone date."""
    return [Person(NOME, p.gender, {a: f"VAL_{a}_{p.gender}" for a in ATTRS}) for p in persons]


def _parole(testo: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", testo.lower()))


def sentence_forms(fmt: str, persons: list[Person], epoch: int = 0) -> pd.DataFrame:
    """Una riga per frase: attributo e genere dal token VAL_, prima se contiene NOME, forma con {v} e {n}."""
    righe = []
    for testo in corpus4.epoch_texts(fmt, _segnaposto(persons), epoch):
        frasi = testo.split(". ")
        if frasi[-1].endswith("."):
            frasi[-1] = frasi[-1][:-1]
        for frase in frasi:
            m = _VALORE.search(frase)
            if m is None:
                raise ValueError(f"frase senza valore: {frase!r}")
            forma = frase.replace(m.group(0), "{v}").replace(NOME, "{n}")
            righe.append((m.group(1), m.group(2), NOME in frase, forma))
    return pd.DataFrame(righe, columns=["attr", "gender", "first", "form"])


def form_stats(fmt: str, persons: list[Person]) -> pd.DataFrame:
    """Per attributo: forme distinte, forme distinte per genere M nelle frasi non iniziali, bit per biografia."""
    df = sentence_forms(fmt, persons)
    chiavi = ["attr", "gender", "first"]
    n_gruppo = df.groupby(chiavi)["form"].transform("size")
    n_forma = df.groupby([*chiavi, "form"])["form"].transform("size")
    df["bits"] = -np.log2(n_forma / n_gruppo)
    righe = []
    for attr in ATTRS:
        sub = df[df["attr"] == attr]
        core = sub[(sub["gender"] == "M") & ~sub["first"]]
        righe.append({
            "format": fmt, "attr": attr,
            "n_forms": sub["form"].nunique(),
            "n_forms_core": core["form"].nunique(),
            "entropy_bits": float(sub["bits"].sum()) / len(persons),
        })
    return pd.DataFrame(righe, columns=FORMS_COLUMNS)


def prompt_conflicts(fmt: str, persons: list[Person]) -> list[str]:
    """Conflitti dei prompt per genere e attributo; lista vuota se il prompt e' non ambiguo."""
    df = sentence_forms(fmt, persons)
    prime: dict[str, set[tuple[str, ...]]] = {a: set() for a in ATTRS}
    tutte: dict[str, set[tuple[str, ...]]] = {a: set() for a in ATTRS}
    for attr, prima, forma in zip(df["attr"], df["first"], df["form"]):
        parte = _parole(forma.split("{v}")[0].replace("{n}", ""))
        tutte[attr].add(parte)
        if prima:
            prime[attr].add(parte)

    conflitti = []
    for genere in GENERI:
        segnaposto = Person(NOME, genere, {a: f"VAL_{a}_{genere}" for a in ATTRS})
        for attr in ATTRS:
            p = _parole(corpus4.prompt(fmt, segnaposto, attr).replace(NOME, ""))
            if not p:
                conflitti.append(f"{fmt} {genere} {attr}: prompt vuoto")
                continue
            if p not in prime[attr]:
                conflitti.append(f"{fmt} {genere} {attr}: prompt {' '.join(p)!r} non e' una parte prima di {{v}} di una prima frase")
            for altro in ATTRS:
                if altro != attr and any(parte[:len(p)] == p for parte in tutte[altro]):
                    conflitti.append(f"{fmt} {genere} {attr}: prompt {' '.join(p)!r} prefisso di una frase di {altro}")
    return conflitti
