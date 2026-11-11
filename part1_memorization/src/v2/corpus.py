"""Corpora v2: rendering per stile/lingua, assegnazione lingue, prompt, parsing dei fatti."""
from __future__ import annotations

import functools
import math
import random
import re
from datetime import date, timedelta

from src import data
from src.config import ATTRS, INTERLINGUA_MARKERS
from src.data import Person
from src.v2 import lexicon
from src.v2.spec import BRIDGE_FRAC, DATA_SEED, LANG_SEED, CorpusSpec, EvalCase, Part

_DATA_INIZIO = date(1950, 1, 1)
_DATA_FINE = date(2005, 12, 31)
_N_GIORNI = (_DATA_FINE - _DATA_INIZIO).days + 1
# Liste canoniche per attributo (la data nascita è un intervallo di giorni)
_LISTE = {
    "nascita_luogo": data.CITTA,
    "universita": data.UNIVERSITA,
    "corso": data.CORSI,
    "professione": data.PROFESSIONI,
    "azienda": data.AZIENDE,
}
# Articoli italiani iniziali: "il ", "lo ", ... oppure "l'" attaccato
_ARTICOLO_IT = re.compile(r"^(?:(?:il|lo|la|gli|le|i)\s+|l')")


def load_persons(n: int) -> list[Person]:
    """Le stesse persone di v1: src.data.generate_persons(n, DATA_SEED)."""
    return data.generate_persons(n, DATA_SEED)


def assign_langs(n: int) -> tuple[list[str], frozenset[int]]:
    """(lingua "it"/"en" per persona, metà e metà; insieme degli indici ponte, BRIDGE_FRAC bilanciato)."""
    rng = random.Random(LANG_SEED)
    permutazione = list(range(n))
    rng.shuffle(permutazione)
    meta = n // 2
    langs = [""] * n
    for pos, idx in enumerate(permutazione):
        langs[idx] = "it" if pos < meta else "en"
    # Posizioni 0, 10, 20... della permutazione: metà in "it" e metà in "en"
    passo = round(1 / BRIDGE_FRAC)
    ponte = frozenset(permutazione[pos] for pos in range(0, n, passo))
    return langs, ponte


def _nome_interlingua(person: Person) -> str:
    return person.name.replace(" ", "_")


def _render_interlingua(person: Person, rng: random.Random) -> str:
    """Layout dell'interlingua v1, con nome e valori uniti da "_" e data ISO."""
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    parti = [_nome_interlingua(person)]
    parti.extend(f"{INTERLINGUA_MARKERS[a]} {surface(a, person.attrs[a], 'interlingua', 'xx')}"
                 for a in ordine)
    return " ".join(parti)


def render(person: Person, style: str, lang: str, rng: random.Random) -> str:
    if style == "pieno" and lang == "it":
        return data.render(person, "italiano_pieno", rng)
    if style == "telegrafico" and lang == "it":
        return data.render(person, "telegrafico", rng)
    if style == "pieno" and lang == "en":
        return lexicon.render_en(person, rng)
    if style == "ricco" and lang == "it":
        return lexicon.render_ricco(person, rng)
    if style == "interlingua_v1":
        return data.render(person, "interlingua", rng)
    if style == "interlingua":
        return _render_interlingua(person, rng)
    raise ValueError(f"stile/lingua sconosciuto: {style}/{lang}")


def eval_prompt(person: Person, attr: str, style: str, lang: str) -> str:
    if style == "pieno" and lang == "it":
        return data.eval_prompt(person, attr, "italiano_pieno")
    if style == "telegrafico" and lang == "it":
        return data.eval_prompt(person, attr, "telegrafico")
    if style == "pieno" and lang == "en":
        return lexicon.prompt_en(person, attr)
    if style == "ricco" and lang == "it":
        return lexicon.prompt_ricco(person, attr)
    if style == "interlingua_v1":
        return data.eval_prompt(person, attr, "interlingua")
    if style == "interlingua":
        return f"{_nome_interlingua(person)} {INTERLINGUA_MARKERS[attr]}"
    raise ValueError(f"stile/lingua sconosciuto: {style}/{lang}")


def surface(attr: str, canonical: str, style: str, lang: str) -> str:
    """Forma superficiale di un valore nello stile/lingua (data ISO e underscore in interlingua...)."""
    if style == "interlingua":
        if attr == "nascita_data":
            giorno, mese, anno = canonical.split(" ")
            return date(int(anno), data.MESI.index(mese) + 1, int(giorno)).isoformat()
        return canonical.replace(" ", "_")
    if style == "pieno" and lang == "en":
        return lexicon.surface_en(attr, canonical)
    return canonical


def _canonici(attr: str) -> list[str]:
    if attr == "nascita_data":
        giorni = [_DATA_INIZIO + timedelta(days=k) for k in range(_N_GIORNI)]
        return [f"{g.day} {data.MESI[g.month - 1]} {g.year}" for g in giorni]
    return _LISTE[attr]


@functools.lru_cache(maxsize=None)
def _tabella(attr: str, style: str, lang: str) -> tuple[dict[str, str], int]:
    """Superficie -> valore canonico, e lunghezza massima delle superfici."""
    tabella = {surface(attr, c, style, lang): c for c in _canonici(attr)}
    return tabella, max(len(s) for s in tabella)


def _e_parola(carattere: str) -> bool:
    return carattere.isalnum() or carattere == "_"


def parse_answer(text: str, attr: str, style: str, lang: str) -> str | None:
    """Valore canonico riconosciuto all'inizio della continuazione generata, o None."""
    testo = text.lstrip()
    if style == "ricco" and lang == "it":
        articolo = _ARTICOLO_IT.match(testo)
        if articolo:
            testo = testo[articolo.end():]
    elif style == "pieno" and lang == "en":
        testo = re.sub(r"^(?:the|an|a)\s+", "", testo)
    tabella, maxlen = _tabella(attr, style, lang)
    # Il prefisso più lungo che finisce a fine testo o prima di un carattere non parola
    for k in range(min(len(testo), maxlen), 0, -1):
        canonico = tabella.get(testo[:k])
        if canonico is not None and (k == len(testo) or not _e_parola(testo[k])):
            return canonico
    return None


def _indici(part: Part, langs: list[str], ponte: frozenset[int]) -> list[int]:
    n = len(langs)
    if part.persons == "all":
        return list(range(n))
    if part.persons == "lang":
        return [i for i in range(n) if langs[i] == part.lang]
    return [i for i in range(n) if i in ponte and langs[i] != part.lang]


def epoch_texts(spec: CorpusSpec, persons: list[Person], epoch: int) -> list[str]:
    """Tutte le biografie del corpus per l'epoca, mescolate; deterministica in (spec.name, epoch)."""
    langs, ponte = assign_langs(len(persons))
    rng = random.Random(f"{spec.name}|{epoch}")
    testi = []
    for part in spec.parts:
        for idx in _indici(part, langs, ponte):
            testi.append(render(persons[idx], part.style, part.lang, rng))
    rng.shuffle(testi)
    return testi


def eval_cases(spec: CorpusSpec, persons: list[Person], max_persons: int) -> list[EvalCase]:
    langs, ponte = assign_langs(len(persons))
    casi = []
    for i in range(min(max_persons, len(persons))):
        persona = persons[i]
        # "in": stile/lingua delle Part di training della persona
        for part in spec.parts:
            if part.persons == "all" or (part.persons == "lang" and part.lang == langs[i]):
                for attr in ATTRS:
                    casi.append(EvalCase("in", i, attr, part.style, part.lang,
                                         eval_prompt(persona, attr, part.style, part.lang),
                                         persona.attrs[attr]))
        # "cross": persona non ponte, valutata nell'altra lingua
        if spec.cross and i not in ponte:
            altra = "en" if langs[i] == "it" else "it"
            for attr in ATTRS:
                casi.append(EvalCase("cross", i, attr, "pieno", altra,
                                     eval_prompt(persona, attr, "pieno", altra),
                                     persona.attrs[attr]))
    return casi


def entropy_bits(n_persons: int) -> dict[str, float]:
    """Bit teorici per attributo (log2 della dimensione della lista o del range di date)."""
    return {a: math.log2(_N_GIORNI if a == "nascita_data" else len(_LISTE[a])) for a in ATTRS}
