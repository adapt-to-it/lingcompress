"""Test di src.v3.realize: realizzatore telegrafico (lista e spacy) e realizzatore interlingua."""
from __future__ import annotations

import pytest

from src.config import ATTRS, INTERLINGUA_MARKERS
from src.v2 import corpus
from src.v3.realize import InterRealizer, TeleRealizer, gender, question
from src.v3.strip import ListStripper, SpacyStripper

PERSONE = corpus.load_persons(300)


@pytest.fixture(scope="module", params=[ListStripper, SpacyStripper], ids=["lista", "spacy"])
def coppia(request):
    stripper = request.param()
    return stripper, TeleRealizer(stripper)


def test_gender_per_tutte_le_persone():
    for p in PERSONE:
        assert gender(p.name) == p.gender


def test_realize_frase_pieno(coppia):
    stripper, realizer = coppia
    casi = 0
    corretti = 0
    for p in PERSONE:
        for attr in ATTRS:
            casi += 1
            domanda = question(p, attr)
            valore = realizer._dopo(domanda, f"{p.attrs[attr]}.")
            try:
                sentenza, canonico = realizer.realize(stripper.strip(domanda), f"{valore}. next")
            except Exception:
                continue  # eccezione: conta come fallimento
            if sentenza == f"{domanda} {p.attrs[attr]}." and canonico == p.attrs[attr]:
                corretti += 1
    assert corretti >= 0.99 * casi, f"{corretti}/{casi} corretti"


def test_interlingua_round_trip():
    inter = InterRealizer()
    for p in PERSONE:
        for attr in ATTRS:
            domanda = question(p, attr)
            prompt = inter.to_prompt(domanda)
            assert prompt == f"{p.name.replace(' ', '_')} {INTERLINGUA_MARKERS[attr]}"
            superficie = corpus.surface(attr, p.attrs[attr], "interlingua", "xx")
            sentenza, canonico = inter.realize(prompt, f"{superficie} next")
            assert canonico == p.attrs[attr]
            assert sentenza == f"{domanda} {p.attrs[attr]}."
