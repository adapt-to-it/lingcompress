"""Test del modulo data: determinismo, nomi unici, template ed estrazione della risposta."""
from __future__ import annotations

import random
import re
from datetime import date
from itertools import permutations

import pytest

from src.config import ATTRS, EOS, FORMATS, INTERLINGUA_MARKERS
from src.data import (
    AZIENDE, CITTA, CORSI, COGNOMI, MESI, NOMI_F, NOMI_M, PROFESSIONI, UNIVERSITA,
    TELEGRAFICO, Person, epoch_texts, eval_prompt, extract_answer, generate_persons, render,
)

MARCATORI = set(INTERLINGUA_MARKERS.values())
RE_DATA = re.compile(r"([1-9]|[12]\d|3[01]) (\w+) (\d{4})")


def test_dimensioni_liste():
    assert len(NOMI_F) >= 150
    assert len(NOMI_M) >= 150
    assert len(COGNOMI) >= 300
    assert len(CITTA) >= 190
    assert len(UNIVERSITA) >= 120
    assert len(CORSI) >= 120
    assert len(PROFESSIONI) >= 120
    assert len(AZIENDE) >= 150


@pytest.mark.parametrize("lista", [NOMI_F, NOMI_M, COGNOMI, CITTA, UNIVERSITA, CORSI, PROFESSIONI, AZIENDE])
def test_liste_senza_duplicati(lista):
    assert len(set(lista)) == len(lista)


@pytest.mark.parametrize("lista", [CITTA, UNIVERSITA, CORSI, PROFESSIONI, AZIENDE])
def test_valori_senza_punto_e_senza_marcatori(lista):
    for valore in lista:
        assert "." not in valore
        assert not set(valore.split()) & MARCATORI, valore


def test_date_formato_e_intervallo():
    for p in generate_persons(500, 3):
        m = RE_DATA.fullmatch(p.attrs["nascita_data"])
        assert m, p.attrs["nascita_data"]
        giorno, mese, anno = int(m.group(1)), MESI.index(m.group(2)) + 1, int(m.group(3))
        data = date(anno, mese, giorno)  # solleva se la data non esiste
        assert date(1950, 1, 1) <= data <= date(2005, 12, 31)


def test_deterministico():
    a = generate_persons(300, 7)
    b = generate_persons(300, 7)
    assert a == b
    assert generate_persons(300, 8) != a
    for fmt in FORMATS:
        assert epoch_texts(a, fmt, 2, 1234) == epoch_texts(b, fmt, 2, 1234)
        assert render(a[0], fmt, random.Random(5)) == render(a[0], fmt, random.Random(5))


def test_nomi_unici():
    persone = generate_persons(3000, 1)
    nomi = [p.name for p in persone]
    assert len(set(nomi)) == len(nomi)


def test_capacita():
    capacita = (len(NOMI_F) + len(NOMI_M)) * len(COGNOMI)
    with pytest.raises(ValueError):
        generate_persons(capacita + 1, 0)
    tutte = generate_persons(capacita, 0)
    assert len({p.name for p in tutte}) == capacita


def test_render_contiene_tutti_i_valori():
    for p in generate_persons(200, 2):
        for fmt in FORMATS:
            for k in range(3):
                testo = render(p, fmt, random.Random(k))
                for valore in p.attrs.values():
                    assert valore in testo


def test_struttura_frasi():
    for p in generate_persons(50, 4):
        for fmt in ("italiano_pieno", "telegrafico"):
            testo = render(p, fmt, random.Random(0))
            assert testo.count(".") == len(ATTRS)
            assert testo.startswith(p.name + " ")
        testo = render(p, "interlingua", random.Random(0))
        assert testo.startswith(p.name + " ")
        assert "." not in testo


def test_prefissi_template0_distinti():
    for fmt in FORMATS:
        for genere, nome in (("F", "Marta Rinaldi"), ("M", "Marco Rinaldi")):
            p = Person(nome, genere, {})
            prompt = [eval_prompt(p, a, fmt) for a in ATTRS]
            assert all(not s.endswith(" ") for s in prompt)
            prefissi = [s[len(nome) + 1:] for s in prompt]
            for i, j in permutations(range(len(ATTRS)), 2):
                assert prefissi[i] != prefissi[j]
                assert not prefissi[j].startswith(prefissi[i]), (fmt, prefissi[i], prefissi[j])


def test_eval_prompt_e_estrazione():
    for p in generate_persons(100, 5):
        for fmt in FORMATS:
            for attr in ATTRS:
                valore = p.attrs[attr]
                prompt = eval_prompt(p, attr, fmt)
                # Il template 0 compare con soggetto = nome solo quando la frase e' la prima:
                # cerco un ordine di campi (seed) in cui compaia.
                trovato = any(
                    prompt + " " + valore in render(p, fmt, random.Random(k))
                    for k in range(2000)
                )
                assert trovato, (p.name, fmt, attr)
                if fmt == "interlingua":
                    continuazione = f" {valore} UNI x"
                else:
                    continuazione = f" {valore}. Lei ha studiato"
                assert extract_answer(continuazione, fmt) == valore


def test_extract_answer_terminatori():
    assert extract_answer(" Bari<eos>Lei", "italiano_pieno") == "Bari"
    assert extract_answer(" Bari\nLei", "telegrafico") == "Bari"
    assert extract_answer(f" Bari {EOS}", "interlingua") == "Bari"
    assert extract_answer(" Politecnico di Torino AZ Barilla", "interlingua") == "Politecnico di Torino"


def test_telegrafico_senza_preposizioni_e_articoli():
    # Solo le parole fuori dal valore; i segnaposto di genere ({nat}, {lau}) sono esclusi
    vietate = {"presso", "in", "di", "con", "a", "il", "la", "lo", "gli", "le", "un", "una", "è", "ha", "si"}
    for attr, templates in TELEGRAFICO.items():
        for tpl in templates:
            parole = [w for w in tpl.replace("{v}", " ").split() if not w.startswith("{")]
            assert not set(parole) & vietate, (attr, tpl)


def test_epoch_texts():
    persone = generate_persons(40, 9)
    for fmt in FORMATS:
        testi = epoch_texts(persone, fmt, 1, 1234)
        assert len(testi) == len(persone)
        assert epoch_texts(persone, fmt, 1, 1234) == testi
        assert epoch_texts(persone, fmt, 2, 1234) != testi
