"""Test di src.v2.corpus: parsing, lingue, epoche, conteggi di valutazione."""
from __future__ import annotations

import math
import random

import pytest

from src import data
from src.config import ATTRS
from src.data import Person
from src.v2 import corpus, lexicon
from src.v2.spec import CORPORA

N = 200
CONTINUAZIONI = (". ", ", ", " NASC_L x", "")
STILI = (("pieno", "it"), ("telegrafico", "it"), ("interlingua", "xx"), ("interlingua_v1", "xx"))


@pytest.fixture
def stub_lexicon(monkeypatch):
    monkeypatch.setattr(lexicon, "render_en", lambda person, rng: f"{person.name} stub.")
    monkeypatch.setattr(lexicon, "prompt_en", lambda person, attr: f"{person.name} the")


def _persona() -> Person:
    return Person("Marta Rinaldi", "F", {
        "nascita_data": "12 marzo 1987",
        "nascita_luogo": "Reggio Calabria",
        "universita": "Politecnico di Torino",
        "corso": "Chimica industriale",
        "professione": "ingegnere",
        "azienda": "Riso Scotti",
    })


@pytest.mark.parametrize("style,lang", STILI)
@pytest.mark.parametrize("cont", CONTINUAZIONI)
def test_parse_round_trip(style, lang, cont):
    for persona in corpus.load_persons(20):
        for attr in ATTRS:
            valore = persona.attrs[attr]
            testo = corpus.surface(attr, valore, style, lang) + cont
            assert corpus.parse_answer(testo, attr, style, lang) == valore


def test_parse_prefisso_piu_lungo_vince():
    assert corpus.parse_answer("Chimica industriale. ", "corso", "pieno", "it") == "Chimica industriale"
    assert corpus.parse_answer("Chimica. ", "corso", "pieno", "it") == "Chimica"


def test_parse_non_parola_dopo_prefisso():
    # "_" dopo il prefisso fa parte della parola: "Chimica" non vale per "Chimica_x"
    assert corpus.parse_answer("Chimica_x", "corso", "interlingua", "xx") is None
    assert corpus.parse_answer("Chimica_industriale", "corso", "interlingua", "xx") == "Chimica industriale"


def test_parse_nativo_nome_data_underscore():
    assert corpus.surface("nascita_data", "12 marzo 1987", "interlingua", "xx") == "1987-03-12"
    assert corpus.surface("universita", "Politecnico di Torino", "interlingua", "xx") == "Politecnico_di_Torino"
    assert corpus.parse_answer("1987-03-12 UNI x", "nascita_data", "interlingua", "xx") == "12 marzo 1987"
    assert corpus.parse_answer("Politecnico_di_Torino NASC_D", "universita", "interlingua", "xx") == \
        "Politecnico di Torino"


def test_parse_articoli_ricco():
    assert corpus.parse_answer("il medico. ", "professione", "ricco", "it") == "medico"
    assert corpus.parse_answer("l'architetto", "professione", "ricco", "it") == "architetto"
    assert corpus.parse_answer("insegnante.", "professione", "ricco", "it") == "insegnante"
    assert corpus.parse_answer("", "professione", "ricco", "it") is None


def test_render_nativo():
    testo = corpus.render(_persona(), "interlingua", "xx", random.Random(0))
    assert testo.startswith("Marta_Rinaldi ")
    assert "NASC_D 1987-03-12" in testo
    assert "NASC_L Reggio_Calabria" in testo
    assert "UNI Politecnico_di_Torino" in testo
    assert "CORSO Chimica_industriale" in testo
    assert "AZ Riso_Scotti" in testo
    assert "LAV ingegnere" in testo
    assert "Marta Rinaldi" not in testo


def test_render_interlingua_v1_e_v1():
    persona = _persona()
    assert corpus.render(persona, "interlingua_v1", "xx", random.Random(7)) == \
        data.render(persona, "interlingua", random.Random(7))


def test_eval_prompt_nativo():
    assert corpus.eval_prompt(_persona(), "universita", "interlingua", "xx") == "Marta_Rinaldi UNI"


def test_assign_langs_bilanciato():
    langs, ponte = corpus.assign_langs(20000)
    assert langs.count("it") == 10000
    assert langs.count("en") == 10000
    assert len(ponte) == 2000
    assert sum(langs[i] == "it" for i in ponte) == 1000
    assert corpus.assign_langs(N) == corpus.assign_langs(N)


def test_epoch_texts_deterministico():
    persone = corpus.load_persons(N)
    spec = CORPORA["italiano_pieno"]
    testi = corpus.epoch_texts(spec, persone, 3)
    assert len(testi) == N
    assert testi == corpus.epoch_texts(spec, persone, 3)
    assert testi != corpus.epoch_texts(spec, persone, 4)


def test_epoch_texts_multi_pieno(stub_lexicon):
    persone = corpus.load_persons(N)
    assert len(corpus.epoch_texts(CORPORA["multi_pieno"], persone, 0)) == N


def test_epoch_texts_multi_bridge(stub_lexicon):
    persone = corpus.load_persons(N)
    _, ponte = corpus.assign_langs(N)
    assert len(corpus.epoch_texts(CORPORA["multi_bridge"], persone, 0)) == N + len(ponte)


@pytest.mark.parametrize("nome", ("italiano_pieno", "telegrafico", "interlingua"))
def test_eval_mono(nome):
    persone = corpus.load_persons(N)
    spec = CORPORA[nome]
    casi = corpus.eval_cases(spec, persone, 50)
    assert len(casi) == 50 * len(ATTRS)
    assert all(c.kind == "in" for c in casi)
    assert all(c.lang == spec.parts[0].lang for c in casi)
    assert all(c.truth == persone[c.person].attrs[c.attr] for c in casi)


@pytest.mark.parametrize("nome", ("multi_pieno", "multi_bridge"))
def test_eval_multi(nome, stub_lexicon):
    persone = corpus.load_persons(N)
    langs, ponte = corpus.assign_langs(N)
    casi = corpus.eval_cases(CORPORA[nome], persone, N)
    in_ = [c for c in casi if c.kind == "in"]
    cross = [c for c in casi if c.kind == "cross"]
    assert len(in_) == N * len(ATTRS)
    assert len(cross) == (N - len(ponte)) * len(ATTRS)
    assert all(c.lang == langs[c.person] for c in in_)
    assert all(c.lang != langs[c.person] and c.style == "pieno" for c in cross)
    assert all(c.person not in ponte for c in cross)


def test_entropy_bits():
    bit = corpus.entropy_bits(N)
    assert set(bit) == set(ATTRS)
    assert bit["nascita_data"] == pytest.approx(math.log2(20454))
    assert bit["nascita_luogo"] == pytest.approx(math.log2(len(data.CITTA)))
