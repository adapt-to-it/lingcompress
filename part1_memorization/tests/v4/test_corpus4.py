"""Test di src.v4.corpus4: canonico, stripping, prompt e atteso, build su dati temporanei."""
from __future__ import annotations

import random
import re

import numpy as np

from src.data import FORME_GENERE, Person, generate_persons
from src.v2 import train
from src.v2.spec import CORPORA
from src.v2.train import _epoch_path, _tok_path
from src.v3 import realize
from src.v3.strip import ListStripper, strip_bio
from src.v4 import corpus4

PERSONA = Person("Marta Rinaldi", "F", {
    "nascita_data": "12 marzo 1987",
    "nascita_luogo": "Ancona",
    "universita": "Università di Chieti-Pescara",
    "corso": "Ingegneria chimica",
    "professione": "ingegnere",
    "azienda": "Barilla",
})


def _forme_assenti(testo: str, genere: str) -> bool:
    return not any(re.search(rf"\b{forma}\b", testo) for forma in FORME_GENERE[genere].values())


def test_canonico_sei_frasi_nome_solo_nella_prima():
    for persona in generate_persons(20, 1):
        testo = corpus4.render_canonico(persona, random.Random(3))
        frasi = testo.split(". ")
        assert len(frasi) == 6
        assert testo.endswith(".")
        assert testo.count(persona.name) == 1
        assert frasi[0].startswith(persona.name + " ")
        assert all(f[0].isupper() for f in frasi[1:])
        assert not re.search(r"\b(Lui|Lei)\b", testo)


def test_canonico_accordo_di_genere():
    for persona in generate_persons(20, 1):
        testo = corpus4.render_canonico(persona, random.Random(5))
        altro = "M" if persona.gender == "F" else "F"
        assert _forme_assenti(testo, altro)


def test_telegrafico_canonico_e_strip_del_canonico():
    persone = generate_persons(10, 2)
    for epoca in (0, 1):
        canonico = corpus4.epoch_texts("italiano_canonico", persone, epoca)
        telegrafico = corpus4.epoch_texts("telegrafico_canonico", persone, epoca)
        assert telegrafico == [strip_bio(ListStripper(), t) for t in canonico]
    assert corpus4.epoch_texts("italiano_canonico", persone, 0) == \
        corpus4.epoch_texts("italiano_canonico", persone, 0)


def test_esempi_strippati_sono_lo_strip_dell_esempio_sorgente():
    pieno = corpus4.examples("italiano_pieno", PERSONA, 2)
    assert corpus4.examples("telegrafico_lista", PERSONA, 2) == \
        [strip_bio(ListStripper(), t) for t in pieno]
    canonico = corpus4.examples("italiano_canonico", PERSONA, 2)
    assert corpus4.examples("telegrafico_canonico", PERSONA, 2) == \
        [strip_bio(ListStripper(), t) for t in canonico]


def test_prompt_e_atteso_formati_pieni():
    domanda = realize.question(PERSONA, "universita")
    assert corpus4.prompt("italiano_pieno", PERSONA, "universita") == domanda
    assert domanda.startswith("Marta Rinaldi ")
    assert corpus4.expected("italiano_pieno", PERSONA, "universita") == "Università di Chieti-Pescara"


def test_prompt_e_atteso_formati_telegrafici():
    domanda = realize.question(PERSONA, "universita")
    for fmt in ("telegrafico_lista", "telegrafico_canonico"):
        assert corpus4.prompt(fmt, PERSONA, "universita") == ListStripper().strip(domanda)
        assert corpus4.expected(fmt, PERSONA, "universita") == "Università Chieti- Pescara"


def test_build_scrive_tokenizer_ed_epoche(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "V2_DATA", tmp_path)
    persone = generate_persons(30, 0)
    corpus4.build("telegrafico_canonico", persone, 2)
    spec = CORPORA[f"telegrafico_canonico@{corpus4.VOCAB}"]
    assert _tok_path(spec, 30).exists()
    for e in (0, 1):
        path = _epoch_path(spec, 30, e)
        assert path.exists()
        with np.load(path) as dati:
            assert len(dati["offsets"]) == 31
    scritto = _epoch_path(spec, 30, 1).stat().st_mtime_ns
    corpus4.build("telegrafico_canonico", persone, 2)
    assert _epoch_path(spec, 30, 1).stat().st_mtime_ns == scritto
