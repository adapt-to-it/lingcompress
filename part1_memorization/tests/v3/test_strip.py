"""Test di src.v3.strip: stripper a lista e spacy, costruzione del corpus strippato (formato v2)."""
from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer

from src.config import EOS
from src.v2 import corpus, train
from src.v2.spec import CORPORA
from src.v3 import strip
from src.v3.strip import ListStripper, SpacyStripper, build_corpus, strip_bio

LISTA = ListStripper()


@pytest.mark.parametrize("testo, atteso", [
    ("Il gatto è nel giardino.", "gatto giardino."),
    ("Mario e Anna sono di Roma.", "Mario e Anna Roma."),
    ("L'uomo, la donna.", "uomo, donna."),
    ("Si chiama Luca.", "Si chiama Luca."),
    ("IL CANE DELLA CASA", "CANE CASA"),
])
def test_lista_cases(testo, atteso):
    assert LISTA.strip(testo) == atteso


def test_lista_spans():
    assert LISTA.spans("Il gatto") == [("gatto", 3, 8)]


def test_lista_strip_many():
    testi = ["Il gatto è nel giardino.", "Si chiama Luca."]
    assert LISTA.strip_many(testi) == [LISTA.strip(t) for t in testi]


SPACY = SpacyStripper()


def test_spacy_elimina_articolo_e_tiene_punteggiatura_attaccata():
    out = SPACY.strip("Il gatto dorme sul divano.")
    assert "gatto" in out.split()
    assert "sul" not in out.split()
    assert out.endswith(".")
    assert " ." not in out


def test_spacy_spans_sono_nel_testo():
    testo = "Mario è nato a Roma."
    for tok, inizio, fine in SPACY.spans(testo):
        assert testo[inizio:fine] == tok


def test_spacy_strip_many_uguale_a_strip():
    testi = ["Mario è nato a Roma.", "Il gatto dorme sul divano."]
    assert SPACY.strip_many(testi) == [SPACY.strip(t) for t in testi]


def test_strip_bio_usa_la_cache_per_frase():
    cache = {}
    out = strip_bio(LISTA, "Il gatto è nel giardino. Si chiama Luca.", cache)
    assert out == "gatto giardino. Si chiama Luca."
    assert set(cache) == {"Il gatto è nel giardino", "Si chiama Luca."}


def test_build_corpus_scrive_file_e_epoca_0_decodifica_testi_strippati(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "V2_DATA", tmp_path)
    persons = corpus.load_persons(50)
    build_corpus("lista_test", LISTA, persons, 2)

    spec = CORPORA["lista_test"]
    assert train._tok_path(spec, 50).exists()
    assert train._epoch_path(spec, 50, 0).exists()
    assert train._epoch_path(spec, 50, 1).exists()

    atteso = [strip.strip_bio(LISTA, bio) for bio in corpus.epoch_texts(CORPORA["italiano_pieno"], persons, 0)]
    tok = Tokenizer.from_file(str(train._tok_path(spec, 50)))
    with np.load(train._epoch_path(spec, 50, 0)) as z:
        tokens, offsets = z["tokens"], z["offsets"]
    ottenuto = []
    for i in range(len(offsets) - 1):
        bio = tokens[offsets[i]:offsets[i + 1]].tolist()
        assert bio[0] == tok.token_to_id(EOS) and bio[-1] == tok.token_to_id(EOS)
        ottenuto.append(tok.decode(bio[1:-1]))
    assert ottenuto == atteso
