"""Test di src.v4.forms4: forme delle frasi, statistiche per attributo e conflitti dei prompt."""
from __future__ import annotations

from src.config import ATTRS
from src.data import generate_persons
from src.v4 import corpus4, forms4
from src.v4.spec4 import FORMS_COLUMNS

PERSONE = generate_persons(3000, 0)
FORMATI = ("italiano_pieno", "italiano_canonico", "telegrafico_lista", "telegrafico_canonico")


def test_sentence_forms_colonne_e_segnaposti():
    df = forms4.sentence_forms("italiano_pieno", PERSONE[:50])
    assert list(df.columns) == ["attr", "gender", "first", "form"]
    assert set(df["attr"]) <= set(ATTRS)
    assert set(df["gender"]) <= {"F", "M"}
    assert all("{v}" in f for f in df["form"])
    assert not any("VAL_" in f or forms4.NOME in f for f in df["form"])
    assert df["first"].dtype == bool


def test_form_stats_colonne_e_attributi():
    stats = forms4.form_stats("italiano_pieno", PERSONE)
    assert list(stats.columns) == list(FORMS_COLUMNS)
    assert list(stats["attr"]) == list(ATTRS)


def test_form_stats_pieno_17_bit():
    stats = forms4.form_stats("italiano_pieno", PERSONE)
    assert abs(stats["entropy_bits"].sum() - 17.0) < 0.05


def test_form_stats_canonico_zero_bit():
    stats = forms4.form_stats("italiano_canonico", PERSONE)
    assert abs(stats["entropy_bits"].sum()) < 0.05


def test_telegrafico_lista_tra_zero_e_17_bit():
    totale = forms4.form_stats("telegrafico_lista", PERSONE)["entropy_bits"].sum()
    assert 0 < totale < 17


def test_nessun_conflitto_nei_quattro_formati():
    for fmt in FORMATI:
        assert forms4.prompt_conflicts(fmt, PERSONE) == [], fmt


def test_conflitto_rilevato_se_due_prompt_uguali(monkeypatch):
    originale = corpus4.prompt

    def finto(fmt, persona, attr):
        return originale(fmt, persona, "universita" if attr == "corso" else attr)

    monkeypatch.setattr(corpus4, "prompt", finto)
    conflitti = forms4.prompt_conflicts("italiano_pieno", PERSONE)
    assert conflitti
    assert any("corso" in c for c in conflitti)
