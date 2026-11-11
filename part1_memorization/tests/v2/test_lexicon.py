"""Test del lessico v2: prompt, articoli, liste inglesi, determinismo, contorno."""
import random

from src.config import ATTRS
from src.data import CORSI, PROFESSIONI, Person, generate_persons
from src.v2.lexicon import (
    CORSI_EN,
    PROFESSIONI_EN,
    _con_articolo,
    prompt_en,
    prompt_ricco,
    render_en,
    render_ricco,
    surface_en,
)

PERSONE = generate_persons(300, seed=1234)
N_RENDER = 60


def _rng(persona: Person, j: int) -> random.Random:
    return random.Random(f"{persona.name}|{j}")


def _occorrenze(testo: str, prompt: str):
    pos = testo.find(prompt)
    while pos != -1:
        yield pos
        pos = testo.find(prompt, pos + 1)


def _atteso_ricco(attr: str, valore: str) -> str:
    """Valore come appare dopo il prompt: con articolo per universita e professione."""
    if attr in ("universita", "professione"):
        return _con_articolo(valore)
    return valore


def _atteso_en(attr: str, valore: str) -> str:
    superficie = surface_en(attr, valore)
    if attr == "professione":
        articolo = "an" if superficie[0].lower() in "aeiou" else "a"
        return f"{articolo} {superficie}"
    return superficie


def _controlla_prompt(testo: str, prompt: str, atteso: str) -> None:
    """Ogni occorrenza del prompt e seguita da ' ' e dal valore atteso."""
    for pos in _occorrenze(testo, prompt):
        resto = testo[pos + len(prompt):]
        assert resto.startswith(" " + atteso), (prompt, atteso, resto[:80])


def test_ricco_prompt_seguito_dal_valore():
    for persona in PERSONE:
        for j in range(N_RENDER):
            testo = render_ricco(persona, _rng(persona, j))
            for attr in ATTRS:
                _controlla_prompt(
                    testo, prompt_ricco(persona, attr), _atteso_ricco(attr, persona.attrs[attr]),
                )


def test_en_prompt_seguito_dal_valore():
    for persona in PERSONE:
        for j in range(N_RENDER):
            testo = render_en(persona, _rng(persona, j))
            for attr in ATTRS:
                _controlla_prompt(testo, prompt_en(persona, attr), _atteso_en(attr, persona.attrs[attr]))


def test_ricco_contiene_ogni_valore():
    for persona in PERSONE:
        for j in range(N_RENDER):
            testo = render_ricco(persona, _rng(persona, j))
            for attr in ATTRS:
                assert persona.attrs[attr] in testo, (persona.name, attr)


def test_ricco_professione_senza_articolo_dopo_come_e():
    # Dopo "come" ed "è" il valore compare nudo; l'articolo resta solo dopo "fa"
    for persona in PERSONE[:50]:
        for j in range(N_RENDER):
            testo = render_ricco(persona, _rng(persona, j))
            for forma in ("come il ", "come lo ", "come la ", "come l'", "è il ", "è lo ", "è la ", "è l'"):
                assert forma not in testo, (forma, testo)


def test_en_contiene_ogni_valore():
    for persona in PERSONE:
        for j in range(N_RENDER):
            testo = render_en(persona, _rng(persona, j))
            for attr in ATTRS:
                assert surface_en(attr, persona.attrs[attr]) in testo, (persona.name, attr)


def test_ricco_contiene_contorno():
    # 6 frasi informative + almeno 1 contorno: ogni frase termina con un punto
    for persona in PERSONE:
        for j in range(N_RENDER):
            testo = render_ricco(persona, _rng(persona, j))
            assert testo.count(".") >= 7, testo


def test_determinismo():
    for persona in PERSONE:
        assert render_ricco(persona, random.Random(7)) == render_ricco(persona, random.Random(7))
        assert render_en(persona, random.Random(7)) == render_en(persona, random.Random(7))


def test_liste_en_allineate_e_uniche():
    assert len(CORSI_EN) == len(CORSI) and len(set(CORSI_EN)) == len(CORSI_EN)
    assert len(PROFESSIONI_EN) == len(PROFESSIONI) and len(set(PROFESSIONI_EN)) == len(PROFESSIONI_EN)


def test_surface_en_date_e_liste():
    assert surface_en("nascita_data", "12 marzo 1987") == "12 March 1987"
    for corso in CORSI:
        assert surface_en("corso", corso) in CORSI_EN
    for professione in PROFESSIONI:
        assert surface_en("professione", professione) in PROFESSIONI_EN


def test_articoli():
    assert _con_articolo("Università di Bologna") == "l'Università di Bologna"
    assert _con_articolo("Politecnico di Torino") == "il Politecnico di Torino"
    assert _con_articolo("Scuola Normale Superiore") == "la Scuola Normale Superiore"
    assert _con_articolo("Accademia di Belle Arti di Brera") == "l'Accademia di Belle Arti di Brera"
    assert _con_articolo("medico") == "il medico"
    assert _con_articolo("ingegnere") == "l'ingegnere"
    assert _con_articolo("statistico") == "lo statistico"
    assert _con_articolo("psicologo") == "lo psicologo"
    assert _con_articolo("guida turistica") == "la guida turistica"
    assert _con_articolo("agente immobiliare") == "l'agente immobiliare"


def test_prefissi_distinti():
    for genere in ("F", "M"):
        persona = Person("Nome Cognome", genere, {})
        for funzione in (prompt_ricco, prompt_en):
            prefissi = [funzione(persona, attr)[len("Nome Cognome "):] for attr in ATTRS]
            assert len(set(prefissi)) == len(ATTRS), prefissi
            for a in prefissi:
                for b in prefissi:
                    assert a == b or not b.startswith(a), (a, b)
