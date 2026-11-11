"""Realizzatore a regole v3: dalla domanda strippata o dal prompt interlingua alla frase pieno."""
from __future__ import annotations

from src.config import ATTRS, INTERLINGUA_MARKERS
from src.data import FORME_GENERE, NOMI_F, NOMI_M, PIENO, Person
from src.v2 import corpus
from src.v3.spec3 import E_QUESTION_TEMPLATE
from src.v3.strip import _Stripper

_F = frozenset(NOMI_F)
_M = frozenset(NOMI_M)
_ATTR_DA_MARCATORE = {marcatore: attr for attr, marcatore in INTERLINGUA_MARKERS.items()}
# Nomi campione per il contesto con cui lo stripper tratta i valori (uno per genere)
_NOME_CAMPIONE = {"F": "Marta Rinaldi", "M": "Marco Rinaldi"}


def gender(name: str) -> str:
    """Genere ("F" | "M") dal primo nome; un nome sconosciuto da' "M"."""
    parole = name.split()
    if parole and parole[0] in _F:
        return "F"
    return "M"


def _prefisso_pieno(attr: str, genere: str, indice: int) -> str:
    """Prefisso del template di indice dato, senza nome e senza valore, nel genere dato."""
    return PIENO[attr][indice].split("{v}")[0].strip().format(**FORME_GENERE[genere])


def _domanda(name: str, attr: str, genere: str) -> str:
    return f"{name} {_prefisso_pieno(attr, genere, E_QUESTION_TEMPLATE[attr])}"


def _coda_in_ordine(parole: list[str], prefisso: list[str]) -> int:
    """Lunghezza della coda piu' lunga di `parole` che compare in `prefisso` nello stesso ordine."""
    j = 0
    for k in range(1, len(parole) + 1):
        it = iter(prefisso)
        if not all(p.lower() in it for p in parole[-k:]):
            break
        j = k
    return j


def question(person: Person, attr: str) -> str:
    """Domanda pieno senza valore: nome e prefisso del template di E_QUESTION_TEMPLATE."""
    return _domanda(person.name, attr, person.gender)


class TeleRealizer:
    """Realizza le risposte del formato telegrafico strippato con tabelle di valori e prefissi."""

    def __init__(self, stripper: _Stripper):
        self._stripper = stripper
        # parole grezze del prefisso: lo stripper ne elimina alcune, in base al contesto
        self._prefissi = [(attr, _prefisso_pieno(attr, genere, E_QUESTION_TEMPLATE[attr]).lower().split())
                          for attr in ATTRS for genere in ("F", "M")]
        self._tabelle = {}
        for attr in ATTRS:
            domanda = _domanda(_NOME_CAMPIONE["F"], attr, "F")
            self._tabelle[attr] = {self._dopo(domanda, f"{canonico}."): canonico
                                   for canonico in corpus._canonici(attr)}

    def _dopo(self, prima: str, coda: str) -> str:
        """Token di `coda` che seguono `prima` nel contesto dello stripper; il punto finale va via."""
        inizio = len(prima) + 1
        tokens = [tok for tok, start, _ in self._stripper.spans(f"{prima} {coda}")
                  if start >= inizio]
        if tokens and tokens[-1] == ".":
            tokens.pop()
        return " ".join(tokens)

    def parse(self, domanda_strippata: str) -> tuple[str, str | None, str]:
        """(nome, attributo o None, genere): il prefisso e' la coda piu' lunga, in ordine nel template."""
        parole = domanda_strippata.split()
        migliore = (0, None)
        for attr, prefisso in self._prefissi:
            j = _coda_in_ordine(parole, prefisso)
            if migliore[0] < j < len(parole):
                migliore = (j, attr)
        if migliore[1] is None:
            name = " ".join(parole)
            return name, None, gender(name)
        j, attr = migliore
        name = " ".join(parole[:-j])
        return name, attr, gender(name)

    def realize(self, domanda_strippata: str, generato: str) -> tuple[str | None, str | None]:
        """(frase pieno, valore canonico) dalla domanda strippata e dal valore generato; (None, None) se non riconosciuta."""
        name, attr, genere = self.parse(domanda_strippata)
        if attr is None:
            return None, None
        valore = generato.split(".", 1)[0].strip()
        canonico = self._tabelle[attr].get(valore)
        if canonico is None:
            canonico = self._piu_simile(valore, attr)
        testo = canonico if canonico is not None else valore
        return f"{_domanda(name, attr, genere)} {testo}.", canonico

    def _piu_simile(self, valore: str, attr: str) -> str | None:
        """Valore canonico con piu' parole in comune con `valore` (confronto sui valori strippati)."""
        parole = set(valore.lower().split())
        migliore, punteggio = None, 0
        for strippato, canonico in self._tabelle[attr].items():
            n = len(parole & set(strippato.lower().split()))
            if n > punteggio:
                migliore, punteggio = canonico, n
        return migliore


class InterRealizer:
    """Realizza le risposte dell'interlingua: prompt "Nome_Cognome MARCATORE", valore in superficie."""

    def __init__(self):
        prefissi = [(_prefisso_pieno(attr, genere, indice), attr)
                    for attr in ATTRS
                    for genere in ("F", "M")
                    for indice in range(len(PIENO[attr]))]
        self._prefissi = sorted(prefissi, key=lambda p: len(p[0]), reverse=True)

    def to_prompt(self, domanda: str) -> str:
        """Prompt interlingua dalla domanda pieno; l'attributo viene dal prefisso di un template."""
        testo = domanda.strip()
        for prefisso, attr in self._prefissi:
            if testo.endswith(" " + prefisso):
                name = testo[:-len(prefisso)].strip()
                return f"{name.replace(' ', '_')} {INTERLINGUA_MARKERS[attr]}"
        raise ValueError(f"domanda non riconosciuta: {domanda}")

    def realize(self, inter_prompt: str, generato: str) -> tuple[str | None, str | None]:
        """(frase pieno, valore canonico) dal prompt interlingua e dal testo generato; (None, None) se non riconosciuto."""
        nome, marcatore = inter_prompt.rsplit(" ", 1)
        attr = _ATTR_DA_MARCATORE[marcatore]
        name = nome.replace("_", " ")
        canonico = corpus.parse_answer(generato, attr, "interlingua", "xx")
        if canonico is None:
            return None, None
        return f"{_domanda(name, attr, gender(name))} {canonico}.", canonico
