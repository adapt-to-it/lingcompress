"""Stripper v3: rimozione di articoli, preposizioni, ausiliari e pronomi soggetto dalle biografie."""
from __future__ import annotations

import functools
import os
import re

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import EOS, PAD
from src.data import Person
from src.v2 import corpus
from src.v2.spec import CORPORA, VOCAB_SIZE, CorpusSpec, Part
from src.v2.train import POOL_MAX, _epoch_path, _tok_path
from src.v3.spec3 import SPACY_MODEL, STRIP_WORDS

# parola: eventuale apostrofo finale ("l'", "un'"), oppure singolo carattere non spazio (punteggiatura)
_TOKEN = re.compile(r"\w+['’]|\w+|[^\w\s]")
_SPACY_DROP_POS = frozenset({"DET", "AUX", "ADP"})
_SPACY_DROP_DEP = frozenset({"nsubj", "nsubj:pass"})


def _punteggiatura(token: str) -> bool:
    return not any(c.isalnum() for c in token)


def _tenuti(analisi: list[tuple[str, int, int, bool]]) -> list[tuple[str, int, int]]:
    return [(tok, inizio, fine) for tok, inizio, fine, tenuto in analisi if tenuto]


def _unisci(parole: list[tuple[str, int, int]]) -> str:
    """Token uniti da uno spazio; la punteggiatura resta attaccata al token precedente."""
    testo = ""
    for tok, _, _ in parole:
        if testo and not _punteggiatura(tok):
            testo += " "
        testo += tok
    return testo


@functools.lru_cache(maxsize=None)
def _nlp():
    import spacy  # importato solo quando serve: il modello e' pesante

    return spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])


def _da_doc(doc) -> list[tuple[str, int, int, bool]]:
    analisi = []
    for t in doc:
        if t.is_space:
            continue
        eliminato = (t.pos_ in _SPACY_DROP_POS
                     or (t.pos_ == "PRON" and t.dep_ in _SPACY_DROP_DEP))
        analisi.append((t.text, t.idx, t.idx + len(t.text), not eliminato))
    return analisi


class _Stripper:
    name: str

    def _analizza(self, text: str) -> list[tuple[str, int, int, bool]]:
        """Tutti i token del testo con (token, inizio, fine, tenuto)."""
        raise NotImplementedError

    def spans(self, text: str) -> list[tuple[str, int, int]]:
        return _tenuti(self._analizza(text))

    def strip(self, text: str) -> str:
        return _unisci(self.spans(text))

    def strip_many(self, texts: list[str], n_process: int = 1) -> list[str]:
        return [self.strip(t) for t in texts]


class ListStripper(_Stripper):
    """Elimina le parole di STRIP_WORDS (confronto in minuscolo); tiene tutto il resto."""
    name = "lista"

    def _analizza(self, text: str) -> list[tuple[str, int, int, bool]]:
        analisi = []
        for m in _TOKEN.finditer(text):
            parola = m.group().lower().replace("’", "'")
            analisi.append((m.group(), m.start(), m.end(), parola not in STRIP_WORDS))
        return analisi


class SpacyStripper(_Stripper):
    """Elimina DET, AUX, ADP e i PRON con dipendenza nsubj o nsubj:pass (modello SPACY_MODEL)."""
    name = "spacy"

    def _analizza(self, text: str) -> list[tuple[str, int, int, bool]]:
        return _da_doc(_nlp()(text))

    def strip_many(self, texts: list[str], n_process: int = 1) -> list[str]:
        return [_unisci(_tenuti(_da_doc(doc)))
                for doc in _nlp().pipe(texts, n_process=n_process)]


def strip_bio(stripper: _Stripper, bio: str, cache: dict[str, str] | None = None) -> str:
    """Strip frase per frase: la biografia si divide sui ". " e ogni frase viene strippata una volta sola."""
    if cache is None:
        cache = {}
    frasi = []
    for frase in bio.split(". "):
        if frase not in cache:
            cache[frase] = stripper.strip(frase)
        if cache[frase]:
            frasi.append(cache[frase])
    return ". ".join(frasi)


def register(name: str) -> CorpusSpec:
    """Aggiunge a CORPORA il corpus `name`: una Part pieno/it/all, tokenizer con la stessa chiave."""
    spec = CorpusSpec(name, (Part("pieno", "it", "all"),), name)
    CORPORA[name] = spec
    return spec


def _addestra_tokenizer(spec: CorpusSpec, testi: list[str]) -> Tokenizer:
    """Stessa configurazione BPE di src.v2.train.get_tokenizer, ma addestrato sui testi dati."""
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if spec.split_digits:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level])
    else:
        tok.pre_tokenizer = byte_level
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(testi, trainer=trainer)
    return tok


def _scrivi_epoca(tok: Tokenizer, testi: list[str], out_path) -> None:
    """Stesso formato npz di src.v2.train._genera_epoca."""
    eos = tok.token_to_id(EOS)
    codificate = tok.encode_batch(testi)
    biografie = [[eos, *enc.ids, eos] for enc in codificate]
    lunghezze = np.array([len(b) for b in biografie], dtype=np.int64)
    offsets = np.zeros(len(biografie) + 1, dtype=np.int64)
    np.cumsum(lunghezze, out=offsets[1:])
    tokens = np.fromiter((t for b in biografie for t in b), dtype=np.uint16, count=int(offsets[-1]))
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, tokens=tokens, offsets=offsets,
                 tokens_per_bio=np.float64(lunghezze.mean() - 2),  # senza EOS
                 max_len=np.int64(lunghezze.max()))                 # con EOS
    os.replace(tmp, out_path)


def build_corpus(name: str, stripper: _Stripper, persons: list[Person], n_epochs: int) -> None:
    """Scrive tokenizer ed epoche npz (formato e percorsi v2) del corpus strippato `name`."""
    if n_epochs < 1:
        raise ValueError("n_epochs deve essere almeno 1")
    spec = register(name)
    n = len(persons)
    epoche = [corpus.epoch_texts(CORPORA["italiano_pieno"], persons, e) for e in range(n_epochs)]

    # ogni frase distinta, su tutte le epoche, viene strippata una sola volta
    distinte = sorted({frase for epoca in epoche for bio in epoca for frase in bio.split(". ")})
    cache = dict(zip(distinte, stripper.strip_many(
        distinte, n_process=min(POOL_MAX, os.cpu_count() or 1))))
    strippate = [[strip_bio(stripper, bio, cache) for bio in epoca] for epoca in epoche]

    tok = _addestra_tokenizer(spec, strippate[0])
    path_tok = _tok_path(spec, n)
    path_tok.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path_tok))
    _epoch_path(spec, n, 0).parent.mkdir(parents=True, exist_ok=True)
    for e, testi in enumerate(strippate):
        _scrivi_epoca(tok, testi, _epoch_path(spec, n, e))


def _scartate(stripper: _Stripper, text: str) -> set[str]:
    """Parole (minuscole, senza punteggiatura) che lo stripper elimina dal testo."""
    return {tok.lower() for tok, _, _, tenuto in stripper._analizza(text)
            if not tenuto and not _punteggiatura(tok)}


def compare(persons: list[Person], n: int = 200) -> pd.DataFrame:
    """Prime n biografie di pieno epoca 0: testi pieno/spacy/lista, conteggi di parole e parole scartate da un solo stripper."""
    testi = corpus.epoch_texts(CORPORA["italiano_pieno"], persons, 0)[:n]
    spacy_s, lista_s = SpacyStripper(), ListStripper()
    spacy_out = spacy_s.strip_many(testi)
    lista_out = lista_s.strip_many(testi)
    righe = []
    for pieno, sp, li in zip(testi, spacy_out, lista_out):
        scartate_sp = _scartate(spacy_s, pieno)
        scartate_li = _scartate(lista_s, pieno)
        righe.append({
            "pieno": pieno,
            "spacy": sp,
            "lista": li,
            "parole_pieno": len(pieno.split()),
            "parole_spacy": len(sp.split()),
            "parole_lista": len(li.split()),
            "solo_spacy": " ".join(sorted(scartate_sp - scartate_li)),
            "solo_lista": " ".join(sorted(scartate_li - scartate_sp)),
        })
    return pd.DataFrame(righe)
