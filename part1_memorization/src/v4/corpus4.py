"""Corpora v4: formato pieno e canonico, con e senza stripping a lista; tokenizer ed epoche come v2."""
from __future__ import annotations

import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, EOS, PAD
from src.data import PIENO, Person, _maiuscola, _predicato
from src.v2 import corpus
from src.v2.spec import CORPORA, CorpusSpec, Part
from src.v2.train import _epoch_path, _tok_path
from src.v3 import realize
from src.v3.strip import ListStripper, _scrivi_epoca, strip_bio
from src.v4.spec4 import CANON_TEMPLATE, F_FORMATS, STRIPPATO, VOCAB, VOCAB_SIZE


def render_canonico(person: Person, rng: random.Random) -> str:
    """Biografia con un solo template per attributo e senza pronomi; il nome solo nella prima frase."""
    ordine = list(ATTRS)
    rng.shuffle(ordine)
    frasi = []
    for i, attr in enumerate(ordine):
        pred = _predicato(person, attr, PIENO[attr][CANON_TEMPLATE[attr]])
        frase = f"{person.name} {pred}" if i == 0 else _maiuscola(pred)
        frasi.append(frase + ".")
    return " ".join(frasi)


def _renderizza(style: str, person: Person, rng: random.Random) -> str:
    if style == "canonico":
        return render_canonico(person, rng)
    return corpus.render(person, style, "it", rng)


def _sorgente(style: str, persons: list[Person], epoch: int) -> list[str]:
    """Biografie dell'epoca nello stile sorgente, prima dello stripping."""
    if style == "pieno":
        return corpus.epoch_texts(CORPORA["italiano_pieno"], persons, epoch)
    rng = random.Random(f"italiano_canonico|{epoch}")
    testi = [render_canonico(p, rng) for p in persons]
    rng.shuffle(testi)
    return testi


def _epoca(fmt: str, persons: list[Person], epoch: int, cache: dict[str, str]) -> list[str]:
    style, stripper = F_FORMATS[fmt]
    testi = _sorgente(style, persons, epoch)
    if stripper == "lista":
        testi = [strip_bio(ListStripper(), t, cache) for t in testi]
    return testi


def epoch_texts(fmt: str, persons: list[Person], epoch: int) -> list[str]:
    """Biografie dell'epoca nel formato dato, con cache di stripping propria della chiamata."""
    return _epoca(fmt, persons, epoch, {})


def register_all() -> dict[str, CorpusSpec]:
    """Spec dei quattro formati; italiano_pieno esiste gia' in CORPORA, gli altri vengono registrati."""
    specs = {}
    for fmt, (style, _) in F_FORMATS.items():
        key = f"{fmt}@{VOCAB}"
        if fmt != "italiano_pieno":
            CORPORA[key] = CorpusSpec(key, (Part(style, "it", "all"),), key)
        specs[fmt] = CORPORA[key]
    return specs


def prompt(fmt: str, person: Person, attr: str) -> str:
    """Prompt della domanda: strippato a lista nei formati telegrafici."""
    domanda = realize.question(person, attr)
    return ListStripper().strip(domanda) if STRIPPATO[fmt] else domanda


def expected(fmt: str, person: Person, attr: str) -> str:
    """Valore atteso dopo il prompt: intero nei formati pieni, strippato in quelli telegrafici."""
    valore = person.attrs[attr]
    if not STRIPPATO[fmt]:
        return valore
    p = prompt(fmt, person, attr)
    s = ListStripper().strip(f"{realize.question(person, attr)} {valore}.")
    if not s.startswith(p):
        raise ValueError(f"prompt non prefisso dello strip: {p!r} / {s!r}")
    s = s[len(p):].strip()
    return (s[:-1] if s.endswith(".") else s).strip()


def examples(fmt: str, person: Person, k: int = 2) -> list[str]:
    """k biografie della stessa persona nello stile sorgente del formato, strippate se il formato lo prevede."""
    style, stripper = F_FORMATS[fmt]
    out = []
    for i in range(k):
        rng = random.Random(f"{style}|esempio|{i}")
        testo = _renderizza(style, person, rng)
        if stripper == "lista":
            testo = strip_bio(ListStripper(), testo)
        out.append(testo)
    return out


def _bpe(testi: list[str], spec: CorpusSpec, vocab_size: int) -> Tokenizer:
    """Stessa configurazione BPE di src.v3.strip._addestra_tokenizer, con vocab_size dato."""
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if spec.split_digits:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level])
    else:
        tok.pre_tokenizer = byte_level
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(testi, trainer=trainer)
    return tok


def build(fmt: str, persons: list[Person], n_epochs: int) -> None:
    """Scrive tokenizer ed epoche del formato, saltando i file gia' presenti; una epoca alla volta."""
    spec = register_all()[fmt]
    n = len(persons)
    cache: dict[str, str] = {}
    path_tok = _tok_path(spec, n)
    if path_tok.exists():
        tok = Tokenizer.from_file(str(path_tok))
    else:
        tok = _bpe(_epoca(fmt, persons, 0, cache), spec, VOCAB_SIZE)
        path_tok.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(path_tok))
    for e in range(n_epochs):
        path = _epoch_path(spec, n, e)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _scrivi_epoca(tok, _epoca(fmt, persons, e, cache), path)
