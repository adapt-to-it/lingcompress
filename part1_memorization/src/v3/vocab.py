"""Fase D (v3): vocabolari di taglia variabile (BPE e a caratteri) e latenza di risposta."""
from __future__ import annotations

import time

import numpy as np
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, EOS, PAD
from src.data import Person
from src.model import GPT
from src.v2 import corpus
from src.v2.spec import CORPORA, CorpusSpec
from src.v2.train import _tok_path
from src.v3.fastgen import StaticGen

VOCABS = ("bpe512", "bpe1024", "bpe2048", "char")
WARMUP_CALLS = 20


def _controlla_vocab(vocab: str) -> None:
    if vocab not in VOCABS:
        raise ValueError(f"vocabolario sconosciuto: {vocab} (attesi {VOCABS})")


def register(fmt: str, vocab: str) -> CorpusSpec:
    """Aggiunge a CORPORA la variante f"{fmt}@{vocab}" con le parti di CORPORA[fmt]."""
    _controlla_vocab(vocab)
    base = CORPORA[fmt]
    nome = f"{fmt}@{vocab}"
    spec = CorpusSpec(nome, base.parts, nome, True, base.cross)
    CORPORA[nome] = spec
    return spec


def _bpe(testi: list[str], spec: CorpusSpec, vocab_size: int) -> Tokenizer:
    """Stessa costruzione di train.get_tokenizer, con la taglia richiesta."""
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
        # alfabeto completo dei 256 byte, cosi' ogni testo e' codificabile
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(testi, trainer=trainer)
    return tok


def _caratteri(testi: list[str]) -> Tokenizer:
    """Un token per carattere (EOS e PAD inclusi), nessun merge, nessun byte-level."""
    alfabeto = sorted(set("".join(testi)))
    vocab = {EOS: 0, PAD: 1}
    for car in alfabeto:
        vocab[car] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token=None))
    # ogni carattere e' un pre-token; [\s\S] include anche il a capo
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens([EOS, PAD])
    return tok


def build_tokenizer(spec: CorpusSpec, vocab: str, persons: list[Person]) -> Tokenizer:
    """Addestra il tokenizer di spec sulle epoche 0 e lo salva dove lo legge train.get_tokenizer."""
    _controlla_vocab(vocab)
    testi = list(corpus.epoch_texts(spec, persons, 0))
    if vocab == "char":
        p = spec.parts[0]   # alfabeto: testi dell'epoca 0 + superfici di tutti i valori canonici
        tok = _caratteri(testi + [corpus.surface(a, v, p.style, p.lang) for a in ATTRS for v in corpus._canonici(a)])
    else:
        tok = _bpe(testi, spec, int(vocab[3:]))
    path = _tok_path(spec, len(persons))
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def max_answer_tokens(spec: CorpusSpec, tok: Tokenizer) -> int:
    """Token massimi di una risposta (valori canonici con spazio iniziale) + 2."""
    style, lang = spec.parts[0].style, spec.parts[0].lang
    massimo = 0
    for attr in ATTRS:
        for valore in corpus._canonici(attr):
            superficie = corpus.surface(attr, valore, style, lang)
            massimo = max(massimo, len(tok.encode(" " + superficie).ids))
    return massimo + 2


@torch.no_grad()
def latency(model: GPT, tok: Tokenizer, spec: CorpusSpec, persons: list[Person],
            n_questions: int, batch: int) -> dict:
    """Latenza di risposta con StaticGen bf16 sulle prime n_questions domande "in".

    Ogni richiesta genera esattamente i token della sua risposta vera (oracolo, come v2 infer).
    batch=1: tempo per richiesta; batch>1: tempo per lotto di al piu' batch richieste.
    """
    device = next(model.parameters()).device
    eos = tok.token_to_id(EOS)
    casi = [c for c in corpus.eval_cases(spec, persons, len(persons)) if c.kind == "in"][:n_questions]
    if not casi:
        raise ValueError("nessuna domanda 'in' da misurare")
    ids = [[eos] + tok.encode(c.prompt).ids for c in casi]
    style, lang = spec.parts[0].style, spec.parts[0].lang
    # oracolo: token della superficie vera nello stile del corpus (non del valore canonico)
    risposte = [len(tok.encode(" " + corpus.surface(c.attr, c.truth, style, lang)).ids) for c in casi]

    # lotti con stesso prompt e stessa lunghezza di risposta: generazione senza padding, token esatti
    gruppi: dict[tuple[int, int], list[int]] = {}
    for i, (riga, k) in enumerate(zip(ids, risposte)):
        gruppi.setdefault((len(riga), k), []).append(i)
    lotti = [(indici[s:s + batch], k)
             for (_, k), indici in gruppi.items()
             for s in range(0, len(indici), batch)]

    gen = StaticGen(model, max_batch=batch, dtype=torch.bfloat16)

    def esegui(lotto: tuple[list[int], int]) -> None:
        indici, k = lotto
        idx = torch.tensor([ids[i] for i in indici], dtype=torch.long, device=device)
        gen.generate(idx, max_new_tokens=k)

    model.eval()
    # cache_enabled=False: i CUDA graph non devono riusare cast dei pesi catturati durante la capture
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False):
        for j in range(WARMUP_CALLS):
            esegui(lotti[j % len(lotti)])
        # un lotto per ogni dimensione di batch: i CUDA graph si catturano tutti fuori dalla misura
        for lotto in {len(l[0]): l for l in lotti}.values():
            esegui(lotto)
        tempi = []
        for lotto in lotti:
            t0 = time.perf_counter()
            esegui(lotto)
            tempi.append(time.perf_counter() - t0)

    latenza_ms = float(np.mean(tempi)) * 1000
    risposta_media = float(np.mean(risposte))
    return {
        "prompt_tokens_mean": float(np.mean([len(r) for r in ids])),
        "answer_tokens_mean": risposta_media,
        "latency_ms_mean": latenza_ms,
        "ms_per_token": latenza_ms / risposta_media,
        "answers_per_s": len(casi) / float(sum(tempi)),
    }
