"""Test di src.v3.vocab: registrazione, tokenizer, lunghezze di risposta e latenza."""
from __future__ import annotations

import pytest
import torch

from src.data import generate_persons
from src.v2 import corpus, train, spec as spec_v2
from src.v2.model2 import GPT2
from src.v2.spec import CORPORA, DATA_SEED
from src.v3 import vocab

N_PERSONS = 100


@pytest.fixture
def persone():
    return generate_persons(N_PERSONS, DATA_SEED)


@pytest.fixture
def v2_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "V2_DATA", tmp_path)
    monkeypatch.setattr(spec_v2, "V2_DATA", tmp_path)
    return tmp_path


@pytest.fixture
def registra():
    creati = []

    def _registra(fmt, voc):
        spec = vocab.register(fmt, voc)
        creati.append(spec.name)
        return spec

    yield _registra
    for nome in creati:
        CORPORA.pop(nome, None)


def test_char_roundtrip_and_vocab_size(v2_tmp, persone, registra):
    spec = registra("italiano_pieno", "char")
    tok = vocab.build_tokenizer(spec, "char", persone)
    testi = corpus.epoch_texts(spec, persone, 0)
    alfabeto = set("".join(testi))
    assert tok.get_vocab_size() >= len(alfabeto) + 2  # + caratteri dei valori canonici
    for t in testi[:50]:
        assert tok.decode(tok.encode(t).ids) == t


def test_bpe512_vocab_size(v2_tmp, persone, registra):
    spec = registra("italiano_pieno", "bpe512")
    tok = vocab.build_tokenizer(spec, "bpe512", persone)
    assert tok.get_vocab_size() == 512


def test_max_answer_tokens_char_larger_than_bpe512(v2_tmp, persone, registra):
    spec_char = registra("italiano_pieno", "char")
    spec_bpe = registra("italiano_pieno", "bpe512")
    tok_char = vocab.build_tokenizer(spec_char, "char", persone)
    tok_bpe = vocab.build_tokenizer(spec_bpe, "bpe512", persone)
    assert vocab.max_answer_tokens(spec_char, tok_char) > vocab.max_answer_tokens(spec_bpe, tok_bpe)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="latency misurata solo su cuda")
def test_latency_positive_on_cuda(v2_tmp, persone, registra):
    torch.manual_seed(0)
    spec = registra("italiano_pieno", "bpe512")
    tok = vocab.build_tokenizer(spec, "bpe512", persone)
    model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=512, n_layer=2, n_embd=64, n_head=4).cuda()
    for batch in (1, 64):
        r = vocab.latency(model, tok, spec, persone, 20, batch)
        assert set(r) == {"prompt_tokens_mean", "answer_tokens_mean", "latency_ms_mean",
                          "ms_per_token", "answers_per_s"}
        assert all(v > 0 for v in r.values())
