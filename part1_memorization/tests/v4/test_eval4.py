"""Test di src.v4.eval4: valutazione per formato con GPT2 minuscolo e tokenizer di prova, su CPU."""
from __future__ import annotations

import math

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import ATTRS, EOS, PAD
from src.data import generate_persons
from src.v2.model2 import GPT2
from src.v2.spec import CORPORA
from src.v4 import corpus4, eval4


def _tokenizer(testi: list[str]) -> Tokenizer:
    """Stessa configurazione BPE dei corpora v4, con vocabolario piccolo."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(testi, trainer=trainer)
    return tok


def test_make_evaluate_chiavi_e_intervalli():
    persone = generate_persons(20, 1)
    for fmt in ("italiano_pieno", "telegrafico_lista"):
        tok = _tokenizer(corpus4.epoch_texts(fmt, persone, 0))
        torch.manual_seed(0)
        model = GPT2(vocab_size=tok.get_vocab_size(), ctx_len=256, n_layer=2, n_embd=32, n_head=4)
        risultato = eval4.make_evaluate(fmt)(model, tok, CORPORA["italiano_pieno"], persone, 5)

        attese = {f"acc_{a}" for a in ATTRS} | {f"cross_{a}" for a in ATTRS} | \
            {"acc_mean", "cross_mean", "cross_lenient_mean"}
        assert set(risultato) == attese
        for a in ATTRS:
            assert 0 <= risultato[f"acc_{a}"] <= 1
            assert math.isnan(risultato[f"cross_{a}"])
        assert 0 <= risultato["acc_mean"] <= 1
        assert math.isnan(risultato["cross_mean"])
