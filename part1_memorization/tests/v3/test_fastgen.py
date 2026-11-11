"""Test di src.v3.fastgen: cache statica e CUDA graph devono dare i token di kvgen (richiede cuda)."""
from __future__ import annotations

import pytest
import torch

from src.model import GPT
from src.v2 import kvgen
from src.v2 import corpus
from src.v2.spec import CORPORA
from src.v2.train import _evaluate, get_tokenizer
from src.v3.fastgen import StaticGen, evaluate_fast

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="richiede cuda")
VOCAB = 97
CTX = 64


def _modello(vocab: int = VOCAB) -> GPT:
    torch.manual_seed(0)
    return GPT(vocab_size=vocab, ctx_len=CTX, n_layer=2, n_embd=64, n_head=4).cuda().eval()


@cuda
@pytest.mark.parametrize("B", [1, 64])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("con_stop", [False, True])
def test_uguale_a_kvgen(B, graphs, con_stop):
    model = _modello()
    torch.manual_seed(1)
    idx = torch.randint(0, VOCAB, (B, 7), device="cuda")
    max_new = 12
    ref, _ = kvgen.generate(model, idx, max_new)
    # stop su un token che compare a meta' generazione, cosi' il troncamento viene esercitato
    stop = int(ref[0, 5]) if con_stop else None
    atteso, _ = kvgen.generate(model, idx, max_new, stop_id=stop)

    gen = StaticGen(model, max_batch=B, dtype=torch.float32, use_graphs=graphs)
    ottenuto, stats = gen.generate(idx, max_new, stop_id=stop)

    assert ottenuto.shape == atteso.shape
    assert torch.equal(ottenuto, atteso)
    assert set(stats) == {"prefill_s", "ttft_s", "decode_s", "n_steps"}
    assert stats["n_steps"] == atteso.shape[1] - 1


@cuda
def test_ripetuto_con_graph_stessa_risposta():
    # il secondo giro riusa i CUDA graph gia' catturati: stessi token
    model = _modello()
    idx = torch.randint(0, VOCAB, (4, 5), device="cuda")
    gen = StaticGen(model, max_batch=4, dtype=torch.float32)
    primo, _ = gen.generate(idx, 10)
    secondo, _ = gen.generate(idx, 10)
    assert torch.equal(primo, secondo)


@cuda
def test_valueerror_come_kvgen():
    model = _modello()
    idx = torch.randint(0, VOCAB, (2, CTX - 2), device="cuda")
    with pytest.raises(ValueError):
        kvgen.generate(model, idx, 4)
    with pytest.raises(ValueError):
        StaticGen(model, max_batch=2, dtype=torch.float32).generate(idx, 4)


@cuda
def test_evaluate_fast_come_train():
    persons = corpus.load_persons(200)
    spec = CORPORA["italiano_pieno"]
    tok = get_tokenizer(spec, persons)
    model = GPT(vocab_size=tok.get_vocab_size(), ctx_len=256, n_layer=2, n_embd=64, n_head=4).cuda()
    atteso = _evaluate(model, tok, spec, persons, 10)
    ottenuto = evaluate_fast(model, tok, spec, persons, 10)
    assert set(ottenuto) == set(atteso)
    assert abs(ottenuto["acc_mean"] - atteso["acc_mean"]) <= 0.02
