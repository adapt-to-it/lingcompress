import math

import pytest
import torch

from src.model import GPT


def make_model(vocab=64, ctx=32, n_layer=2, n_embd=32, n_head=4):
    torch.manual_seed(0)
    return GPT(vocab_size=vocab, ctx_len=ctx, n_layer=n_layer, n_embd=n_embd, n_head=n_head)


def test_forward_shapes_and_loss_finite():
    model = make_model()
    idx = torch.randint(0, 64, (3, 16))
    targets = torch.randint(0, 64, (3, 16))
    targets[0, :4] = -100

    logits, loss = model(idx)
    assert logits.shape == (3, 16, 64)
    assert loss is None

    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, 64)
    assert loss is not None
    assert torch.isfinite(loss)


def test_forward_rejects_longer_than_ctx():
    model = make_model(ctx=8)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_non_embedding_params_close_to_12_L_d2():
    model = GPT(vocab_size=4096, ctx_len=128, n_layer=4, n_embd=72, n_head=4)
    non_emb, total = model.count_params()
    expected = 12 * 4 * 72 ** 2
    assert abs(non_emb - expected) / expected < 0.05
    assert total > non_emb


def test_tied_weights_counted_once():
    model = make_model()
    assert model.lm_head.weight is model.tok_emb.weight

    # conteggio manuale dei tensori unici, il peso condiviso una sola volta
    unique = {id(p): p for p in model.parameters()}
    manual_total = sum(p.numel() for p in unique.values())
    non_emb, total = model.count_params()
    assert total == manual_total
    assert non_emb == total - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()


def test_generate_greedy_shape():
    model = make_model(ctx=16)
    model.eval()
    idx = torch.randint(0, 64, (2, 5))
    out = model.generate_greedy(idx, max_new_tokens=20)
    assert out.shape == (2, 20)
    assert out.dtype == torch.long


def test_generate_greedy_crops_context():
    model = make_model(ctx=8)
    idx = torch.randint(0, 64, (2, 12))
    out = model.generate_greedy(idx, max_new_tokens=5)
    assert out.shape == (2, 5)


def test_generate_greedy_stops_early_and_pads():
    model = make_model()
    idx = torch.randint(0, 64, (2, 4))
    # ogni token e uno stop id: tutte le righe terminano al primo passo
    out = model.generate_greedy(idx, max_new_tokens=7, stop_ids=set(range(64)))
    assert out.shape == (2, 7)
    assert torch.all(out == out[:, :1])


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    model = GPT(vocab_size=32, ctx_len=16, n_layer=2, n_embd=32, n_head=2)
    idx = torch.randint(0, 32, (4, 16))
    targets = torch.randint(0, 32, (4, 16))
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    loss = None
    for _ in range(200):
        _, loss = model(idx, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    assert loss.item() < 0.5
