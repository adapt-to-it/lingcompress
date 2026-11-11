"""Test di src/v2/kvgen.py su CPU fp32 con un GPT minuscolo casuale."""
import pytest
import torch

from src.model import GPT
from src.v2 import kvgen


@pytest.fixture
def modello():
    torch.manual_seed(0)
    return GPT(vocab_size=50, ctx_len=32, n_layer=2, n_embd=16, n_head=2).eval()


def test_stessi_token_di_generate_greedy(modello):
    torch.manual_seed(1)
    idx = torch.randint(0, 50, (3, 5))
    ref = modello.generate_greedy(idx, max_new_tokens=8)
    new, stats = kvgen.generate(modello, idx, max_new_tokens=8)
    assert new.shape == (3, 8)
    assert new.dtype == torch.long
    assert torch.equal(new, ref)
    assert stats["n_steps"] == 7


def test_stop_id_stessi_token_e_uscita_anticipata(modello):
    torch.manual_seed(2)
    idx = torch.randint(0, 50, (1, 4))
    libero = modello.generate_greedy(idx, max_new_tokens=10)[0].tolist()
    stop = libero[3]
    atteso_k = libero.index(stop) + 1  # primo passo in cui la riga emette lo stop
    assert atteso_k < 10
    ref = modello.generate_greedy(idx, max_new_tokens=10, stop_ids={stop})
    new, _ = kvgen.generate(modello, idx, max_new_tokens=10, stop_id=stop)
    assert new.size(1) == atteso_k
    assert torch.equal(new, ref[:, :atteso_k])


def test_stop_id_batch_come_generate_greedy(modello):
    torch.manual_seed(3)
    idx = torch.randint(0, 50, (3, 4))
    stop = int(modello.generate_greedy(idx, max_new_tokens=10)[1, 3])
    ref = modello.generate_greedy(idx, max_new_tokens=10, stop_ids={stop})
    new, _ = kvgen.generate(modello, idx, max_new_tokens=10, stop_id=stop)
    assert new.size(1) <= 10
    assert torch.equal(new, ref[:, :new.size(1)])


def test_stats_chiavi_e_tempi(modello):
    idx = torch.randint(0, 50, (2, 6))
    _, stats = kvgen.generate(modello, idx, max_new_tokens=4)
    assert set(stats) == {"prefill_s", "ttft_s", "decode_s", "n_steps"}
    assert 0 <= stats["prefill_s"] <= stats["ttft_s"]
    assert stats["decode_s"] >= 0


def test_prompt_oltre_ctx_solleva(modello):
    idx = torch.randint(0, 50, (1, 30))
    with pytest.raises(ValueError):
        kvgen.generate(modello, idx, max_new_tokens=5)


def test_flops_formula():
    prefill, decode = kvgen.flops(params_total=100, n_layer=2, n_embd=8, prompt_len=5, n_new=3)
    assert prefill == 2 * 100 * 5 + 2 * 2 * 8 * 5 ** 2  # 1800
    assert decode == (200 + 4 * 2 * 8 * 5) + (200 + 4 * 2 * 8 * 6) + (200 + 4 * 2 * 8 * 7)  # 1752
    assert kvgen.flops(100, 2, 8, 5, 0) == (1800, 0)
