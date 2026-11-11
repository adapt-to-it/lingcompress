"""Test di src/eval6.py su CPU, con un modello Qwen3 casuale minuscolo, il tokenizer in cache e un processor che forza EOS."""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, LogitsProcessor, Qwen3Config, Qwen3ForCausalLM

from src.eval6 import evaluate6, generate6, latency6
from src.grammar6 import check_c6
from src.spec5 import LATENCY_N, MAX_NEW_TOKENS
from src.spec6 import OUTPUT6_FIELDS

TOKENIZER = "Qwen/Qwen3-0.6B-Base"
AFTER = 3                                   # il processor forza EOS dopo 3 token generati


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    return Qwen3ForCausalLM(cfg).to("cpu").eval()


class ForceEOS(LogitsProcessor):
    """Dopo AFTER token generati forza EOS e conta i token forzati per riga."""

    def __init__(self, eos, prompt_len, n_rows):
        self.eos = eos
        self.prompt_len = prompt_len
        self.n_forced = [0] * n_rows

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_len == AFTER:
            scores[:] = float("-inf")
            scores[:, self.eos] = 0.0
            self.n_forced = [k + 1 for k in self.n_forced]
        return scores


def factory_for(tok, calls=None):
    def build(prompt_len, n_rows):
        if calls is not None:
            calls.append(n_rows)
        return ForceEOS(tok.eos_token_id, prompt_len, n_rows)
    return build


ROWS = [
    {"id": f"test-{i}", "question": f"What is {i} + {i}?", "answer": str(2 * i)}
    for i in range(5)
]

CANON_C6 = ("NOTE each ship doubles\n"
            "GIVEN total_n = 10\n"
            "SOLVE x_n: x_n + 1 = 10 -> x_n = 9\n"
            "ANSWER = 9")


def test_generate6_n_forced_reported(model, tok):
    out = generate6(model, tok, [r["question"] for r in ROWS[:3]], "C6", 2, factory_for(tok))
    assert len(out) == 3
    for text, n, n_forced in out:
        assert isinstance(text, str)
        assert n == AFTER + 1
        assert n_forced == 1


def test_generate6_without_processor(model, tok):
    out = generate6(model, tok, [r["question"] for r in ROWS[:2]], "A", 2)
    assert len(out) == 2
    for text, n, n_forced in out:
        assert isinstance(text, str)
        assert 1 <= n <= MAX_NEW_TOKENS["A"]
        assert n_forced == 0


def test_generate6_unknown_fmt(model, tok):
    with pytest.raises(ValueError):
        generate6(model, tok, [ROWS[0]["question"]], "D", 1)


def test_evaluate6_keys_and_file(model, tok, tmp_path):
    out_path = tmp_path / "sub" / "out.jsonl"
    res = evaluate6(model, tok, "C6", ROWS[:3], batch=2, processor_factory=factory_for(tok), out_path=out_path)
    assert set(res) == {"n_eval", "accuracy", "conform_rate", "tok_mean", "tok_median", "forced_mean",
                        "throughput_time_s", "answers_per_s", "tokens_per_s", "note_rate", "solve_rate"}
    assert res["n_eval"] == 3
    assert res["forced_mean"] == 1.0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["conform_rate"] <= 1.0
    assert 0.0 <= res["note_rate"] <= 1.0
    assert 0.0 <= res["solve_rate"] <= 1.0

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line, row in zip(lines, ROWS[:3]):
        rec = json.loads(line)
        assert tuple(rec) == OUTPUT6_FIELDS
        assert rec["id"] == row["id"]
        assert rec["gold"] == row["answer"]
        assert rec["n_forced"] == 1
        assert rec["conform"] == check_c6(rec["output"], arithmetic=False)[0]


def test_evaluate6_without_processor_c_metrics_zero(model, tok):
    res = evaluate6(model, tok, "A", ROWS[:2], batch=2)
    assert res["forced_mean"] == 0.0
    assert res["note_rate"] == 0.0
    assert res["solve_rate"] == 0.0


def test_evaluate6_c6_conformity_and_rates(model, tok, monkeypatch):
    texts = [(CANON_C6, 20, 0), ("garbage", 5, 0)]
    monkeypatch.setattr("src.eval6.generate6", lambda *a, **k: texts)
    res = evaluate6(model, tok, "C6", ROWS[:2])
    assert res["conform_rate"] == 0.5
    assert res["note_rate"] == 0.5
    assert res["solve_rate"] == 0.5
    assert res["forced_mean"] == 0.0


def test_evaluate6_rejects_empty(model, tok):
    with pytest.raises(ValueError):
        evaluate6(model, tok, "A", [])


def test_latency6_factory_per_call(model, tok):
    calls = []
    summary, items = latency6(model, tok, "C5", ROWS, processor_factory=factory_for(tok, calls))
    assert summary["n"] == min(LATENCY_N, len(ROWS)) == len(ROWS)
    assert {it["id"] for it in items} == {r["id"] for r in ROWS}
    for it in items:
        assert it["seconds"] >= 0
        assert it["n_tokens"] == AFTER + 1
    # un processor per chiamata di generate: warm-up piu' un campione per riga
    assert len(calls) == 1 + len(ROWS)
    assert set(summary) == {"n", "lat_mean_s", "lat_median_s", "ms_per_token_mean", "tok_mean"}
