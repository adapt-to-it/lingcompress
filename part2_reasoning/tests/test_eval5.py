"""Test di src/eval5.py su CPU, con un modello Qwen3 casuale minuscolo e il tokenizer in cache."""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from src.eval5 import evaluate, generate, latency
from src.spec5 import LATENCY_N, MAX_NEW_TOKENS, OUTPUT_FIELDS

TOKENIZER = "Qwen/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    return Qwen3ForCausalLM(cfg).to("cpu").eval()


ROWS = [
    {"id": f"test-{i}", "question": f"What is {i} + {i}?", "answer": str(2 * i)}
    for i in range(5)
]


def test_generate_n_tokens_bounded(model, tok):
    out = generate(model, tok, [r["question"] for r in ROWS[:3]], "D", 2)
    assert len(out) == 3
    for text, n in out:
        assert isinstance(text, str)
        assert 1 <= n <= MAX_NEW_TOKENS["D"]


def test_evaluate_keys_and_file(model, tok, tmp_path):
    out_path = tmp_path / "out.jsonl"
    res = evaluate(model, tok, "D", ROWS[:3], batch=2, out_path=out_path)
    assert set(res) == {"n_eval", "accuracy", "conform_rate", "tok_mean", "tok_median",
                        "throughput_time_s", "answers_per_s", "tokens_per_s"}
    assert res["n_eval"] == 3
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["conform_rate"] <= 1.0

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line, row in zip(lines, ROWS[:3]):
        rec = json.loads(line)
        assert tuple(rec) == OUTPUT_FIELDS
        assert rec["id"] == row["id"]
        assert rec["gold"] == row["answer"]


def test_latency_sample_capped_and_deterministic(model, tok):
    summary, items = latency(model, tok, "D", ROWS)
    assert summary["n"] == min(LATENCY_N, len(ROWS)) == len(ROWS)
    assert len(items) == len(ROWS)
    assert {it["id"] for it in items} == {r["id"] for r in ROWS}
    for it in items:
        assert it["seconds"] >= 0
        assert 1 <= it["n_tokens"] <= MAX_NEW_TOKENS["D"]

    _, again = latency(model, tok, "D", ROWS)
    assert [it["id"] for it in again] == [it["id"] for it in items]
