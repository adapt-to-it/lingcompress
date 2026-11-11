"""Test di src/train5.py su CPU, con un modello Qwen3 casuale minuscolo e il tokenizer in cache."""
import json
import os
import statistics

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from src.spec5 import PROMPT
from src.train5 import encode, run_id, train

TOKENIZER = "Qwen/Qwen3-0.6B-Base"

ROW = {"id": "test-0", "question": "What is 2 + 3?", "solution": "2 + 3 = 5.\nThe answer is 5.", "answer": "5"}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    return Qwen3ForCausalLM(cfg).to("cpu")


def test_run_id():
    assert run_id("small", "A", 2e-4, 0) == "small_A_lr0.0002_s0"


def test_encode_masks_prompt(tok):
    ids, labels = encode(tok, ROW, 640)
    prompt_len = len(tok(PROMPT.format(question=ROW["question"]), add_special_tokens=False)["input_ids"])
    assert len(ids) == len(labels)
    assert labels[:prompt_len] == [-100] * prompt_len
    assert labels[prompt_len:] == ids[prompt_len:]
    assert labels[-1] == tok.eos_token_id


def test_encode_truncates(tok):
    ids, labels = encode(tok, ROW, 5)
    assert len(ids) == len(labels) == 5


def test_train_loss_decreases_and_writes_outputs(model, tok, tmp_path):
    rows = [dict(ROW, id=f"test-{i}") for i in range(8)]
    out_dir = tmp_path / "adapter"
    res = train("small", "A", 1e-3, 0, rows=rows, device="cpu", model=model, tok=tok,
                out_dir=out_dir, max_steps=30)

    assert res["steps"] == 30
    assert res["n_train"] == 8
    assert len(res["losses"]) == 30
    assert statistics.mean(res["losses"][-5:]) < statistics.mean(res["losses"][:5])

    assert (out_dir / "adapter_config.json").exists()
    log = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert set(log) == {"run_id", "model", "format", "lr", "seed", "n_train", "steps", "train_time_s",
                        "final_loss", "losses"}
    assert log["run_id"] == run_id("small", "A", 1e-3, 0)
    assert log["final_loss"] == pytest.approx(res["final_loss"])
