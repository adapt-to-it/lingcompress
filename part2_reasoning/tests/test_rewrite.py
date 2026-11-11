"""Test di src/rewrite.py con generatore finto."""
import json

import pytest

from src import rewrite
from src.spec5 import FEWSHOT, FEWSHOT_IDS

# Soluzione ufficiale fittizia: passaggi 5 e 10, risposta 10.
ANSWER = "Two plus three is <<2+3=5>>5.\nThe total is <<5*2=10>>10.\n#### 10"
ITEM = {"id": "x-1", "question": "Two plus three, times two?", "answer": ANSWER}
ITEM2 = {"id": "x-2", "question": "Other?", "answer": ANSWER}
GOOD_A = "2 + 3 = 5.\n5 * 2 = 10.\nThe answer is 10."
GOOD_C = "a = 2 + 3 = 5\nb = 5 * 2 = 10\nANSWER = 10"

TRAIN = [{"id": fid, "question": f"Question {fid}", "answer": "Step <<1+1=2>>2.\n#### 2"} for fid in FEWSHOT_IDS]


class FakeGen:
    """gen_fn finto: reply(messages, temperature) per ogni richiesta; registra (n. richieste, temperatura)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, messages_list, temperature):
        self.calls.append((len(messages_list), temperature))
        return [self.reply(msgs, temperature) for msgs in messages_list]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_train(monkeypatch):
    monkeypatch.setattr(rewrite, "load_gsm8k", lambda split: TRAIN)


def test_build_messages_turns():
    by_id = {x["id"]: x for x in TRAIN}
    msgs = rewrite.build_messages("A", ITEM, by_id)
    assert msgs[0] == {"role": "system", "content": rewrite.GEN_SYSTEM["A"]}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * len(FEWSHOT_IDS) + ["user"]
    assert msgs[2]["content"] == FEWSHOT["A"][FEWSHOT_IDS[0]]
    assert msgs[-1]["content"] == rewrite.GEN_USER.format(question=ITEM["question"], answer=ANSWER)


def test_validate_a_ok():
    assert rewrite.validate("A", ITEM, GOOD_A) == (True, "")


def test_validate_a_format():
    assert rewrite.validate("A", ITEM, "The answer is 10.") == (False, "format")


def test_validate_a_answer():
    assert rewrite.validate("A", ITEM, "2 + 3 = 5.\n5 * 2 = 10.\nThe answer is 9.") == (False, "answer")


def test_validate_a_steps():
    text = "Two and three make ten.\nThe answer is 10."
    assert rewrite.validate("A", ITEM, text) == (False, "steps")
    assert rewrite.validate("A", ITEM, text, require_steps=False) == (True, "")


def test_validate_c_ok():
    assert rewrite.validate("C", ITEM, GOOD_C) == (True, "")


def test_validate_c_format():
    assert rewrite.validate("C", ITEM, "a = apples + 3 = 5\nANSWER = 10") == (False, "format")


def test_validate_c_arith():
    assert rewrite.validate("C", ITEM, "a = 2 + 3 = 6\nb = 6 * 2 = 12\nANSWER = 12") == (False, "arith")


def test_validate_c_answer():
    assert rewrite.validate("C", ITEM, "a = 2 + 3 = 5\nb = 5 * 2 = 10\nANSWER = 5") == (False, "answer")


def test_validate_c_steps():
    assert rewrite.validate("C", ITEM, "a = 6 + 4 = 10\nANSWER = 10") == (False, "steps")


def test_run_resumes_without_regenerating(tmp_path):
    out = tmp_path / "A.jsonl"
    gen = FakeGen(lambda msgs, t: GOOD_A)
    assert rewrite.run("A", [ITEM], gen, out) == {"": 1}
    assert gen.calls == [(1, 0)]
    again = FakeGen(lambda msgs, t: GOOD_A)
    assert rewrite.run("A", [ITEM], again, out) == {"": 1}
    assert again.calls == []
    assert len(_read_jsonl(out)) == 1


def test_run_retries_failed_at_temperature_0_7(tmp_path):
    out = tmp_path / "A.jsonl"
    gen = FakeGen(lambda msgs, t: GOOD_A if t > 0 else "The answer is 10.")
    assert rewrite.run("A", [ITEM], gen, out) == {"": 1}
    assert gen.calls == [(1, 0), (1, 0.7)]
    rows = _read_jsonl(out)
    assert [(r["attempt"], r["ok"], r["reason"]) for r in rows] == [(1, False, "format"), (2, True, "")]


def test_run_retries_only_failed_items(tmp_path):
    out = tmp_path / "A.jsonl"
    gen = FakeGen(lambda msgs, t: GOOD_A if msgs[-1]["content"].startswith("Problem: Two plus") or t > 0 else "bad")
    assert rewrite.run("A", [ITEM, ITEM2], gen, out) == {"": 2}
    assert gen.calls == [(2, 0), (1, 0.7)]


def test_run_gives_up_after_attempts(tmp_path):
    out = tmp_path / "A.jsonl"
    gen = FakeGen(lambda msgs, t: "bad")
    assert rewrite.run("A", [ITEM], gen, out, attempts=3) == {"format": 1}
    assert gen.calls == [(1, 0), (1, 0.7), (1, 0.7)]
    assert [r["attempt"] for r in _read_jsonl(out)] == [1, 2, 3]
    again = FakeGen(lambda msgs, t: GOOD_A)
    assert rewrite.run("A", [ITEM], again, out, attempts=3) == {"format": 1}
    assert again.calls == []


def test_run_batches(tmp_path):
    out = tmp_path / "A.jsonl"
    gen = FakeGen(lambda msgs, t: GOOD_A)
    assert rewrite.run("A", [ITEM, ITEM2], gen, out, batch=1) == {"": 2}
    assert gen.calls == [(1, 0), (1, 0)]


def test_run_rejects_wrong_number_of_texts(tmp_path):
    with pytest.raises(ValueError):
        rewrite.run("A", [ITEM, ITEM2], lambda msgs, t: [GOOD_A], tmp_path / "A.jsonl")


def test_build_sft_intersection(tmp_path, monkeypatch):
    rew = tmp_path / "rewrites"
    sft = tmp_path / "sft"
    monkeypatch.setattr(rewrite, "REWRITES", rew)
    monkeypatch.setattr(rewrite, "SFT", sft)
    train = [
        {"id": "train-0", "question": "Q0", "answer": "Step <<1+1=2>>2\n#### 72"},
        {"id": "train-1", "question": "Q1", "answer": "#### 3"},
        {"id": "train-3", "question": "Q3", "answer": "#### 4"},
    ]
    test = [{"id": "test-1", "question": "Q test", "answer": "x <<1+6=7>>7\n#### 7"}]
    monkeypatch.setattr(rewrite, "load_gsm8k", lambda split: train if split == "train" else test)
    _write_jsonl(rew / "A.jsonl", [
        {"id": "train-0", "text": "The answer is 72.", "ok": True, "reason": "", "attempt": 1},
        {"id": "train-1", "text": "The answer is 3.", "ok": True, "reason": "", "attempt": 1},
        {"id": "train-3", "text": "The answer is 4.", "ok": True, "reason": "", "attempt": 1},
        {"id": "train-3", "text": "bad", "ok": False, "reason": "answer", "attempt": 2},
    ])
    _write_jsonl(rew / "C.jsonl", [
        {"id": "train-0", "text": "ANSWER = 72", "ok": True, "reason": "", "attempt": 1},
        {"id": "train-1", "text": "ANSWER = 3", "ok": False, "reason": "arith", "attempt": 1},
        {"id": "train-3", "text": "ANSWER = 4", "ok": True, "reason": "", "attempt": 1},
    ])
    assert rewrite.build_sft() == {"A": 1, "B": 1, "C": 1, "D": 1, "test": 1}
    head = {"id": "train-0", "question": "Q0", "answer": "72"}
    assert _read_jsonl(sft / "A_train.jsonl") == [{**head, "solution": "The answer is 72."}]
    assert _read_jsonl(sft / "B_train.jsonl") == [{**head, "solution": "answer 72."}]
    assert _read_jsonl(sft / "C_train.jsonl") == [{**head, "solution": "ANSWER = 72"}]
    assert _read_jsonl(sft / "D_train.jsonl") == [{**head, "solution": "72"}]
    assert _read_jsonl(sft / "test.jsonl") == [{"id": "test-1", "question": "Q test", "solution": "", "answer": "7"}]
