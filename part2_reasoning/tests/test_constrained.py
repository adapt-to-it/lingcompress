"""Test di src/constrained.py: punteggi finti (nessun modello), tokenizer Qwen in cache."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer

from src.constrained import ReasoningProcessor, _a_calc_value, audit_a_trigger, make_processor

MODEL = "Qwen/Qwen3-0.6B-Base"
SOLVE_LINE = ("SOLVE first_ship_n: first_ship_n + 2 * first_ship_n + 4 * first_ship_n = 847 -> first_ship_n =")


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


def enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def new_proc(tok, mode, grammar, calc, batch=1):
    prompt = enc(tok, "Solve.\n")
    ids = torch.tensor([prompt] * batch)
    return ReasoningProcessor(tok, mode, grammar, calc, len(prompt), batch), ids


def advance(tok, proc, ids, wants):
    """Un passo greedy: wants[b] ha il logit piu' alto della riga b; restituisce (ids estesi, id scelti)."""
    scores = torch.zeros(ids.shape[0], len(tok))
    for b, w in enumerate(wants):
        scores[b, w] = 1.0
    chosen = proc(ids, scores).argmax(dim=-1)
    return torch.cat([ids, chosen.view(-1, 1)], dim=1), chosen


def feed(tok, proc, ids, text):
    """Fa scegliere al processore i token di text uno per uno: ognuno deve essere accettato."""
    for t in enc(tok, text):
        ids, chosen = advance(tok, proc, ids, [t])
        assert chosen.tolist() == [t], repr(tok.decode([t]))
    return ids


def test_make_processor_builds_factory(tok):
    proc = make_processor(tok, "C6", True, True)(7, 2)
    assert isinstance(proc, ReasoningProcessor)
    assert proc.n_forced == [0, 0]


def test_c5_rejects_prose_at_line_start(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    hello = enc(tok, "Hello")[0]
    _, chosen = advance(tok, proc, ids, [hello])
    assert chosen.item() != hello


def test_c5_rejects_space_after_lowercase_word(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    ids = feed(tok, proc, ids, "the")
    cat = enc(tok, " cat")[0]
    _, chosen = advance(tok, proc, ids, [cat])
    assert chosen.item() != cat


def test_c5_rejects_redefinition(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    ids = feed(tok, proc, ids, "pay_n = 4 = 4\n")
    ids = feed(tok, proc, ids, "pay_n")
    eq = enc(tok, " =")[0]
    _, chosen = advance(tok, proc, ids, [eq])
    assert chosen.item() != eq


def test_c5_rejects_eos_before_answer(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    ids = feed(tok, proc, ids, "pay_n = 4 = 4\n")
    _, chosen = advance(tok, proc, ids, [tok.eos_token_id])
    assert chosen.item() != tok.eos_token_id


def test_c5_rejects_newline_after_answer(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    ids = feed(tok, proc, ids, "pay_n = 4 = 4\nANSWER = 4")
    newline = enc(tok, "\n")[0]
    _, chosen = advance(tok, proc, ids, [newline])
    assert "\n" not in tok.decode([chosen.item()])


def test_c5_allows_eos_after_answer(tok):
    proc, ids = new_proc(tok, "C5", True, False)
    ids = feed(tok, proc, ids, "pay_n = 4 = 4\nANSWER = 4")
    _, chosen = advance(tok, proc, ids, [tok.eos_token_id])
    assert chosen.item() == tok.eos_token_id


def test_c6_rejects_unit_outside_list(tok):
    proc, ids = new_proc(tok, "C6", True, False)
    ids = feed(tok, proc, ids, "pay_furlongs")
    eq = enc(tok, " =")[0]
    _, chosen = advance(tok, proc, ids, [eq])
    assert chosen.item() != eq


def test_c6_rejects_third_note(tok):
    proc, ids = new_proc(tok, "C6", True, False)
    ids = feed(tok, proc, ids, "NOTE one\nNOTE two\n")
    note = enc(tok, "NOTE")[0]
    _, chosen = advance(tok, proc, ids, [note])
    assert chosen.item() != note


def test_c5_calc_forces_value_then_only_newline(tok):
    proc, ids = new_proc(tok, "C5", True, True)
    ids = feed(tok, proc, ids, "x = 48 / 2 =")
    letter = enc(tok, "apples")[0]
    forced = enc(tok, " 24")
    for t in forced:
        ids, chosen = advance(tok, proc, ids, [letter])
        assert chosen.item() == t
    assert proc.n_forced[0] == len(forced)
    _, chosen = advance(tok, proc, ids, [letter])
    assert tok.decode([chosen.item()]).startswith("\n")


def test_c6_solve_forces_root_then_only_newline(tok):
    proc, ids = new_proc(tok, "C6", True, True)
    ids = feed(tok, proc, ids, SOLVE_LINE)
    letter = enc(tok, "apples")[0]
    forced = enc(tok, " 121")
    for t in forced:
        ids, chosen = advance(tok, proc, ids, [letter])
        assert chosen.item() == t
    assert proc.n_forced[0] == len(forced)
    _, chosen = advance(tok, proc, ids, [letter])
    assert tok.decode([chosen.item()]).startswith("\n")


def test_a_calc_fires_on_expression(tok):
    proc, ids = new_proc(tok, "A", False, True)
    ids = feed(tok, proc, ids, "she sold 48 / 2 =")
    letter = enc(tok, "apples")[0]
    forced = enc(tok, " 24")
    for t in forced:
        ids, chosen = advance(tok, proc, ids, [letter])
        assert chosen.item() == t
    assert proc.n_forced[0] == len(forced)
    _, chosen = advance(tok, proc, ids, [enc(tok, "5")[0]])
    assert not tok.decode([chosen.item()])[:1].isdigit()


def test_a_calc_does_not_fire_on_year(tok):
    proc, ids = new_proc(tok, "A", False, True)
    ids = feed(tok, proc, ids, "in year 2020 =")
    word = enc(tok, "apples")[0]
    _, chosen = advance(tok, proc, ids, [word])
    assert chosen.item() == word
    assert proc.n_forced[0] == 0


def test_batch_rows_have_independent_state(tok):
    a = enc(tok, "she sold 48 / 2 =")
    b = enc(tok, "she sold 48 / 2 +")
    assert len(a) == len(b)
    proc, ids = new_proc(tok, "A", False, True, batch=2)
    letter = enc(tok, "apples")[0]
    for ta, tb in zip(a, b):
        ids, chosen = advance(tok, proc, ids, [ta, tb])
        assert chosen.tolist() == [ta, tb]
    forced = enc(tok, " 24")
    ids, chosen = advance(tok, proc, ids, [letter, letter])
    assert chosen.tolist() == [forced[0], letter]
    for t in forced[1:]:
        ids, chosen = advance(tok, proc, ids, [letter, letter])
        assert chosen.tolist() == [t, letter]
    assert proc.n_forced == [len(forced), 0]


def test_a_calc_skips_tail_of_longer_expression():
    assert _a_calc_value("There are 60 students * 2/3 =") is None
    assert _a_calc_value("There are 60 students x 2/3 =") is None
    assert _a_calc_value("she sold 48 / 2 =") == 24


def test_a_calc_percent_sum_and_difference_stay_percent():
    assert _a_calc_value("discount 100% - 40% =") == pytest.approx(60)


def test_a_calc_mixed_percent_is_divided_by_100():
    assert _a_calc_value("20% * 50 =") == pytest.approx(10)
    assert _a_calc_value("50 / 20% =") == pytest.approx(250)


def test_audit_a_trigger_counts():
    texts = ["she sold 48 / 2 = 24 apples. in year 2020 = 2020"]
    res = audit_a_trigger(texts)
    assert res["equals"] == 2
    assert res["fired"] == 1
    assert res["agree"] == 1
    assert res["disagree"] == 0
    assert len(res["not_fired_examples"]) == 1
    assert res["disagree_examples"] == []
