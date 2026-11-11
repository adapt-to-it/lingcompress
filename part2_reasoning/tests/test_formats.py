"""Test di src/formats.py."""
import pytest

from src.formats import (
    check_c,
    conforms,
    extract_answer,
    norm_num,
    official_answer,
    official_steps,
    render_d,
    same_number,
    strip_b,
)
from src.spec5 import FEWSHOT

# Soluzione ufficiale di GSM8K train-0 (riga 1 di data/gsm8k/train.jsonl).
GSM8K_TRAIN_0 = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


@pytest.mark.parametrize("text, expected", [
    ("$1,000", "1000"),
    ("24.0", "24"),
    ("0.50", "0.5"),
    ("72.", "72"),
    ("50%", "50"),
    ("-0.0", "0"),
    ("abc", None),
    ("", None),
    ("1.2.3", None),
])
def test_norm_num(text, expected):
    assert norm_num(text) == expected


def test_same_number():
    assert same_number("24", "24.0")
    assert not same_number("1", "1.1")
    assert not same_number("abc", "1")


def test_official_answer_and_steps_train_0():
    assert official_answer(GSM8K_TRAIN_0) == "72"
    assert official_steps(GSM8K_TRAIN_0) == ["24", "72"]


@pytest.mark.parametrize("text, expected", [
    ("The answer is 72.", "answer 72."),
    ("She's ready.", "ready."),
    ("It'll rain.", "rain."),
    ("The sum is 3.\n\nThe total is 4.", "sum 3.\ntotal 4."),
])
def test_strip_b(text, expected):
    assert strip_b(text) == expected


def test_check_c_accepts_valid_program():
    assert check_c("x = 48 / 2 = 24\nANSWER = 24") == (True, [])
    # arrotondamento: NUM con due decimali uguale al calcolo arrotondato
    assert check_c("a = 1 / 3 = 0.33\nANSWER = 0.33")[0]


def test_check_c_accepts_true_if():
    assert check_c("y = 2 + 2 = 4\nIF 2 + 2 >= 4\nANSWER = 4")[0]


def test_check_c_redefined_name():
    ok, errors = check_c("x = 48 / 2 = 24\nx = 48 + 24 = 72\nANSWER = 72")
    assert not ok
    assert any("gia' definito" in e for e in errors)


def test_check_c_wrong_value():
    ok, errors = check_c("x = 48 / 2 = 25\nANSWER = 25")
    assert not ok
    assert check_c("x = 48 / 2 = 25\nANSWER = 25", arithmetic=False)[0]


def test_check_c_name_inside_expression():
    assert not check_c("clips = apples / 2 = 24\nANSWER = 24")[0]
    assert not check_c("clips = apples / 2 = 24\nANSWER = 24", arithmetic=False)[0]


def test_check_c_missing_answer():
    ok, errors = check_c("x = 48 / 2 = 24")
    assert not ok
    assert any("ANSWER" in e for e in errors)


def test_check_c_false_if():
    text = "y = 2 + 2 = 4\nIF 2 + 2 > 5\nANSWER = 4"
    ok, errors = check_c(text)
    assert not ok
    assert any("IF" in e for e in errors)
    assert check_c(text, arithmetic=False)[0]


def test_check_c_text_after_answer():
    ok, errors = check_c("x = 2 + 2 = 4\nANSWER = 4\ny = 1 + 1 = 2")
    assert not ok
    assert "testo dopo ANSWER" in errors


@pytest.mark.parametrize("text, expected", [
    ("The answer is 1,000.", "1000"),
    ("The answer is 3.\nthe answer is 4.", "4"),
    ("We had 48 and 24 and 72 in total", "72"),
])
def test_extract_answer_a(text, expected):
    assert extract_answer("A", text) == expected


def test_extract_answer_b_c_d():
    assert extract_answer("B", "answer 72.") == "72"
    assert extract_answer("C", "x = 1\nANSWER = 0.50") == "0.5"
    assert extract_answer("C", "no answer 7 here") == "7"
    assert extract_answer("D", " 1,000 ") == "1000"
    assert extract_answer("D", "none") is None
    assert extract_answer("A", "nothing") is None


def test_render_d():
    assert render_d(72) == "72"
    assert render_d(10.0) == "10"
    assert render_d(0.2) == "0.2"
    assert render_d("3.50") == "3.5"


def test_conforms_a():
    assert conforms("A", "Step 2 + 3 = 5.\nThe answer is 5.")
    assert not conforms("A", "The answer is 5.")
    assert not conforms("A", "a\nThe answer is 1,000.")


def test_conforms_b():
    assert conforms("B", "answer 72.")
    assert not conforms("B", "the answer 72.")
    assert not conforms("B", "answer is 72.")


def test_conforms_d():
    assert conforms("D", "72")
    assert conforms("D", "-3.5")
    assert not conforms("D", "72 dollars")
    assert not conforms("D", "1,000")


@pytest.mark.parametrize("fmt", ["C"])
@pytest.mark.parametrize("example", sorted(FEWSHOT["C"]))
def test_fewshot_c_passes(fmt, example):
    text = FEWSHOT[fmt][example]
    ok, errors = check_c(text)
    assert ok, errors
    assert conforms("C", text)


@pytest.mark.parametrize("example", sorted(FEWSHOT["A"]))
def test_fewshot_a_conforms_and_strips_to_b(example):
    text = FEWSHOT["A"][example]
    assert conforms("A", text)
    assert conforms("B", strip_b(text))
