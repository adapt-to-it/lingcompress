"""Test di src/grammar6.py."""
import pytest

from src.grammar6 import check_c6, fmt_value, normalize_c6, solve_linear, stats_c6
from src.spec6 import FEWSHOT_C6, NOTE_MAX_WORDS

TRAIN10_EQ = "first_ship_n + 2 * first_ship_n + 4 * first_ship_n = 847"
TRAIN10_SOLVE = TRAIN10_EQ + " -> first_ship_n = 121"


@pytest.mark.parametrize("text", list(FEWSHOT_C6.values()))
def test_fewshot_passes_and_is_normal(text):
    ok, errors = check_c6(text)
    assert ok, errors
    assert normalize_c6(text) == text


@pytest.mark.parametrize("v, expected", [
    (24.0, "24"),
    (0.2, "0.2"),
    (1 / 3, "0.33"),
    (0.004, "0.004"),
    (-0.5, "-0.5"),
    (0.00123456, "0.001235"),
    (2.5, "2.5"),
])
def test_fmt_value(v, expected):
    assert fmt_value(v) == expected


def test_solve_linear_train10():
    assert solve_linear(TRAIN10_EQ, "first_ship_n") == pytest.approx(121)


def test_solve_linear_rejects_nonlinear_and_other_identifiers():
    assert solve_linear("x * x = 4", "x") is None
    assert solve_linear("x + y = 3", "x") is None
    assert solve_linear("x - x = 0", "x") is None


def test_stats_train10():
    assert stats_c6(FEWSHOT_C6["train-10"]) == {
        "n_lines": 4, "n_note": 1, "n_solve": 1, "units": ["ratio", "n"]}


def test_normalize_c6_strips_note_and_normalizes_numbers():
    text = "  NOTE costs $1,000 \n price_usd = $1,000.50 \nANSWER = 1"
    assert normalize_c6(text) == "NOTE costs $1,000\nGIVEN price_usd = 1000.5\nANSWER = 1"


@pytest.mark.parametrize("text", [
    "pay_per_h_xyz = 12 / 60 = 0.2\nANSWER = 0.2",
    "FACT hour_min = 60\nFACT hour_min = 60\nANSWER = 60",
    "NOTE a\nNOTE b\nNOTE c\nANSWER = 1",
    "NOTE " + " ".join(["word"] * (NOTE_MAX_WORDS + 1)) + "\nANSWER = 1",
    "GIVEN discount_pct = 150\nANSWER = 150",
    "calc_pct = 10 * 20 / 100 = 200\nANSWER = 200",
    "FACT hour_min = 60\n" + TRAIN10_SOLVE.replace("= 121", "= 120") + "\nANSWER = 120",
    TRAIN10_EQ.join(["SOLVE first_ship_n: ", " -> second_ship_n = 121"]) + "\nANSWER = 121",
    FEWSHOT_C6["train-10"] + "\nFACT x_n = 1",
    "pay_per_min_usd = 12 / 60 = 0.3\nANSWER = 0.3",
])
def test_check_c6_errors(text):
    ok, errors = check_c6(text)
    assert not ok
    assert errors


def test_check_c6_arithmetic_flag_skips_values():
    text = "pay_per_min_usd = 12 / 60 = 0.3\nANSWER = 0.3"
    assert check_c6(text, arithmetic=False)[0]
    assert not check_c6(text)[0]
