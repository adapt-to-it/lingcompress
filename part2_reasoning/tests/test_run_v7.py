"""Test di run_v7.py su dati sintetici: nomi degli adapter, modo di decodifica e costruzione delle righe SFT."""
from run_v7 import _adapter, _mode, build_rows
from src.spec5 import RUNS
from src.spec6 import LR6


def test_adapter_names_per_condition():
    assert _adapter("small", 1, 0) == RUNS / f"small_A_lr{LR6:g}_s0" / "adapter"
    assert _adapter("large", 3, 2) == RUNS / f"large_C_lr{LR6:g}_s2" / "adapter"
    assert _adapter("small", 6, 1) == RUNS / f"small_C6_lr{LR6:g}_s1" / "adapter"
    assert _adapter("small", 9, 1) == RUNS / f"small_A1_lr{LR6:g}_s1" / "adapter"
    assert _adapter("large", 11, 0) == RUNS / f"large_A2_lr{LR6:g}_s0" / "adapter"


def test_mode_maps_a_variants_to_a():
    assert _mode("A") == "A"
    assert _mode("A1") == "A"
    assert _mode("A2") == "A"
    assert _mode("C5") == "C5"
    assert _mode("C6") == "C6"


def test_build_rows_soft_keeps_strict_and_soft_in_a_order():
    a_rows = [{"id": "a", "question": "qa", "solution": "old a", "answer": "1"},
              {"id": "b", "question": "qb", "solution": "old b", "answer": "2"},
              {"id": "c", "question": "qc", "solution": "old c", "answer": "3"}]
    chosen = {"b": ("new b", False), "a": ("new a", True)}
    rows = build_rows(a_rows, chosen, "soft")
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0] == {"id": "a", "question": "qa", "solution": "new a", "answer": "1"}
    assert rows[1]["solution"] == "new b"
    assert a_rows[0]["solution"] == "old a"


def test_build_rows_strict_drops_soft():
    a_rows = [{"id": "a", "question": "qa", "solution": "old a", "answer": "1"},
              {"id": "b", "question": "qb", "solution": "old b", "answer": "2"}]
    chosen = {"a": ("new a", True), "b": ("new b", False)}
    rows = build_rows(a_rows, chosen, "strict")
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["solution"] == "new a"
