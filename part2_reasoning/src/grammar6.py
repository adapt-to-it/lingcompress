"""Parser del canonico C6 (v6): formato dei valori, risoluzione lineare, verifica e normalizzazione."""
import math
import re

import regex

from src.formats import _compare, _eval_expr, _ok_value, normalize
from src.spec6 import CMP, EQSIDE, EXPR, NAME6, NOTE_MAX_LINES, NOTE_MAX_WORDS, NUM, PCT_RANGE, line_regex

_LINE6 = regex.compile(line_regex("C6"))
_ANSWER6 = regex.compile(rf"ANSWER = ({NUM})")
_IF6 = regex.compile(rf"IF ({EXPR}) ({CMP}) ({EXPR})")
_SOLVE6 = regex.compile(rf"SOLVE ({NAME6}): ({EQSIDE} = {EQSIDE}) -> ({NAME6}) = ({NUM})")
_GIVEN6 = regex.compile(rf"(?:GIVEN|FACT) ({NAME6}) = ({NUM})")
_CALC6 = regex.compile(rf"({NAME6}) = ({EXPR}) = ({NUM})")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def fmt_value(v):
    """Valore forzato: intero se |v - round(v)| < 1e-9; altrimenti 2 decimali (4 cifre significative se |v| < 0.01),
    senza zeri finali."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    decimals = 3 - math.floor(math.log10(abs(v))) if abs(v) < 0.01 else 2
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def solve_linear(eq, name):
    """Radice di 'lhs = rhs' in un'unica incognita name (numeri, + - * / ( ), nessun altro identificatore).
    None se il lato non e' valido, se f non e' affine o se il coefficiente e' nullo."""
    parts = eq.split("=")
    if len(parts) != 2:
        return None
    if any(ident != name for ident in _IDENT_RE.findall(eq)):
        return None
    lhs, rhs = parts
    pattern = rf"\b{re.escape(name)}\b"

    def f(t):
        a = _eval_expr(re.sub(pattern, f"({t})", lhs))
        b = _eval_expr(re.sub(pattern, f"({t})", rhs))
        return None if a is None or b is None else a - b

    f0, f1, f2 = f(0), f(1), f(2)
    if f0 is None or f1 is None or f2 is None:
        return None
    slope = f1 - f0
    if abs(f2 - f1 - slope) > 1e-9 * max(1.0, abs(f0), abs(f1), abs(f2)):
        return None
    if abs(slope) < 1e-12:
        return None
    return -f0 / slope


def _parse(line):
    """Tipo di una riga C6 e suoi campi: ('NOTE'|'ANSWER'|'IF'|'SOLVE'|'GIVEN'|'CALC', gruppi) o (None, ())."""
    if line.startswith("NOTE "):
        return "NOTE", ()
    for kind, pat in (("ANSWER", _ANSWER6), ("IF", _IF6), ("SOLVE", _SOLVE6), ("GIVEN", _GIVEN6), ("CALC", _CALC6)):
        m = pat.fullmatch(line)
        if m:
            return kind, m.groups()
    return None, ()


def check_c6(text, arithmetic=True):
    """Verifica il canonico C6 secondo la grammatica di spec6; con arithmetic controlla anche valori, IF e SOLVE."""
    errors = []
    defined = set()
    answers = []
    n_note = 0
    lines = text.strip().split("\n")

    def define(name, i):
        if name in defined:
            errors.append(f"riga {i}: nome {name} gia' definito")
        defined.add(name)

    def check_pct(name, value, i):
        if name.endswith("_pct") and not PCT_RANGE[0] <= value <= PCT_RANGE[1]:
            errors.append(f"riga {i}: {name} fuori da {PCT_RANGE[0]}-{PCT_RANGE[1]}")

    for i, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(f"riga {i}: riga vuota")
            continue
        if not _LINE6.fullmatch(line):
            errors.append(f"riga {i}: formato non riconosciuto")
            continue
        kind, g = _parse(line)
        if kind == "NOTE":
            n_note += 1
            if len(line.split()) - 1 > NOTE_MAX_WORDS:
                errors.append(f"riga {i}: NOTE oltre {NOTE_MAX_WORDS} parole")
        elif kind == "ANSWER":
            answers.append(i)
        elif kind == "IF":
            lhs, op, rhs = g
            a = _eval_expr(lhs)
            b = _eval_expr(rhs)
            if a is None or b is None:
                errors.append(f"riga {i}: espressione non valida (nomi e unita' non ammessi)")
            elif arithmetic and not _compare(a, op, b):
                errors.append(f"riga {i}: condizione IF falsa")
        elif kind == "GIVEN":
            name, num = g
            define(name, i)
            check_pct(name, float(num), i)
        elif kind == "CALC":
            name, expr, num = g
            define(name, i)
            v = _eval_expr(expr)
            if v is None:
                errors.append(f"riga {i}: espressione non valida (nomi e unita' non ammessi)")
            elif arithmetic and not _ok_value(v, num):
                errors.append(f"riga {i}: valore {num} diverso dal calcolo {v}")
            check_pct(name, float(num), i)
        elif kind == "SOLVE":
            sname, eq, sname2, num = g
            define(sname, i)
            if sname != sname2:
                errors.append(f"riga {i}: SOLVE {sname} e -> {sname2} non coincidono")
            root = solve_linear(eq, sname)
            if root is None:
                errors.append(f"riga {i}: equazione non lineare o senza soluzione in {sname}")
            elif arithmetic and not _ok_value(root, num):
                errors.append(f"riga {i}: valore {num} diverso dalla radice {root}")

    if n_note > NOTE_MAX_LINES:
        errors.append(f"piu' di {NOTE_MAX_LINES} NOTE")
    if len(answers) != 1:
        errors.append(f"ANSWER deve comparire una volta, trovate {len(answers)}")
    elif answers[0] != len(lines):
        errors.append("testo dopo ANSWER")
    return not errors, errors


def normalize_c6(text):
    """Normalizzazione C di formats.normalize per le righe non NOTE; le righe NOTE sono solo ripulite agli estremi."""
    out = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(ln if ln.startswith("NOTE ") else normalize("C", ln))
    return "\n".join(out)


def stats_c6(text):
    """Conteggi di un testo C6: righe, NOTE, SOLVE e suffissi di unita' dei nomi definiti, in ordine."""
    lines = text.strip().split("\n")
    parsed = [_parse(ln) for ln in lines]
    units = [g[0].rsplit("_", 1)[1] for kind, g in parsed if kind in ("GIVEN", "CALC", "SOLVE")]
    return {
        "n_lines": len(lines),
        "n_note": sum(kind == "NOTE" for kind, _ in parsed),
        "n_solve": sum(kind == "SOLVE" for kind, _ in parsed),
        "units": units,
    }
