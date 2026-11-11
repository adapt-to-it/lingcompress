"""Formati di v5: estrazione e normalizzazione della risposta, stripper B e verifica del canonico C."""
import ast
import operator
import re

from src.spec5 import A_FINAL, B_FINAL, C_CMP, C_NAME_RE, C_NUM_RE, FORMATS, STRIP_PRONOUNS, STRIP_WORDS

# Numero nel testo libero: puo' avere $ e virgole di migliaia; il punto finale di frase non fa parte del numero.
_ANY = r"\$?-?[0-9][0-9,]*(?:\.[0-9]+)?"
_NUM_TXT = re.compile(r"(?<![0-9A-Za-z])" + _ANY)
_A_RE = re.compile(r"the answer is\s+(" + _ANY + ")", re.IGNORECASE)
_B_RE = re.compile(r"answer\s+(" + _ANY + ")", re.IGNORECASE)
_C_RE = re.compile(r"ANSWER = (" + _ANY + ")")

# Parole intere da togliere (B) e contrazioni soggetto+ausiliare.
_STRIP_RE = re.compile(r"(?<![\w'’])(?:" + "|".join(sorted(STRIP_WORDS)) + r")(?![\w'’])", re.IGNORECASE)
_CONTR_RE = re.compile(r"(?<![\w'’])(?:" + "|".join(STRIP_PRONOUNS) + r")['’](?:s|re|ll|ve|d|m)(?![\w'’])",
                       re.IGNORECASE)

# Grammatica del canonico C (vedi spec5).
_CMP_ALT = "|".join(re.escape(c) for c in C_CMP)
_GIVEN_RE = re.compile(r"(GIVEN|FACT) (" + C_NAME_RE + r") = (" + C_NUM_RE + r")")
_IF_RE = re.compile(r"IF (.+?) (" + _CMP_ALT + r") (.+)")
_CALC_RE = re.compile(r"(" + C_NAME_RE + r") = (.+?) = (" + C_NUM_RE + r")")
_ANSWER_RE = re.compile(r"ANSWER = (" + C_NUM_RE + r")")
_EXPR_CHARS = re.compile(r"[0-9+\-*/(). ]+")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _tpl_re(tpl):
    """Regex della riga finale da un modello con {value}; il valore e' un NUM senza virgole."""
    pre, post = tpl.split("{value}")
    return re.compile(re.escape(pre) + "(" + C_NUM_RE + ")" + re.escape(post))


_FINAL_A = _tpl_re(A_FINAL)
_FINAL_B = _tpl_re(B_FINAL)


def norm_num(s):
    """Toglie $ , % e il punto finale; interi come '24', '0.50' come '0.5'. None se non e' un numero."""
    t = s.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if t.endswith("."):
        t = t[:-1]
    if not re.fullmatch(C_NUM_RE, t):
        return None
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return "0" if t == "-0" else t


def official_answer(answer_text):
    """Numero dopo l'ultimo '####' della soluzione ufficiale GSM8K, normalizzato."""
    found = re.findall(r"####\s*(.*)", answer_text)
    if not found:
        raise ValueError("soluzione ufficiale senza '####'")
    return norm_num(found[-1]) or found[-1].strip()


def official_steps(answer_text):
    """Risultati delle note <<expr=res>> della soluzione ufficiale, normalizzati, in ordine."""
    steps = []
    for note in re.findall(r"<<(.*?)>>", answer_text):
        if "=" not in note:
            continue
        res = note.rsplit("=", 1)[1].strip()
        steps.append(norm_num(res) or res)
    return steps


def same_number(a, b):
    """Uguaglianza numerica con tolleranza 1e-6; False se uno dei due non e' un numero."""
    try:
        return abs(float(a) - float(b)) <= 1e-6
    except (TypeError, ValueError):
        return False


def strip_b(text):
    """Stripper B: per riga toglie le contrazioni pronome+ausiliare e le parole STRIP_WORDS; righe vuote scartate."""
    out = []
    for line in text.splitlines():
        line = _CONTR_RE.sub("", line)
        line = _STRIP_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([.,;:!?])", r"\1", line)
        if line:
            out.append(line)
    return "\n".join(out)


def _close(a, b):
    return abs(a - b) <= 1e-6


def _ev(node):
    """Valutazione ricorsiva di un nodo ast ammesso; ValueError per tutto il resto."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ev(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    raise ValueError("nodo non ammesso")


def _eval_expr(expr):
    """Valuta EXPR (numeri, + - * /, unario -, parentesi); None se non e' un'espressione valida."""
    expr = expr.strip()
    if not _EXPR_CHARS.fullmatch(expr):
        return None
    try:
        return _ev(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _ok_value(v, num):
    """Valore di un passaggio: |eval - NUM| <= 1e-6, oppure NUM con >= 2 decimali uguale a eval arrotondato."""
    n = float(num)
    if _close(v, n):
        return True
    dec = len(num.split(".")[1]) if "." in num else 0
    return dec >= 2 and _close(round(v, dec), n)


def _compare(a, op, b):
    if op == "==":
        return _close(a, b)
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[op]


def check_c(text, arithmetic=True):
    """Verifica il canonico C secondo la grammatica di spec5; con arithmetic controlla anche valori e confronti IF."""
    errors = []
    defined = set()
    answers = []
    lines = text.strip().split("\n")

    def define(name, i):
        if name in defined:
            errors.append(f"riga {i}: nome {name} gia' definito")
        defined.add(name)

    for i, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(f"riga {i}: riga vuota")
        elif line.startswith("ANSWER"):
            answers.append(i)
            if not _ANSWER_RE.fullmatch(line):
                errors.append(f"riga {i}: ANSWER non valido")
        elif line.startswith("IF "):
            m = _IF_RE.fullmatch(line)
            if not m:
                errors.append(f"riga {i}: IF non valido")
                continue
            lhs, op, rhs = m.groups()
            a = _eval_expr(lhs)
            b = _eval_expr(rhs)
            if a is None or b is None:
                errors.append(f"riga {i}: espressione non valida (nomi e unita' non ammessi)")
            elif arithmetic and not _compare(a, op, b):
                errors.append(f"riga {i}: condizione IF falsa")
        elif _GIVEN_RE.fullmatch(line):
            define(_GIVEN_RE.fullmatch(line).group(2), i)
        else:
            m = _CALC_RE.fullmatch(line)
            if not m:
                errors.append(f"riga {i}: formato non riconosciuto")
                continue
            name, expr, num = m.groups()
            define(name, i)
            v = _eval_expr(expr)
            if v is None:
                errors.append(f"riga {i}: espressione non valida (nomi e unita' non ammessi)")
            elif arithmetic and not _ok_value(v, num):
                errors.append(f"riga {i}: valore {num} diverso dal calcolo {v}")

    if len(answers) != 1:
        errors.append(f"ANSWER deve comparire una volta, trovate {len(answers)}")
    elif answers[0] != len(lines):
        errors.append("testo dopo ANSWER")
    return not errors, errors


def extract_answer(fmt, text):
    """Risposta numerica del testo nel formato dato; A/B/C ripiegano sull'ultimo numero, D sul primo."""
    if fmt not in FORMATS:
        raise ValueError(f"formato sconosciuto: {fmt}")
    if fmt == "D":
        m = _NUM_TXT.search(text)
        return norm_num(m.group(0)) if m else None
    pat = {"A": _A_RE, "B": _B_RE, "C": _C_RE}[fmt]
    found = pat.findall(text)
    if found:
        raw = found[-1]
    else:
        nums = _NUM_TXT.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    return norm_num(raw)


def conforms(fmt, text):
    """Il testo rispetta il formato: A con riga finale A_FINAL, B telegrafico, C canonico, D solo numero."""
    if fmt not in FORMATS:
        raise ValueError(f"formato sconosciuto: {fmt}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if fmt == "A":
        return len(lines) >= 2 and _FINAL_A.fullmatch(lines[-1]) is not None
    if fmt == "B":
        return (bool(lines) and _FINAL_B.fullmatch(lines[-1]) is not None
                and _STRIP_RE.search(text) is None and _CONTR_RE.search(text) is None)
    if fmt == "C":
        return check_c(text, arithmetic=False)[0]
    return re.fullmatch(C_NUM_RE, text.strip()) is not None


def render_d(value):
    """Numero del formato D (solo risposta), normalizzato; ValueError se il valore non e' un numero."""
    txt = f"{value:.10f}" if isinstance(value, float) else str(value)
    n = norm_num(txt)
    if n is None:
        raise ValueError(f"valore non numerico: {value!r}")
    return n


_A_LAST = re.compile(r"the answer is\s+(" + _ANY + r")\s*\.?", re.IGNORECASE)
_C_BARE = re.compile(C_NAME_RE + r" = " + C_NUM_RE)
_C_NUM_TOKEN = re.compile(r"(?<![\w.])[0-9]+(?:\.[0-9]+)?(?![\w.])")


def normalize(fmt, text):
    """Ripulitura deterministica di una riscrittura prima della verifica.
    Tutti: spazi in coda e righe vuote. A: riga finale nella forma esatta di A_FINAL.
    C: niente $ e virgole delle migliaia, numeri in forma normale (18.00 -> 18), 'nome = numero' -> 'GIVEN nome = numero'."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if fmt == "A" and lines:
        m = _A_LAST.fullmatch(lines[-1])
        if m and norm_num(m.group(1)) is not None:
            lines[-1] = A_FINAL.format(value=norm_num(m.group(1)))
    elif fmt == "C":
        out = []
        for ln in lines:
            ln = re.sub(r"(?<=[0-9]),(?=[0-9]{3})", "", ln.replace("$", ""))
            ln = re.sub(r"\s+", " ", _C_NUM_TOKEN.sub(lambda m: norm_num(m.group(0)) or m.group(0), ln))
            out.append("GIVEN " + ln if _C_BARE.fullmatch(ln) else ln)
        lines = out
    return "\n".join(lines)
