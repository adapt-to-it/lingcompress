"""Decodifica vincolata greedy per i formati A, C5 e C6: LogitsProcessor con grammatica di riga e calcolatrice."""
import regex
import torch
import transformers

from src.formats import _eval_expr, norm_num
from src.grammar6 import fmt_value, solve_linear
from src.spec6 import (A_CALC_TRIGGER, ANSWER_LINE, CALC_TRIGGER, DEF_PREFIX, NOTE_MAX_LINES, SOLVE_TRIGGER,
                       line_regex)

MODES = ("A", "C5", "C6")
TOP_K = (64, 4096)                                       # candidati in ordine di logit: prima 64, poi 4096

_ANSWER_RE = regex.compile(ANSWER_LINE)
_DEF_RE = regex.compile(DEF_PREFIX)
_CALC_RE = regex.compile(CALC_TRIGGER)
_SOLVE_RE = regex.compile(SOLVE_TRIGGER)
_A_CALC_RE = regex.compile(A_CALC_TRIGGER)
_NEXT_NUM_RE = regex.compile(r"\s*(\$?-?[0-9][0-9,]*(?:\.[0-9]+)?%?)")
_TOKEN_CACHE = {}


def _token_info(tok):
    """Stringhe di tutti i token e id speciali; calcolati una sola volta per tokenizer."""
    hit = _TOKEN_CACHE.get(id(tok))
    if hit is None or hit[0] is not tok:
        strs = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tok))]
        special = set(tok.all_special_ids)
        special.update(i for i, t in tok.added_tokens_decoder.items() if getattr(t, "special", False))
        hit = (tok, strs, frozenset(special))
        _TOKEN_CACHE[id(tok)] = hit
    return hit[1], hit[2]


_PCT_RE = regex.compile(r"(\d[\d.]*)%")
_OP_RE = regex.compile(r"[-+*/×x]")
_PCT_FLAG_RE = regex.compile(r"\d[\d,]*(?:\.\d+)?(%?)")


def _a_calc_value(text):
    """Valore della calcolatrice A se il testo termina con un'espressione seguita da ' ='; None altrimenti."""
    m = _A_CALC_RE.search(text)
    if m is None:
        return None
    # Coda di un'espressione piu' lunga (es. "60 students * 2/3 ="): nessuno scatto.
    before = text[: m.start("expr")].rstrip()
    if before[-1:] in ("+", "-", "*", "/", "×") or regex.search(r"(?:^|\s)x$", before):
        return None
    expr = m.group("expr").replace("$", "").replace(",", "")
    flags = _PCT_FLAG_RE.findall(expr)
    if flags and all(flags) and set(_OP_RE.findall(expr)) <= {"+", "-"}:
        expr = expr.replace("%", "")                     # 100% - 40% = 60: somme e differenze di percentuali
    else:
        expr = _PCT_RE.sub(r"(\1/100)", expr)            # 20% * 50 = 10: il % diventa /100 tra parentesi
    expr = expr.replace("×", "*").replace("x", "*")
    return _eval_expr(expr)


class _Row:
    """Stato di una riga del batch."""

    def __init__(self):
        self.text = ""           # testo generato
        self.line = ""           # riga aperta
        self.names = set()       # nomi gia' definiti
        self.notes = 0           # righe NOTE completate
        self.queue = []          # id forzati in attesa
        self.next_rule = None    # vincolo sul token successivo: "newline" | "nodigit" | None
        self.done = False


class ReasoningProcessor(transformers.LogitsProcessor):
    """Maschera i logit a ogni passo greedy: grammatica C5/C6 (se grammar), calcolatrice (se calc)."""

    def __init__(self, tok, mode, grammar, calc, prompt_len, batch_size):
        if mode not in MODES:
            raise ValueError(f"modo sconosciuto: {mode}")
        if grammar and mode == "A":
            raise ValueError("la grammatica vale solo per C5 e C6")
        self.tok = tok
        self.mode = mode
        self.grammar = grammar
        self.calc = calc
        self.prompt_len = prompt_len
        self.strs, self.special = _token_info(tok)
        self.eos = tok.eos_token_id
        self.line_re = regex.compile(line_regex(mode)) if mode != "A" else None
        self.n_forced = [0] * batch_size
        self.states = [_Row() for _ in range(batch_size)]

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > self.prompt_len:
            for st, tid in zip(self.states, input_ids[:, -1].tolist()):
                self._advance(st, tid)
        return torch.stack([self._mask_row(b, st, scores[b]) for b, st in enumerate(self.states)])

    def _mask_row(self, b, st, row):
        """Lascia un solo logit finito: il token forzato, il primo ammesso, o EOS se nessuno e' ammesso."""
        out = torch.full_like(row, float("-inf"))
        if st.done:
            out[self.eos] = 0.0
            return out
        if st.queue:
            self.n_forced[b] += 1
            tid = st.queue[0]
        else:
            tid = self._pick(st, row)
            if tid is None:
                tid = self.eos
        out[tid] = row[tid]
        return out

    def _pick(self, st, row):
        """Primo token ammesso in ordine di logit decrescente (prima top 64, poi top 4096)."""
        for k in TOP_K:
            for tid in torch.topk(row, min(k, row.shape[-1])).indices.tolist():
                if self._allowed(st, tid):
                    return tid
        return None

    def _allowed(self, st, tid):
        """Il token tid puo' seguire lo stato st: regola dopo il valore forzato, poi grammatica."""
        if tid == self.eos:
            if st.next_rule == "newline":
                return False
            return not self.grammar or _ANSWER_RE.fullmatch(st.line) is not None
        if tid >= len(self.strs):
            return False
        s = self.strs[tid]
        if st.next_rule == "newline" and not s.startswith("\n"):
            return False
        if st.next_rule == "nodigit" and s[:1].isdigit():
            return False
        if not self.grammar:
            return True
        if tid in self.special or not s or not s.isascii():
            return False
        return self._grammar_ok(st, s)

    def _grammar_ok(self, st, s):
        """Il testo della riga aperta piu' s resta prefisso valido: righe completate intere, riga aperta parziale."""
        parts = (st.line + s).split("\n")
        names = set(st.names)
        notes = st.notes
        for line in parts[:-1]:
            if not self._line_complete_ok(line, names):
                return False
            if line.startswith("NOTE "):
                notes += 1
                if notes > NOTE_MAX_LINES:
                    return False
        return self._open_ok(parts[-1], names, notes)

    def _line_complete_ok(self, line, names):
        """Riga completata: non vuota, valida per il formato, non ANSWER (niente a capo dopo ANSWER), nome nuovo."""
        if not line or _ANSWER_RE.fullmatch(line):
            return False
        m = self.line_re.fullmatch(line)
        if m is None:
            return False
        g = m.groupdict()
        if g.get("sname") != g.get("sname2"):
            return False
        d = _DEF_RE.match(line)
        if d is not None:
            name = d.group(1) or d.group(2)
            if name in names:
                return False
            names.add(name)
        return True

    def _open_ok(self, line, names, notes):
        """Riga aperta: prefisso parziale della grammatica, nessun nome gia' definito, al piu' NOTE_MAX_LINES NOTE."""
        if self.line_re.fullmatch(line, partial=True) is None:
            return False
        if line and notes >= NOTE_MAX_LINES and ("NOTE ".startswith(line) or line.startswith("NOTE ")):
            return False
        d = _DEF_RE.match(line)
        return d is None or (d.group(1) or d.group(2)) not in names

    def _advance(self, st, tid):
        """Registra nel gruppo di stato il token appena generato (o forzato)."""
        if st.done:
            return
        if tid == self.eos:
            st.done = True
            return
        s = self.strs[tid]
        forced = bool(st.queue)
        if forced:
            st.queue.pop(0)
        else:
            st.next_rule = None
        st.text += s
        parts = (st.line + s).split("\n")
        for line in parts[:-1]:
            self._commit(st, line)
        st.line = parts[-1]
        if self.calc and not forced:
            self._check_triggers(st)

    def _commit(self, st, line):
        """Riga completata: registra il nome definito e il conteggio delle NOTE."""
        d = _DEF_RE.match(line)
        if d is not None:
            st.names.add(d.group(1) or d.group(2))
        if line.startswith("NOTE "):
            st.notes += 1

    def _check_triggers(self, st):
        """Calcolatrice: se il testo o la riga aperta termina come un'espressione, forza il valore."""
        if self.mode == "A":
            v = _a_calc_value(st.text)
            if v is not None:
                self._force(st, v, "nodigit")
            return
        m = _CALC_RE.fullmatch(st.line)
        if m is not None:
            v = _eval_expr(m.group("expr"))
            if v is not None:
                self._force(st, v, "newline")
            return
        m = _SOLVE_RE.fullmatch(st.line)
        if m is not None:
            root = solve_linear(m.group("eq"), m.group("u"))
            if root is not None:
                self._force(st, root, "newline")

    def _force(self, st, v, rule):
        """Accoda il valore forzato (con lo spazio iniziale) e la regola sul token che lo segue."""
        st.queue = self.tok.encode(" " + fmt_value(v), add_special_tokens=False)
        st.next_rule = rule


def make_processor(tok, mode, grammar, calc):
    """Fabbrica di ReasoningProcessor per una condizione: factory(prompt_len, batch_size)."""
    def factory(prompt_len, batch_size):
        return ReasoningProcessor(tok, mode, grammar, calc, prompt_len, batch_size)
    return factory


def audit_a_trigger(texts):
    """Per ogni '=' dei testi: la calcolatrice A scatta sul prefisso? Se si', fmt_value concorda con il numero successivo?"""
    counts = {"equals": 0, "fired": 0, "agree": 0, "disagree": 0}
    not_fired, disagree = [], []
    for text in texts:
        for pos, ch in enumerate(text):
            if ch != "=":
                continue
            counts["equals"] += 1
            ctx = text[max(0, pos - 60): pos + 30]
            v = _a_calc_value(text[: pos + 1])
            if v is None:
                if len(not_fired) < 10:
                    not_fired.append(ctx)
                continue
            counts["fired"] += 1
            m = _NEXT_NUM_RE.match(text, pos + 1)
            nxt = norm_num(m.group(1)) if m else None
            if nxt is not None and nxt == norm_num(fmt_value(v)):
                counts["agree"] += 1
            else:
                counts["disagree"] += 1
                if len(disagree) < 10:
                    disagree.append(ctx)
    return {**counts, "not_fired_examples": not_fired, "disagree_examples": disagree}
