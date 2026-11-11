"""Specifica condivisa di v7: prosa concisa A1 / A2 (riscritture di A di v5), condizioni 9-12, scala a 4B."""
from src.spec5 import RESULTS, REWRITES

RESULTS7 = RESULTS.parent / "v7"
FORMATS7 = ("A1", "A2")
FORMAT7_NAMES = {"A1": "conciso", "A2": "molto conciso"}
# Lunghezza massima in token Qwen3 rispetto alla soluzione A dello stesso problema (oltre: nuovo tentativo).
MAX_RATIO = {"A1": 0.85, "A2": 0.65}
ATTEMPTS7 = 3                             # tentativo 1 a temperatura 0, poi 0.7 (come rewrite.run)
# File del generatore: REWRITES/{fmt}.jsonl con {"id", "text", "ok", "reason", "attempt", "ratio"}.
# reason: "" ok, "format" (manca 'The answer is N.'), "answer", "steps", "length".
REWRITE7 = {fmt: REWRITES / f"{fmt}.jsonl" for fmt in FORMATS7}
# SFT: SFT/{fmt}_train.jsonl con {"id", "question", "solution", "answer"}: i soli id di A_train[:-DEV_SIZE] validi.
# Parole di relazione da conservare (riportate nelle statistiche dei dati).
RELATION_WORDS = ("so", "then", "because", "of", "from", "with", "per", "each", "left", "more", "less", "times",
                  "total", "twice", "half")
A2_MAX_WORDS = 8                          # parole per riga di A2 oltre al conto (solo indicativo nel prompt)

GEN_SYSTEM7 = {
    "A1": (
        "You rewrite step-by-step solutions of grade-school math problems in concise, grammatical English. Rules: "
        "keep exactly the same steps, numbers, arithmetic and final answer; one short sentence per step; keep every "
        "computation written as 'x op y = z' with + - * / ; keep the connectives (so, then, because) and the words "
        "that state how quantities relate (of, from, with, per, each, left, more, less, times, total, twice, half); "
        "remove repetitions of the question, filler phrases ('Let's think', 'First we need to find', 'We know "
        "that') and sentences that only restate a given number; use pronouns instead of repeating names; never "
        "invent variable names; the last line is exactly 'The answer is N.' Output only the solution."
    ),
    "A2": (
        "You rewrite step-by-step solutions of grade-school math problems as very short draft notes. Rules: one "
        "line per step; each line is a short phrase of at most 8 words, then ':' and the computation 'x op y = z' "
        "with + - * / ; keep exactly the same steps, numbers, arithmetic and final answer; keep the words that "
        "state how quantities relate (of, from, with, per, each, left, more, less, times, total, twice, half); no "
        "full sentences, no filler, no lines that only restate a given number; use plain English words, never "
        "invented variable names, underscores or abbreviations; the last line is exactly 'The answer is N.' "
        "Output only the solution."
    ),
}
GEN_USER7 = "Problem: {question}\n\nSolution:\n{solution}"
# Esempi: la soluzione di ingresso e' quella di SFT/A_train.jsonl per lo stesso id (la stessa che il training usa).
FEWSHOT_IDS7 = ("train-5", "train-2", "train-3")
FEWSHOT7 = {
    "A1": {
        "train-5": "There are 80/100 * 10 = 8 more purple flowers than yellow.\n"
                   "So there are 10 + 8 = 18 purple flowers.\n"
                   "Purple and yellow total 10 + 18 = 28.\n"
                   "Green flowers are 25/100 * 28 = 7.\n"
                   "So he has 28 + 7 = 35 plants in total.\n"
                   "The answer is 35.",
        "train-2": "She has half of the $100, so 100 / 2 = 50.\n"
                   "Her grandparents gave twice her parents' 15, so 15 * 2 = 30.\n"
                   "She has 50 + 15 + 30 = 95 in total.\n"
                   "She still needs 100 - 95 = 5.\n"
                   "The answer is 5.",
        "train-3": "Today she read 12 * 2 = 24 pages.\n"
                   "So she read 12 + 24 = 36 in total.\n"
                   "Then 120 - 36 = 84 pages are left.\n"
                   "Tomorrow she reads half: 84 / 2 = 42.\n"
                   "The answer is 42.",
    },
    "A2": {
        "train-5": "Purple, more than yellow: 80/100 * 10 = 8.\n"
                   "Purple: 10 + 8 = 18.\n"
                   "Purple and yellow: 10 + 18 = 28.\n"
                   "Green: 25/100 * 28 = 7.\n"
                   "Total: 28 + 7 = 35.\n"
                   "The answer is 35.",
        "train-2": "Half of 100: 100 / 2 = 50.\n"
                   "Twice parents: 15 * 2 = 30.\n"
                   "Total: 50 + 15 + 30 = 95.\n"
                   "Still needed: 100 - 95 = 5.\n"
                   "The answer is 5.",
        "train-3": "Today: 12 * 2 = 24.\n"
                   "Total: 12 + 24 = 36.\n"
                   "Left: 120 - 36 = 84.\n"
                   "Half of left: 84 / 2 = 42.\n"
                   "The answer is 42.",
    },
}

# --- Condizioni nuove (si aggiungono a CONDITIONS di spec6: stessa forma). Calcolatrice = quella di A (mode "A").
CONDITIONS7 = {
    9: dict(fmt="A1", adapter="v7", grammar=False, calc=False),
    10: dict(fmt="A1", adapter="v7", grammar=False, calc=True),
    11: dict(fmt="A2", adapter="v7", grammar=False, calc=False),
    12: dict(fmt="A2", adapter="v7", grammar=False, calc=True),
}
COND7_LABELS = {9: "A1", 10: "A1+calc", 11: "A2", 12: "A2+calc"}
# Riprese da v6 (runs, output, latenze): small/large 1, 2, 3, 5; xl 2 e 8.
REUSED_V6 = {"small": (1, 2, 3, 5), "large": (1, 2, 3, 5), "xl": (2, 8)}
NEW_CONDS = {"small": (9, 10, 11, 12), "large": (9, 10, 11, 12)}   # xl: 5 e la migliore tra 10 e 12
CALC_AUDIT_MIN = 0.90                     # quota di '=' su cui la calcolatrice A deve scattare
COMPARISONS7 = ((2, 10), (2, 12), (5, 10), (5, 12), (10, 12), (9, 10), (11, 12), (1, 9), (1, 11), (2, 5), (2, 8),
                (8, 10), (8, 12), (3, 9), (3, 11))
# Run id delle valutazioni: f"{model}_c{cond}_s{seed}" in RESULTS7/outputs (come v6).
# Adapter v7: RUNS/{model}_{A1|A2}_lr0.0004_s{seed}/adapter; xl C5: RUNS/xl_C_lr0.0004_s0/adapter.
# Selezione dei dati SFT: "strict" = solo tentativi entro MAX_RATIO; "soft" = anche il piu' corto bocciato solo per
# lunghezza (concise7.select7). Deciso dopo la generazione: vedi REPORT_v7.md.
SELECT7 = "soft"
