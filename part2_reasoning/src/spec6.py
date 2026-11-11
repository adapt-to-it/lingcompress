"""Specifica condivisa di v6: grammatica C6, regex di riga per la decodifica vincolata, condizioni di valutazione."""
from src.spec5 import C_NAME_RE, C_NUM_RE, RESULTS

RESULTS6 = RESULTS.parent / "v6"
LR6 = 4e-4                               # lr di v5 per tutti
SEEDS6 = {"small": (0, 1, 2), "large": (0, 1), "xl": (0,)}

# --- Unita' di C6: ultimo segmento di ogni NAME. Le velocita'/tariffe mettono il denominatore nel nome
# e l'unita' del valore in fondo: pay_per_h_usd, speed_per_h_mi. Conteggi puri: _n; fattori/frazioni: _ratio.
UNITS = ("usd", "pct", "n", "ratio", "s", "min", "h", "days", "weeks", "months", "years",
         "mi", "km", "m", "ft", "in", "lb", "oz", "kg", "l", "gal")
NOTE_MAX_WORDS = 12
NOTE_MAX_LINES = 2
PCT_RANGE = (0, 100)                     # GIVEN/FACT/calc con nome _pct: valore in [0, 100]

# --- Regex di riga (stringhe; usarle con il modulo `regex`, anche in modalita' partial per la decodifica).
NUM = C_NUM_RE                                           # -?[0-9]+(\.[0-9]+)?
EXPR = r"[0-9(\-][0-9+\-*/(). ]*"                        # come C5: solo numeri, operatori, parentesi, spazi
CMP = r"(?:>=|<=|==|>|<)"
NAME5 = C_NAME_RE
NAME6 = r"[a-z][a-z0-9]*(?:_[a-z0-9]+){0,4}_(?:" + "|".join(UNITS) + r")"
EQSIDE = r"[0-9a-z_(\-][0-9a-z_+\-*/(). ]*"              # lato di un'equazione SOLVE: numeri, operatori, l'incognita
NOTE = r"NOTE [A-Za-z0-9,.'%$/\-]+(?: [A-Za-z0-9,.'%$/\-]+){0," + str(NOTE_MAX_WORDS - 1) + r"}"


def line_regex(fmt):
    """Regex (stringa) di una riga valida per fmt in {"C5", "C6"}; i gruppi nominati servono ai controlli dinamici."""
    name = NAME5 if fmt == "C5" else NAME6
    alts = [rf"(?:GIVEN|FACT) (?P<gname>{name}) = {NUM}",
            rf"(?P<cname>{name}) = (?P<expr>{EXPR}) = {NUM}",
            rf"IF {EXPR} {CMP} {EXPR}",
            rf"ANSWER = {NUM}"]
    if fmt == "C6":
        alts += [rf"SOLVE (?P<sname>{name}): (?P<eq>{EQSIDE} = {EQSIDE}) -> (?P<sname2>{name}) = {NUM}", NOTE]
    return "(?:" + "|".join(alts) + ")"


ANSWER_LINE = rf"ANSWER = {NUM}"
# Inizio di riga che definisce un nome (per vietare le ridefinizioni gia' durante la decodifica).
DEF_PREFIX = r"^(?:(?:GIVEN|FACT) )?([a-z][a-z0-9_]*) =|^SOLVE ([a-z][a-z0-9_]*):"
# Calcolatrice C5/C6: la riga corrente termina con "NAME = EXPR =" oppure "SOLVE NAME: EQ -> NAME =".
CALC_TRIGGER = r"^[a-z][a-z0-9_]* = (?P<expr>" + EXPR + r") =$"
SOLVE_TRIGGER = r"^SOLVE (?P<u>[a-z][a-z0-9_]*): (?P<eq>" + EQSIDE + " = " + EQSIDE + r") -> (?P=u) =$"
# Calcolatrice A: il testo generato termina con un'espressione aritmetica (almeno un operatore) e " =".
A_CALC_TRIGGER = (r"(?<![\w.])(?P<expr>\(?\$?\d[\d,]*(?:\.\d+)?%?\)?(?:\s*[-+*/×x]\s*\(?\$?\d[\d,]*(?:\.\d+)?%?\)?)+)"
                  r"\s*=$")
# Valore forzato: intero se |v - round(v)| < 1e-9; altrimenti 2 decimali (4 cifre significative se |v| < 0.01),
# senza zeri finali. I token forzati sono tok.encode(" " + valore) (Qwen: " " e cifre singole).

# --- Problemi "con incognite" (soluzione ufficiale algebrica): sul train l'83% fallisce la riscrittura C5
# contro il 7% degli altri. Applicare alla parte prima di "####" della soluzione ufficiale.
ALGEBRA_RE = (r"\b[Ll]et\s+[A-Za-z]\s+(?:be|=|represent)|(?<![\w$.])\d+(?:\.\d+)?[A-Za-z](?![\w])"
              r"|(?<![\w'’])[A-Za-z]\s*=\s*\d")

# --- Condizioni di valutazione (fmt: "A", "C5" = formato C di v5, "C6").
CONDITIONS = {
    1: dict(fmt="A", adapter="v5", grammar=False, calc=False),
    2: dict(fmt="A", adapter="v5", grammar=False, calc=True),
    3: dict(fmt="C5", adapter="v5", grammar=False, calc=False),
    4: dict(fmt="C5", adapter="v5", grammar=True, calc=False),
    5: dict(fmt="C5", adapter="v5", grammar=True, calc=True),
    6: dict(fmt="C6", adapter="v6", grammar=False, calc=False),
    7: dict(fmt="C6", adapter="v6", grammar=True, calc=False),
    8: dict(fmt="C6", adapter="v6", grammar=True, calc=True),
}
REUSED_V5 = (1, 3)                                       # risultati v5 copiati, non rivalutati
# run_id di una valutazione v6: f"{model}_c{cond}_s{seed}"; adapter v5: RUNS/{model}_{A|C}_lr0.0004_s{seed}/adapter,
# adapter v6: RUNS/{model}_C6_lr0.0004_s{seed}/adapter. Output: RESULTS6/outputs/{run_id}_test.jsonl.
OUTPUT6_FIELDS = ("id", "output", "n_tokens", "n_forced", "pred", "gold", "correct", "conform")
RUN6_COLUMNS = ("run_id", "model", "cond", "format", "grammar", "calc", "seed", "split", "n_eval", "accuracy",
                "conform_rate", "tok_mean", "tok_median", "forced_mean", "throughput_time_s", "answers_per_s",
                "tokens_per_s", "note_rate", "solve_rate")
LATENCY6_COLUMNS = ("run_id", "model", "cond", "format", "n", "lat_mean_s", "lat_median_s", "ms_per_token_mean",
                    "tok_mean")
COMPARISONS = ((3, 4), (4, 5), (5, 8), (3, 6), (6, 7), (7, 8), (1, 2), (2, 8), (1, 8), (2, 5))

# --- Generatore C6 (llama.cpp, Qwen3-4B-Instruct Q8): prompt e 4 esempi.
GEN_SYSTEM_C6 = (
    "You rewrite official solutions of grade-school math problems in a controlled canonical language. Grammar, "
    "one line per step, no other text:\n"
    "  name = expression = value              an arithmetic step\n"
    "  GIVEN name = value                     a quantity stated in the problem, reused without computation\n"
    "  FACT name = value                      a common-knowledge constant (hour_min = 60, week_days = 7)\n"
    "  IF expression > expression             a true comparison that decides which case applies (> < >= <= ==)\n"
    "  SOLVE name: equation -> name = value   a linear equation in ONE unknown (the name), solved\n"
    "  NOTE short text                        at most 12 words, at most 2 per solution, only for a choice or a "
    "conversion the other lines cannot express\n"
    "  ANSWER = value                         the last line\n"
    "Names: short lower_snake_case quantity names that END with the unit of the value, one of: "
    + ", ".join("_" + u for u in UNITS) + ". Use _n for plain counts, _ratio for multipliers and fractions, "
    "_usd for money. For rates put the denominator in the name and the unit last: pay_per_h_usd, speed_per_h_mi. "
    "Each name is defined once and must describe exactly that quantity. Percentages are _pct names with values "
    "0-100 and are applied only inside an explicit expression x * p / 100. Expressions contain only numbers, "
    "+ - * / and parentheses, never names (except the unknown in SOLVE). Values are plain numbers without units, "
    "$, % or commas; every expression must evaluate exactly to its value. Keep the same steps, order and final "
    "answer as the official solution. Output only the lines."
)
FEWSHOT_IDS_C6 = ("train-1", "train-5", "train-10", "train-3")
FEWSHOT_C6 = {
    "train-1": "FACT hour_min = 60\n"
               "pay_per_min_usd = 12 / 60 = 0.2\n"
               "earnings_usd = 0.2 * 50 = 10\n"
               "ANSWER = 10",
    "train-5": "purple_extra_n = 10 * 80 / 100 = 8\n"
               "purple_flowers_n = 10 + 8 = 18\n"
               "yellow_purple_n = 10 + 18 = 28\n"
               "green_flowers_n = 28 * 25 / 100 = 7\n"
               "total_flowers_n = 28 + 7 = 35\n"
               "ANSWER = 35",
    "train-10": "NOTE each ship carries twice the people of the previous one\n"
                "third_ship_ratio = 2 * 2 = 4\n"
                "SOLVE first_ship_n: first_ship_n + 2 * first_ship_n + 4 * first_ship_n = 847 -> first_ship_n = 121\n"
                "ANSWER = 121",
    "train-3": "pages_today_n = 12 * 2 = 24\n"
               "pages_read_n = 12 + 24 = 36\n"
               "pages_left_n = 120 - 36 = 84\n"
               "pages_tomorrow_n = 84 / 2 = 42\n"
               "ANSWER = 42",
}
COND_LABELS = {1: "A", 2: "A+calc", 3: "C5", 4: "C5+gram", 5: "C5+gram+calc", 6: "C6", 7: "C6+gram",
               8: "C6+gram+calc"}

# v7: condizioni 9-12 (prosa concisa A1 / A2), stessa forma delle precedenti.
from src.spec7 import COND7_LABELS, CONDITIONS7  # noqa: E402

CONDITIONS.update(CONDITIONS7)
COND_LABELS.update(COND7_LABELS)
