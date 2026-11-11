"""Specifica condivisa di v5: percorsi, modelli, formati, grammatica del canonico, stripper e prompt del generatore."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GSM8K = DATA / "gsm8k"            # train.jsonl / test.jsonl: {"id", "question", "answer"}
REWRITES = DATA / "rewrites"      # {fmt}.jsonl dal generatore: {"id", "text", "ok", "reason", "attempt"}
SFT = DATA / "sft"                # {fmt}_{split}.jsonl: {"id", "question", "solution", "answer"}
RESULTS = ROOT / "results" / "v5"
RUNS = ROOT / "runs"

MODELS = {"small": "Qwen/Qwen3-0.6B-Base", "large": "Qwen/Qwen3-1.7B-Base", "xl": "Qwen/Qwen3-4B-Base"}
QLORA = {"xl"}                   # caricati in 4 bit NF4 (bitsandbytes), adapter non fuso in valutazione
GENERATOR = "Qwen/Qwen3-4B-Instruct-2507"
FORMATS = ("A", "B", "C", "D")    # A pieno, B telegrafico (strip di A), C canonico, D solo risposta
FORMAT_NAMES = {"A": "pieno", "B": "telegrafico", "C": "canonico", "D": "solo risposta"}

# Prompt di SFT e di valutazione, identico per tutti i formati; la loss e' solo su solution + EOS.
PROMPT = "Question: {question}\nAnswer:\n"

# Riga finale di risposta per formato (D: l'intero output e' il numero).
A_FINAL = "The answer is {value}."
B_FINAL = "answer {value}."        # = strip_b(A_FINAL)
C_FINAL = "ANSWER = {value}"

# --- Stripper B: parole intere tolte (case-insensitive). Numeri, operatori, punteggiatura e le parole KEEP restano.
STRIP_ARTICLES = ("a", "an", "the")
STRIP_AUX = ("is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has", "have", "had",
             "will", "would", "shall", "should", "can", "could", "may", "might")
STRIP_PRONOUNS = ("i", "you", "he", "she", "it", "we", "they")
STRIP_PREPS = ("of", "to", "in", "on", "at", "by", "with", "from", "for")
STRIP_WORDS = frozenset(STRIP_ARTICLES + STRIP_AUX + STRIP_PRONOUNS + STRIP_PREPS)
KEEP_WORDS = frozenset(("so", "then", "because", "each", "per", "left", "more", "less", "times", "total"))
# Contrazioni soggetto+ausiliare (she's, they're, it's, he'll ...) sono tolte per intero.

# --- Grammatica del canonico C (una riga per passaggio, nessuna riga vuota):
#   calc    := NAME " = " EXPR " = " NUM          passaggio aritmetico
#   given   := "GIVEN " NAME " = " NUM            quantita' del testo ripresa senza calcolo
#   fact    := "FACT " NAME " = " NUM             costante di conoscenza comune (60 minuti in un'ora, 7 giorni ...)
#   check   := "IF " EXPR CMP EXPR                condizione vera che sceglie il caso (CMP in > < >= <= ==)
#   final   := "ANSWER = " NUM                    ultima riga, una sola volta
#   NAME    := [a-z][a-z0-9]* ("_" [a-z0-9]+){0,4}  definito una sola volta
#   EXPR    := numeri, + - * / e parentesi; nessun nome, nessuna unita'
#   NUM     := -?[0-9]+(\.[0-9]+)?                 niente virgole, $, %
# Verifica aritmetica: |eval(EXPR) - NUM| <= 1e-6, oppure NUM scritto con >= 2 decimali e uguale a eval arrotondato.
C_MARKERS = ("GIVEN", "FACT", "IF", "ANSWER")
C_NAME_RE = r"[a-z][a-z0-9]*(?:_[a-z0-9]+){0,4}"
C_NUM_RE = r"-?[0-9]+(?:\.[0-9]+)?"
C_CMP = (">=", "<=", "==", ">", "<")

# --- Generatore locale: prompt di riscrittura (chat), con la soluzione ufficiale.
GEN_SYSTEM = {
    "A": (
        "You rewrite official solutions of grade-school math problems as clear step-by-step reasoning in natural "
        "English. Rules: keep exactly the same steps, numbers and final answer as the official solution; one step "
        "per line; each step is a complete sentence that says what is computed and why, and shows the arithmetic "
        "as 'x op y = z' using + - * / ; do not use the <<...>> calculator notes; do not add new methods; the last "
        "line is exactly 'The answer is N.' where N is the final number without units or commas. Output only the "
        "solution."
    ),
    "C": (
        "You rewrite official solutions of grade-school math problems in a controlled canonical language. Grammar, "
        "one line per step, no other text:\n"
        "  name = expression = value      an arithmetic step\n"
        "  GIVEN name = value             a quantity stated in the problem, reused without computation\n"
        "  FACT name = value              a common-knowledge constant (minutes_per_hour = 60, days_per_week = 7)\n"
        "  IF expression > expression     a true comparison that decides which case applies (> < >= <= ==)\n"
        "  ANSWER = value                 the last line\n"
        "Rules: names are short lower_snake_case quantity names (e.g. apples_left, cost_per_day), at most 5 words, "
        "each defined once; expressions contain only numbers, + - * / and parentheses, never names; values are "
        "plain numbers without units, $, % or commas; every expression must evaluate exactly to its value; keep "
        "the same steps, order and final answer as the official solution; put a FACT line just before the first "
        "step that uses it. Output only the lines."
    ),
}
GEN_USER = "Problem: {question}\n\nOfficial solution:\n{answer}"

# Esempi few-shot (id del train, testo atteso). Gli id vanno letti da GSM8K per domanda e soluzione ufficiale.
FEWSHOT_IDS = ("train-0", "train-1", "train-3")
FEWSHOT = {
    "A": {
        "train-0": "In May, Natalia sold half as many clips as in April, so she sold 48 / 2 = 24 clips in May.\n"
                   "Altogether she sold 48 + 24 = 72 clips in April and May.\n"
                   "The answer is 72.",
        "train-1": "There are 60 minutes in an hour, so Weng earns 12 / 60 = 0.2 dollars per minute.\n"
                   "She babysat for 50 minutes, so she earned 0.2 * 50 = 10 dollars.\n"
                   "The answer is 10.",
        "train-3": "Today Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today.\n"
                   "So far she has read 12 + 24 = 36 pages in total.\n"
                   "The pages left to read are 120 - 36 = 84.\n"
                   "She wants to read half of the remaining pages tomorrow, so she should read 84 / 2 = 42 pages.\n"
                   "The answer is 42.",
    },
    "C": {
        "train-0": "clips_may = 48 / 2 = 24\n"
                   "clips_total = 48 + 24 = 72\n"
                   "ANSWER = 72",
        "train-1": "FACT minutes_per_hour = 60\n"
                   "pay_per_minute = 12 / 60 = 0.2\n"
                   "earnings = 0.2 * 50 = 10\n"
                   "ANSWER = 10",
        "train-3": "pages_today = 12 * 2 = 24\n"
                   "pages_read = 12 + 24 = 36\n"
                   "pages_left = 120 - 36 = 84\n"
                   "pages_tomorrow = 84 / 2 = 42\n"
                   "ANSWER = 42",
    },
}

# --- Training LoRA: stessi iperparametri per ogni modello x formato; stesso numero di step (stessi problemi).
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(epochs=2, batch_size=32, warmup_frac=0.03, weight_decay=0.0, max_len=640, grad_clip=1.0)
MICRO_BATCH = {"small": 6, "large": 4, "xl": 4}   # xl: QLoRA con gradient checkpointing      # oltre ~10 GB Windows riversa in RAM e lo step rallenta 5-7x
LR_SWEEP = (1e-4, 2e-4, 4e-4)               # solo A sul modello piccolo, valutato sul dev
LR_DEFAULT = 2e-4
SEEDS = (0, 1, 2)
DEV_SIZE = 300                              # ultimi DEV_SIZE problemi dell'intersezione: fuori dal training di tutti

# --- Valutazione: greedy, KV cache statica, batch; prompt PROMPT, stop su EOS.
MAX_NEW_TOKENS = {"A": 384, "B": 320, "C": 320, "D": 16}
EVAL_BATCH = 32                             # anche la misura di throughput sull'intero test
LATENCY_N = 200                             # problemi del test, campione fisso con LATENCY_SEED
LATENCY_SEED = 0

# run_id = f"{model_key}_{fmt}_lr{lr:g}_s{seed}"; adapter in RUNS/run_id/adapter, log in RUNS/run_id/train.json
# Output per problema: RESULTS/outputs/{run_id}_{split}.jsonl con OUTPUT_FIELDS.
OUTPUT_FIELDS = ("id", "output", "n_tokens", "pred", "gold", "correct", "conform")
RUN_COLUMNS = ("run_id", "model", "format", "lr", "seed", "stage", "n_train", "steps", "train_time_s", "final_loss",
               "split", "n_eval", "accuracy", "conform_rate", "tok_mean", "tok_median",
               "throughput_time_s", "answers_per_s", "tokens_per_s")
LATENCY_COLUMNS = ("run_id", "model", "format", "n", "lat_mean_s", "lat_median_s", "ms_per_token_mean", "tok_mean")
