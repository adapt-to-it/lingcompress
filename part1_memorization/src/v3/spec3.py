"""Esperimento v3: costanti condivise (senza logica)."""
from __future__ import annotations

from src.config import DATA_DIR, RESULTS_DIR, RUNS_DIR

V3_DATA = DATA_DIR / "v3"
V3_RUNS = RUNS_DIR / "v3"
V3_RESULTS = RESULTS_DIR / "v3"

LOG_EVERY = 200        # unica sincronizzazione periodica del training (loss.item)
DONE_CHECK_EVERY = 4   # generazione: controllo di fine (sync) ogni N passi di decodifica

# ---------------- Fase E: stripper e realizzatore a regole ----------------
E_SIZES = ("0.15M", "0.20M", "0.31M", "0.38M", "0.60M", "0.79M")   # telegrafico_auto, 2 seed
PIPE_SIZES = ("0.60M", "1.49M", "4M")                              # misura pipeline, seed 0
PIPE_QUESTIONS = 1000            # prime 1000 domande: persone 0.., 6 attributi ciascuna
PIPE_BATCHES = (1, 64)
E_FORMATS = {"telegrafico_auto": "spacy", "telegrafico_lista": "lista"}   # formato -> stripper
SPACY_MODEL = "it_core_news_sm"
# Tempi in ms: per richiesta con batch=1, per lotto con batch=64 (convenzione Fase C)
PIPE_COLUMNS = (
    "format", "size", "batch", "n_questions", "prompt_tokens_mean", "answer_tokens_mean",
    "strip_cpu_ms", "core_gpu_ms", "realize_cpu_ms", "total_ms", "pieno_ms", "ratio_vs_pieno",
    "core_flops", "pieno_flops", "core_acc", "e2e_acc", "pieno_acc",
)

# ---------------- Fase D: sweep del vocabolario ----------------
D_FORMATS = ("italiano_pieno", "telegrafico", "interlingua_v1enc")
D_VOCABS = ("bpe512", "bpe1024", "char", "bpe2048")   # bpe4096: run v2 riusati (A_main, A_abl)
D_SIZES = E_SIZES
LR_CHECK_SIZE = "0.38M"
LR_CHECK_MULTS = (1.0, 2.0, 4.0)                      # attorno al 2x scelto in v2 a 0.38M
LAT_COLUMNS = (
    "corpus", "format", "vocab", "size", "seed", "batch", "n_questions", "params_nonemb", "params_total",
    "prompt_tokens_mean", "answer_tokens_mean", "latency_ms_mean", "ms_per_token", "answers_per_s",
)

# Stripper a lista: parole eliminate (minuscole; il confronto ignora maiuscole)
STRIP_WORDS = frozenset("""
il lo la i gli le l' un uno una un'
è e' ha hanno ho hai abbiamo avete sono era erano fu sia
io tu lui lei noi voi loro egli ella
di a da in con su per tra fra presso
del dello della dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle
nel nello nella nei negli nelle sul sullo sulla sui sugli sulle col coi
""".split())

# Domande della Fase E: indice del template in src.data.PIENO[attr] il cui prefisso, dopo lo stripping
# (spacy e lista, entrambi i generi), resta non vuoto e diverso dai prefissi degli altri attributi.
# Con il template 0 data/luogo diventano entrambi "nato" e professione/azienda "lavora".
E_QUESTION_TEMPLATE = {"nascita_data": 3, "nascita_luogo": 1, "universita": 0, "corso": 0,
                       "professione": 2, "azienda": 0}
