"""Esperimento v2: costanti, griglie e tipi condivisi (senza logica)."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import ATTRS, DATA_DIR, RESULTS_DIR, RUNS_DIR, ModelSize

V2_DATA = DATA_DIR / "v2"
V2_RUNS = RUNS_DIR / "v2"
V2_RESULTS = RESULTS_DIR / "v2"

N_PERSONS = 20000
DATA_SEED = 1234            # stesse persone di v1 (src.data.generate_persons)
LANG_SEED = 777             # assegnazione lingua / persone ponte
BIOS_PER_STEP = 512         # protocollo "passi": biografie per passo, uguale per ogni corpus
VOCAB_SIZE = 4096
LR_MULTS = (0.5, 1.0, 2.0)
BRIDGE_FRAC = 0.10
EVAL_PERSONS = 5000         # persone valutate (le prime per indice), 6 attributi ciascuna
MAX_NEW_TOKENS = 24

# Griglia taglie: 8 log-spaziate 0,15M-1,5M non-embedding + 4M di controllo.
# head_dim = 16 (multiplo di 8: abilita i kernel flash di SDPA).
# lr base = lr v1 della taglia v1 più vicina in log-parametri.
SIZES_V2 = {
    "0.15M": ModelSize("0.15M", 3, 64, 4, 2e-3),
    "0.20M": ModelSize("0.20M", 4, 64, 4, 2e-3),
    "0.31M": ModelSize("0.31M", 4, 80, 5, 2e-3),
    "0.38M": ModelSize("0.38M", 5, 80, 5, 2e-3),
    "0.60M": ModelSize("0.60M", 4, 112, 7, 2e-3),
    "0.79M": ModelSize("0.79M", 4, 128, 8, 1.5e-3),
    "0.99M": ModelSize("0.99M", 4, 144, 9, 1.5e-3),
    "1.49M": ModelSize("1.49M", 4, 176, 11, 1.5e-3),
    "4M": ModelSize("4M", 4, 288, 18, 1e-3),
}

# Stili superficiali di una biografia
STYLES = ("pieno", "telegrafico", "interlingua", "interlingua_v1", "ricco")
LANGS = ("it", "en", "xx")  # "xx" = interlingua


@dataclass(frozen=True)
class Part:
    style: str      # uno di STYLES
    lang: str       # "it" | "en" (solo per style "pieno") | "xx"
    persons: str    # "all" | "lang": persone assegnate a questa lingua
                    # | "bridge": persone ponte assegnate all'ALTRA lingua (le aggiunge in questa)


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    parts: tuple            # tuple[Part, ...]: il corpus di un'epoca è l'unione delle parti
    tokenizer: str          # chiave del BPE: corpora con la stessa chiave condividono il tokenizer
    split_digits: bool = True
    cross: bool = False     # valuta anche la recall cross-lingua (it <-> en)


def _mono(name: str, style: str, lang: str, tok: str | None = None, digits: bool = True) -> CorpusSpec:
    return CorpusSpec(name, (Part(style, lang, "all"),), tok or name, digits)


class _Corpora(dict):
    """Crea su richiesta le varianti v3 "formato@vocab" (anche nei worker spawn, che non le hanno registrate)."""

    def __missing__(self, key):
        base, sep, var = key.partition("@")
        if sep and var in ("bpe512", "bpe1024", "bpe2048", "char") and base in self:
            spec = CorpusSpec(key, self[base].parts, key, True, self[base].cross)
            self[key] = spec
            return spec
        raise KeyError(key)


CORPORA = _Corpora({
    # Fase A
    "italiano_pieno": _mono("italiano_pieno", "pieno", "it"),
    "telegrafico": _mono("telegrafico", "telegrafico", "it"),
    "interlingua": _mono("interlingua", "interlingua", "xx"),
    "italiano_ricco": _mono("italiano_ricco", "ricco", "it"),
    "italiano_pieno@shared": _mono("italiano_pieno@shared", "pieno", "it", "shared"),
    "interlingua@shared": _mono("interlingua@shared", "interlingua", "xx", "shared"),
    # ablazioni: isolano l'effetto delle correzioni 4 e 5
    "interlingua_v1enc": _mono("interlingua_v1enc", "interlingua_v1", "xx"),
    "italiano_pieno@nodigits": _mono("italiano_pieno@nodigits", "pieno", "it", digits=False),
    # Fase B
    "multi_pieno": CorpusSpec("multi_pieno", (Part("pieno", "it", "lang"), Part("pieno", "en", "lang")),
                              "multi", cross=True),
    "multi_bridge": CorpusSpec("multi_bridge", (Part("pieno", "it", "lang"), Part("pieno", "en", "lang"),
                                                Part("pieno", "it", "bridge"), Part("pieno", "en", "bridge")),
                               "multi", cross=True),
})

# Corpora su cui si addestra un tokenizer condiviso (default: il corpus stesso)
TOKENIZER_SOURCES = {
    "shared": ("italiano_pieno", "interlingua"),
    "multi": ("multi_bridge",),
}


@dataclass(frozen=True)
class EvalCase:
    kind: str        # "in" (lingua di training) | "cross" (persona vista solo nell'altra lingua)
    person: int      # indice nella lista persone
    attr: str        # uno di ATTRS
    style: str
    lang: str
    prompt: str      # termina subito prima del valore (o del suo articolo)
    truth: str       # valore canonico = stringa v1 in Person.attrs


@dataclass(frozen=True)
class Job:
    stage: str              # es. "A_lr", "A_main", "A_exp", "A_esp", "A_shared", "A_abl", "B_main", "*_quick"
    corpus: str             # chiave di CORPORA
    protocol: str           # "passi" | "esposizioni"
    exposures: int
    size: str               # chiave di SIZES_V2
    lr_mult: float
    seed: int
    n_persons: int = N_PERSONS
    eval_persons: int = EVAL_PERSONS

    @property
    def run_id(self) -> str:
        return (f"{self.stage}_{self.corpus}_{self.protocol}_e{self.exposures}_{self.size}"
                f"_lr{self.lr_mult:g}_s{self.seed}_n{self.n_persons}")


JOB_KEYS = ("stage", "corpus", "protocol", "exposures", "size", "lr_mult", "seed", "n_persons")

RESULT_COLUMNS_V2 = (
    *JOB_KEYS, "eval_persons",
    "n_layer", "n_embd", "n_head", "lr", "params_nonemb", "params_total", "vocab",
    "tokens_per_bio", "bios_per_step", "steps", "train_tokens", "flops",
    "train_time_s", "steady_tokens_per_s", "steady_facts_per_s", "peak_vram_mb",
    "final_loss", "eval_time_s",
    *(f"acc_{a}" for a in ATTRS), "acc_mean",
    *(f"cross_{a}" for a in ATTRS), "cross_mean", "cross_lenient_mean",
    "bits_memorized", "bits_per_param",
)

# ---------------- Fase C: costi di training e di risposta end-to-end ----------------
C_FORMATS = ("italiano_pieno", "telegrafico", "interlingua")
C_SIZES = ("0.60M", "1.49M", "4M")          # modelli core di A_main, seed 0
C_QUESTIONS = 1000                           # prime 1000 domande "in" di eval_cases
C_BATCHES = (1, 64)
C_TRANSLATE_SIZE = "0.99M"                   # stripper e traduttore (~1M non-embedding)
C_TRANSLATE_TRAIN_PERSONS = (5000, 20000)    # persone di training dei traduttori (eval: persone 0..)
C_MAX_OUT_TOKENS = 64

# Tempi in ms; per batch=64 i tempi sono per batch (attesa di ogni utente del batch).
INFER_COLUMNS = (
    "corpus", "size", "seed", "batch", "n_questions", "params_nonemb", "params_total",
    "prompt_tokens_mean", "answer_tokens_mean",
    "prefill_ms", "ttft_ms", "decode_ms", "latency_ms_mean", "latency_ms_p50", "latency_ms_p90",
    "ms_per_decode_token", "answers_per_s",
    "flops_prefill", "flops_decode", "flops_per_answer", "acc",
)
TRANSLATE_COLUMNS = (
    "model", "format", "batch", "n_questions", "params_nonemb", "params_total",
    "train_steps", "train_time_s", "train_flops",
    "in_tokens_mean", "out_tokens_mean", "sentence_em", "values_em",
    "prefill_ms", "decode_ms", "latency_ms_mean", "answers_per_s", "flops_per_request",
)
E2E_COLUMNS = (
    "format", "size", "batch", "n_questions",
    "stripper_ms", "core_ms", "translator_ms", "pipeline_ms", "pieno_core_ms", "speedup_vs_pieno",
    "pipeline_flops", "pieno_flops", "pipeline_acc", "pieno_acc",
)
