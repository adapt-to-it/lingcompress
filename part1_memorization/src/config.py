"""Configurazione condivisa dell'esperimento (costanti e dataclass, senza logica)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # tokenizer e cache dei token per epoca
RUNS_DIR = ROOT / "runs"          # checkpoint per run
RESULTS_DIR = ROOT / "results"    # results.csv e grafici

FORMATS = ("italiano_pieno", "telegrafico", "interlingua")
# Ordine canonico degli attributi (chiavi di Person.attrs e colonne acc_<attr>)
ATTRS = ("nascita_data", "nascita_luogo", "universita", "corso", "professione", "azienda")

# Marcatori interlingua, uno per attributo (stesso ordine di ATTRS)
INTERLINGUA_MARKERS = {
    "nascita_data": "NASC_D",
    "nascita_luogo": "NASC_L",
    "universita": "UNI",
    "corso": "CORSO",
    "professione": "LAV",
    "azienda": "AZ",
}

EOS = "<eos>"
PAD = "<pad>"

# Colonne di results*.csv, nell'ordine. flops = 6 * params_total * train_tokens.
# train_tokens/tokens_per_s contano solo token non-PAD.
RESULT_COLUMNS = (
    "tag", "format", "size", "seed", "n_persons", "exposures",
    "n_layer", "n_embd", "n_head", "params_nonemb", "params_total",
    "ctx_len", "tokens_per_bio", "train_tokens", "steps", "flops",
    "train_time_s", "tokens_per_s", "peak_vram_mb", "final_loss", "eval_time_s",
    *(f"acc_{a}" for a in ATTRS), "acc_mean",
)


@dataclass(frozen=True)
class ModelSize:
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    lr: float


# ~12 * n_layer * n_embd^2 parametri non-embedding
MODEL_SIZES = {
    "0.25M": ModelSize("0.25M", 4, 72, 4, 2e-3),
    "0.5M": ModelSize("0.5M", 4, 104, 4, 2e-3),
    "1M": ModelSize("1M", 6, 120, 4, 1.5e-3),
    "2M": ModelSize("2M", 6, 168, 6, 1.5e-3),
    "4M": ModelSize("4M", 8, 208, 8, 1e-3),
    "8M": ModelSize("8M", 8, 288, 8, 1e-3),
}


@dataclass
class ExpConfig:
    mode: str = "full"
    n_persons: int = 20000
    exposures: int = 200            # passate su ogni biografia (= epoche)
    sizes: tuple = tuple(MODEL_SIZES)
    formats: tuple = FORMATS
    seeds: tuple = (0,)
    data_seed: int = 1234
    vocab_size: int = 4096
    batch_size: int = 256           # sequenze per step
    ctx_len: int = 0                # 0 = calcolata dalla biografia più lunga
    warmup_frac: float = 0.02
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    grad_clip: float = 1.0
    eval_persons: int = 0           # 0 = tutte
    eval_batch: int = 2048
    max_new_tokens: int = 16
    ckpt_every_s: float = 300.0     # checkpoint ogni N secondi
    log_every: int = 100
    tag: str = ""                   # suffisso per separare pilota/quick/full nel CSV


def quick_config() -> ExpConfig:
    return ExpConfig(mode="quick", n_persons=2000, exposures=30,
                     sizes=("0.25M", "1M", "4M"), batch_size=128, tag="quick")


def full_config() -> ExpConfig:
    return ExpConfig(mode="full", tag="full")
