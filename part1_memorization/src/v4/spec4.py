"""Esperimento v4: regolarità vs lunghezza (disegno 2x2). Costanti condivise, senza logica."""
from __future__ import annotations

from src.config import RESULTS_DIR, RUNS_DIR, ModelSize
from src.v2.spec import RESULT_COLUMNS_V2, SIZES_V2
from src.v3.spec3 import E_QUESTION_TEMPLATE

V4_RUNS = RUNS_DIR / "v4"
V4_RESULTS = RESULTS_DIR / "v4"

VOCAB = "bpe2048"          # suffisso dei corpora: f"{formato}@{VOCAB}"
VOCAB_SIZE = 2048
VOCAB_PAD = 64             # embedding e testa arrotondati al multiplo superiore (righe mai viste nei dati)

# Taglie nuove nella zona di transizione (head_dim multiplo di 8). 0.52M: 13 teste da 8.
SIZES_V2.setdefault("0.45M", ModelSize("0.45M", 4, 96, 6, 2e-3))
SIZES_V2.setdefault("0.52M", ModelSize("0.52M", 4, 104, 13, 2e-3))
F_SIZES = ("0.15M", "0.20M", "0.31M", "0.38M", "0.45M", "0.52M", "0.60M", "0.79M")
F_SEEDS = (0, 1, 2)

# formato -> (stile sorgente, stripper). Sorgente "pieno" = italiano_pieno v2/v3; "canonico" = un template per attributo.
F_FORMATS = {
    "italiano_pieno": ("pieno", None),
    "italiano_canonico": ("canonico", None),
    "telegrafico_lista": ("pieno", "lista"),
    "telegrafico_canonico": ("canonico", "lista"),
}
F_ORDER = ("italiano_pieno", "italiano_canonico", "telegrafico_canonico", "telegrafico_lista")   # ordine di esecuzione
VARIO = {"italiano_pieno": True, "italiano_canonico": False, "telegrafico_lista": True, "telegrafico_canonico": False}
STRIPPATO = {"italiano_pieno": False, "italiano_canonico": False, "telegrafico_lista": True, "telegrafico_canonico": True}

# italiano_canonico: indice del template in src.data.PIENO[attr]. Sono i template delle domande della Fase E v3:
# il prefisso resta univoco anche dopo lo stripping a lista, quindi la domanda e' la stessa nei 4 formati.
CANON_TEMPLATE = dict(E_QUESTION_TEMPLATE)

N_EPOCHS = 201             # epoche per corpus (E200 a pari passi usa le epoche 0..199)
LR_CHECK_FORMAT = "italiano_canonico"
LR_CHECK_SIZE = "0.38M"
LR_CHECK_MULTS = (1.0, 2.0, 4.0)

# F_main.csv: colonne v2 + formato e conteggi con padding (params_total e vocab usano il V reale)
F_COLUMNS = (*RESULT_COLUMNS_V2, "format", "vocab_padded", "params_total_padded")
# F_forms.csv: una riga per formato e attributo
#   n_forms: forme distinte della frase (valore e nome sostituiti), entrambi i generi, prima frase e non
#   n_forms_core: forme distinte per genere M nelle frasi non iniziali
#   entropy_bits: media per biografia di -log2 p(forma | attributo, genere, prima frase); la somma sugli attributi = bit/bio
FORMS_COLUMNS = ("format", "attr", "n_forms", "n_forms_core", "entropy_bits")
