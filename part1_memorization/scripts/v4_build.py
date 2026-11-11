"""Costruisce tokenizer BPE 2048 ed epoche dei formati v4 (italiano_pieno@bpe2048 esiste gia' da v3)."""
import time

from src.v2 import corpus
from src.v4 import corpus4
from src.v4.spec4 import F_FORMATS, N_EPOCHS

if __name__ == "__main__":
    persone = corpus.load_persons(20000)
    specs = corpus4.register_all()
    for fmt in F_FORMATS:
        t0 = time.perf_counter()
        corpus4.build(fmt, persone, N_EPOCHS)
        print(f"{fmt} ({specs[fmt].name}) costruito in {time.perf_counter() - t0:.0f} s", flush=True)
