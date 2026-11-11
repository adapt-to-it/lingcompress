"""Fase D v3: tokenizer e cache delle 201 epoche per formato x vocabolario (solo CPU), nell'ordine dei blocchi."""
import time
from src.v2 import corpus, train
from src.v3 import vocab
from src.v3.spec3 import D_FORMATS, D_VOCABS

if __name__ == "__main__":
    persons = corpus.load_persons(20000)
    for v in D_VOCABS:
        for f in D_FORMATS:
            t0 = time.perf_counter()
            spec = vocab.register(f, v)
            if not train._tok_path(spec, len(persons)).exists():
                vocab.build_tokenizer(spec, v, persons)
            train.prepare(spec, persons, 201)
            print(spec.name, "pronto in %.0f s" % (time.perf_counter() - t0), flush=True)
