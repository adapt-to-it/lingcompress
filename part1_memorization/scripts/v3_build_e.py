"""Fase E v3: genera telegrafico_auto (spaCy) e telegrafico_lista (lista) dal pieno pre-generato, 201 epoche."""
import time
from src.v2 import corpus
from src.v3 import strip
from src.v3.spec3 import V3_RESULTS

if __name__ == "__main__":
    persons = corpus.load_persons(20000)
    V3_RESULTS.mkdir(parents=True, exist_ok=True)
    strip.compare(persons, 200).to_csv(V3_RESULTS / "E_strip_compare.csv", index=False)
    for name, s in (("telegrafico_lista", strip.ListStripper()), ("telegrafico_auto", strip.SpacyStripper())):
        t0 = time.perf_counter()
        strip.register(name)
        strip.build_corpus(name, s, persons, 201)
        print(name, "costruito in %.0f s" % (time.perf_counter() - t0), flush=True)
