"""Esegue la griglia di run e accoda ogni risultato a results.csv (o results_<tag>.csv)."""
from __future__ import annotations

import argparse
import csv
import time
from datetime import timedelta
from pathlib import Path

from src.config import (FORMATS, MODEL_SIZES, RESULT_COLUMNS, RESULTS_DIR, ExpConfig,
                        full_config, quick_config)
from src.data import generate_persons
from src.pipeline import run_one
from src.report import write_all


def pilot_config() -> ExpConfig:
    cfg = full_config()
    cfg.formats = ("italiano_pieno",)
    cfg.sizes = ("0.25M", "8M")
    cfg.tag = "pilot"
    return cfg


def _lista(valore: str) -> tuple:
    return tuple(v.strip() for v in valore.split(",") if v.strip())


def _seeds(valore: str) -> tuple:
    return tuple(int(v) for v in _lista(valore))


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Griglia di run dell'esperimento.")
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--quick", action="store_true", help="preset quick")
    modo.add_argument("--full", action="store_true", help="preset full")
    modo.add_argument("--pilot", action="store_true", help="pilota: italiano_pieno, 0.25M e 8M")
    p.add_argument("--report", action="store_true", help="solo report dal CSV del tag, senza run")
    p.add_argument("--n", type=int, help="numero di persone")
    p.add_argument("--exposures", type=int, help="passate su ogni biografia")
    p.add_argument("--sizes", type=_lista, help="taglie separate da virgola, es. 0.25M,1M")
    p.add_argument("--formats", type=_lista, help="formati separati da virgola")
    p.add_argument("--seeds", type=_seeds, help="seed separati da virgola, es. 0,1")
    p.add_argument("--tag", help="suffisso del CSV e dei run")
    args = p.parse_args()
    if not (args.quick or args.full or args.pilot or args.report):
        p.error("scegli --quick, --full, --pilot oppure --report")
    return args


def _csv_path(tag: str) -> Path:
    return RESULTS_DIR / ("results.csv" if tag == "full" else f"results_{tag}.csv")


def _chiave(tag, fmt, size, seed, n_persons, exposures) -> tuple:
    return (str(tag), str(fmt), str(size), str(seed), str(n_persons), str(exposures))


def _eseguiti(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {_chiave(r["tag"], r["format"], r["size"], r["seed"], r["n_persons"], r["exposures"])
                for r in csv.DictReader(f)}


def _accoda(path: Path, riga: dict) -> None:
    nuovo = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if nuovo:
            writer.writeheader()
        writer.writerow(riga)


def main() -> None:
    args = _parse()
    if args.pilot:
        cfg = pilot_config()
    elif args.quick:
        cfg = quick_config()
    else:
        cfg = full_config()
    if args.n is not None:
        cfg.n_persons = args.n
    if args.exposures is not None:
        cfg.exposures = args.exposures
    if args.sizes is not None:
        cfg.sizes = args.sizes
    if args.formats is not None:
        cfg.formats = args.formats
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.tag is not None:
        cfg.tag = args.tag
    for taglia in cfg.sizes:
        if taglia not in MODEL_SIZES:
            raise SystemExit(f"taglia sconosciuta: {taglia}")
    for fmt in cfg.formats:
        if fmt not in FORMATS:
            raise SystemExit(f"formato sconosciuto: {fmt}")

    csv_path = _csv_path(cfg.tag)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        write_all(csv_path, RESULTS_DIR)
        return

    persone = generate_persons(cfg.n_persons, cfg.data_seed)
    eseguiti = _eseguiti(csv_path)
    piano = [(fmt, MODEL_SIZES[taglia], seed)
             for fmt in cfg.formats for taglia in cfg.sizes for seed in cfg.seeds]
    da_fare = [(fmt, size, seed) for fmt, size, seed in piano
               if _chiave(cfg.tag, fmt, size.name, seed, cfg.n_persons, cfg.exposures) not in eseguiti]
    print(f"{len(da_fare)} run da eseguire, {len(piano) - len(da_fare)} gia' presenti in {csv_path}")

    tempi = []
    for i, (fmt, size, seed) in enumerate(da_fare, 1):
        t0 = time.perf_counter()
        riga = run_one(persone, fmt, size, seed, cfg)
        tempi.append(time.perf_counter() - t0)
        _accoda(csv_path, riga)
        restanti = len(da_fare) - i
        eta = sum(tempi) / len(tempi) * restanti
        print(f"[{i}/{len(da_fare)}] {cfg.tag} {fmt} {size.name} seed {seed}: "
              f"acc_mean {riga['acc_mean']:.3f}, ETA {timedelta(seconds=int(eta))}")


if __name__ == "__main__":
    main()
