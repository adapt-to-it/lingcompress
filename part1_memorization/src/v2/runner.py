"""Piani di job per fase/stadio, selezione del learning rate, CSV riprendibili, stima dei tempi."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2 import corpus, train
from src.v2.spec import CORPORA, EVAL_PERSONS, LR_MULTS, N_PERSONS, RESULT_COLUMNS_V2, SIZES_V2, Job

PHASE_STAGES = {
    "A": ("A_lr", "A_main", "A_exp", "A_esp", "A_shared", "A_abl"),
    "B": ("B_main",),
}

A_CORPORA = ("italiano_pieno", "telegrafico", "interlingua", "italiano_ricco")

# Definizione degli stadi (modalita full). lr "all": tutte le LR_MULTS; "best": lr migliore dallo stadio lr.
STAGE_SPECS = {
    "A_lr": {"corpora": A_CORPORA, "protocol": "passi", "exposures": (200,), "lr": "all", "seeds": (0,)},
    "A_main": {"corpora": A_CORPORA, "protocol": "passi", "exposures": (200,), "lr": "best", "seeds": (0, 1, 2)},
    "A_exp": {"corpora": A_CORPORA, "protocol": "passi", "exposures": (100, 500), "lr": "best", "seeds": (0, 1)},
    "A_esp": {"corpora": ("telegrafico", "interlingua", "italiano_ricco"), "protocol": "esposizioni",
              "exposures": (200,), "lr": "best", "seeds": (0, 1)},
    "A_shared": {"corpora": ("italiano_pieno@shared", "interlingua@shared"), "protocol": "passi",
                 "exposures": (200,), "lr": "best", "seeds": (0, 1)},
    "A_abl": {"corpora": ("interlingua_v1enc", "italiano_pieno@nodigits"), "protocol": "passi",
              "exposures": (200,), "lr": "best", "seeds": (0, 1)},
    "B_main": {"corpora": ("multi_pieno", "multi_bridge"), "protocol": "passi", "exposures": (200,),
               "lr": "best", "seeds": (0, 1, 2)},
}

# Modalita quick
QUICK_N_PERSONS = 2000
QUICK_EVAL_PERSONS = 500
QUICK_SIZES = ("0.15M", "0.60M", "1.49M")
QUICK_SEEDS = (0,)
QUICK_LR_MULTS = (0.5, 1.0)
QUICK_EXPOSURES = (50,)
QUICK_EXPOSURES_BY_STAGE = {"A_exp": (30, 80)}

OVERRIDE_KEYS = ("sizes", "seeds", "exposures", "corpora")


def plan(phase: str, stage: str, mode: str, best_lr: dict | None = None, **overrides) -> list[Job]:
    """Job di uno stadio. phase "A"|"B", mode "quick"|"full"; overrides: sizes, seeds, exposures, corpora."""
    base = stage.removesuffix("_quick")
    if base not in PHASE_STAGES.get(phase, ()):
        raise ValueError(f"stadio {stage!r} non appartiene alla fase {phase!r}")
    if mode not in ("quick", "full"):
        raise ValueError(f"mode non valido: {mode!r}")
    unknown = set(overrides) - set(OVERRIDE_KEYS)
    if unknown:
        raise TypeError(f"override non validi: {sorted(unknown)}")

    spec = STAGE_SPECS[base]
    quick = mode == "quick"
    corpora = tuple(overrides.get("corpora") or spec["corpora"])
    sizes = tuple(overrides.get("sizes") or (QUICK_SIZES if quick else tuple(SIZES_V2)))
    seeds = tuple(overrides.get("seeds") or (QUICK_SEEDS if quick else spec["seeds"]))
    default_exposures = QUICK_EXPOSURES_BY_STAGE.get(base, QUICK_EXPOSURES) if quick else spec["exposures"]
    exposures = tuple(overrides.get("exposures") or default_exposures)
    lr_mults = QUICK_LR_MULTS if quick else LR_MULTS
    n_persons = QUICK_N_PERSONS if quick else N_PERSONS
    eval_persons = QUICK_EVAL_PERSONS if quick else EVAL_PERSONS

    unknown_corpora = [c for c in corpora if c not in CORPORA]
    if unknown_corpora:
        raise ValueError(f"corpora sconosciuti: {unknown_corpora}")
    unknown_sizes = [s for s in sizes if s not in SIZES_V2]
    if unknown_sizes:
        raise ValueError(f"taglie sconosciute: {unknown_sizes}")

    best = best_lr or {}
    job_stage = base + ("_quick" if quick else "")
    jobs = []
    for c in corpora:
        for size in sizes:
            for e in exposures:
                lrs = lr_mults if spec["lr"] == "all" else (_lookup_lr(best, c, size),)
                for lr in lrs:
                    for seed in seeds:
                        jobs.append(Job(job_stage, c, spec["protocol"], e, size, lr, seed,
                                        n_persons, eval_persons))
    return jobs


def _lookup_lr(best: dict, c: str, size: str) -> float:
    """lr dello stadio lr per (corpus base, taglia); fallback 1.0."""
    if c == "interlingua_v1enc":
        key = "interlingua"
    elif c.startswith("multi_"):
        key = "italiano_pieno"
    else:
        key = c.split("@", 1)[0]
    if (key, size) in best:
        return best[(key, size)]
    # taglia non provata nello sweep: lr della taglia provata più vicina in log-parametri
    provate = [s for (k, s) in best if k == key]
    if not provate:
        return 1.0
    vicina = min(provate, key=lambda s: abs(np.log(_params_nonemb(s)) - np.log(_params_nonemb(size))))
    return best[(key, vicina)]


def best_lr(df: pd.DataFrame) -> dict:
    """{(corpus, size): lr_mult} dai run dello stadio lr."""
    if df.empty:
        return {}
    top = (df.sort_values(["acc_mean", "final_loss"], ascending=[False, True])
             .drop_duplicates(["corpus", "size"]))
    return {(c, s): float(lr) for c, s, lr in zip(top["corpus"], top["size"], top["lr_mult"])}


def _key(rec) -> tuple:
    """Chiave JOB_KEYS normalizzata: uguale per Job (vars), dict e righe CSV."""
    return (str(rec["stage"]), str(rec["corpus"]), str(rec["protocol"]), int(rec["exposures"]),
            str(rec["size"]), float(rec["lr_mult"]), int(rec["seed"]), int(rec["n_persons"]))


def _read_rows(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(RESULT_COLUMNS_V2))


def _append(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row], columns=list(RESULT_COLUMNS_V2)).to_csv(
        csv_path, mode="a", header=write_header, index=False)


def _fmt_duration(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def run_stage(jobs: list[Job], csv_path: Path, reuse=()) -> pd.DataFrame:
    """Esegue i job mancanti (chiave JOB_KEYS), accoda ogni riga al CSV, stampa ETA.
    Un job gia presente in un CSV di reuse (stessa chiave tranne lo stage) viene copiato con lo stage del job."""
    csv_path = Path(csv_path)
    done = {_key(r) for r in _read_rows(csv_path)}
    donors = {}
    for path in reuse:
        for r in _read_rows(Path(path)):
            donors.setdefault(_key(r)[1:], r)

    pending = []
    for job in jobs:
        key = _key(vars(job))
        if key in done:
            continue
        done.add(key)
        donor = donors.get(key[1:])
        if donor is not None:
            _append(csv_path, {**donor, "stage": job.stage})
            print(f"[riuso] {job.run_id}")
        else:
            pending.append(job)
    if not pending:
        return _load(csv_path)

    persons = {}
    for job in pending:
        if job.n_persons not in persons:
            persons[job.n_persons] = corpus.load_persons(job.n_persons)
    max_exposures = {}
    for job in pending:
        k = (job.corpus, job.n_persons)
        max_exposures[k] = max(max_exposures.get(k, 0), job.exposures)
    for (name, n), exposures in max_exposures.items():
        train.prepare(CORPORA[name], persons[n], exposures + 1)

    total = len(pending)
    start = time.perf_counter()
    for k, job in enumerate(pending, 1):
        row = train.run_job(job, persons[job.n_persons])
        _append(csv_path, row)
        per_run = (time.perf_counter() - start) / k
        print(f"[{k}/{total}] {job.run_id} acc_mean={row['acc_mean']:.3f} "
              f"ETA {_fmt_duration(per_run * (total - k))}")
    return _load(csv_path)


def _interp_loglog(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Interpolazione lineare in log-log (estrapolazione dai due punti estremi)."""
    order = np.argsort(xs)
    lx = np.log(xs[order])
    ly = np.log(ys[order])
    if len(lx) == 1:
        return float(np.exp(ly[0]))
    lt = math.log(x)
    i = int(np.clip(np.searchsorted(lx, lt) - 1, 0, len(lx) - 2))
    slope = (ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i])
    return float(np.exp(ly[i] + slope * (lt - lx[i])))


def _curves(q: pd.DataFrame, name: str) -> dict:
    """Curve dal quick di un corpus (fallback italiano_pieno): per taglia, parametri, throughput, tempo eval/persona."""
    rows = q[q["corpus"] == name]
    if rows.empty:
        rows = q[q["corpus"] == "italiano_pieno"]
    if rows.empty:
        raise ValueError(f"nessuna riga quick per {name!r} ne per italiano_pieno")
    rows = rows.assign(eval_rate=rows["eval_time_s"] / rows["eval_persons"])
    by_size = rows.groupby("size")[["params_nonemb", "steady_tokens_per_s", "eval_rate"]].mean()
    return {
        "tokens_per_bio": float(rows["tokens_per_bio"].mean()),
        "params": by_size["params_nonemb"].to_numpy(dtype=float),
        "tps": by_size["steady_tokens_per_s"].to_numpy(dtype=float),
        "eval_rate": by_size["eval_rate"].to_numpy(dtype=float),
    }


def _params_nonemb(size: str) -> float:
    """Parametri non-embedding stimati come 12 * n_layer * n_embd^2 (come in config)."""
    s = SIZES_V2[size]
    return 12.0 * s.n_layer * s.n_embd ** 2


def estimate(jobs: list[Job], quick_csv: Path) -> float:
    """Secondi stimati per i job dai throughput a regime misurati nel quick."""
    q = pd.read_csv(quick_csv)
    curves = {}
    total = 0.0
    for job in jobs:
        if job.corpus not in curves:
            curves[job.corpus] = _curves(q, job.corpus)
        cv = curves[job.corpus]
        params = _params_nonemb(job.size)
        tps = _interp_loglog(cv["params"], cv["tps"], params)
        eval_rate = _interp_loglog(cv["params"], cv["eval_rate"], params)
        # steps_full * bios_per_step = exposures * n_persons bios, quindi tokens = exposures * n_persons * tokens_per_bio
        train_s = job.exposures * job.n_persons * cv["tokens_per_bio"] / tps
        total += train_s + eval_rate * job.eval_persons
    return total
