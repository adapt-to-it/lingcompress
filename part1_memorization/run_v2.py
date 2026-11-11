"""Esperimento v2: esegue uno stadio o una fase, stima i tempi, genera il report."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.v2.runner import PHASE_STAGES, best_lr, estimate, plan, run_stage
from src.v2.spec import V2_RESULTS


def _csv(stage: str, quick: bool) -> Path:
    return V2_RESULTS / f"{stage}{'_quick' if quick else ''}.csv"


def _best(quick: bool) -> dict:
    path = _csv("A_lr", quick)
    return best_lr(pd.read_csv(path)) if path.exists() else {}


def _split(text: str) -> tuple:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _overrides(args: argparse.Namespace) -> dict:
    ov = {}
    if args.sizes:
        ov["sizes"] = _split(args.sizes)
    if args.seeds:
        ov["seeds"] = tuple(int(x) for x in _split(args.seeds))
    if args.exposures:
        ov["exposures"] = tuple(int(x) for x in _split(args.exposures))
    if args.corpora:
        ov["corpora"] = _split(args.corpora)
    return ov


def _run(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    if args.phase is None:
        ap.error("--phase richiesto")
    if args.stage != "all" and args.stage not in PHASE_STAGES[args.phase]:
        ap.error(f"stadio {args.stage!r} non nella fase {args.phase}")
    quick = args.quick
    overrides = _overrides(args)
    stages = PHASE_STAGES[args.phase] if args.stage == "all" else (args.stage,)
    for stage in stages:
        best = _best(quick)
        if not best and stage != "A_lr":
            print("ATTENZIONE: A_lr assente, lr di default 1.0")
        jobs = plan(args.phase, stage, "quick" if quick else "full", best, **overrides)
        reuse = (_csv("A_lr", quick),) if stage == "A_main" else ()
        run_stage(jobs, _csv(stage, quick), reuse=reuse)


def _estimate(overrides: dict) -> None:
    best = _best(False)
    quick_csvs = sorted(V2_RESULTS.glob("*_quick.csv"))
    if not quick_csvs:
        raise SystemExit("nessun CSV *_quick: eseguire prima --quick")
    total = 0.0
    for phase, stages in PHASE_STAGES.items():
        for stage in stages:
            own = _csv(stage, True)
            quick_csv = own if own.exists() else quick_csvs[0]
            jobs = plan(phase, stage, "full", best, **overrides)
            hours = estimate(jobs, quick_csv) / 3600
            total += hours
            print(f"{stage:<10} {len(jobs):>5} job  {hours:9.2f} h  (quick: {quick_csv.name})")
    print(f"totale: {total:.2f} h")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Esperimento v2 LingCompress.")
    ap.add_argument("--phase", choices=sorted(PHASE_STAGES), help="fase A o B")
    ap.add_argument("--stage", default="all", help="nome dello stadio, oppure 'all'")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="quick", action="store_true")
    mode.add_argument("--full", dest="quick", action="store_false")
    ap.set_defaults(quick=False)
    ap.add_argument("--sizes", help="taglie separate da virgola")
    ap.add_argument("--seeds", help="seed separati da virgola")
    ap.add_argument("--exposures", help="esposizioni separate da virgola")
    ap.add_argument("--corpora", help="corpora separati da virgola")
    ap.add_argument("--estimate", action="store_true", help="stima ore per stadio dal quick")
    ap.add_argument("--report", action="store_true", help="genera il report da V2_RESULTS")
    args = ap.parse_args(argv)

    if args.report:
        from src.v2.analysis import write_all
        write_all(V2_RESULTS)
    elif args.estimate:
        _estimate(_overrides(args))
    else:
        _run(args, ap)


if __name__ == "__main__":
    main()
