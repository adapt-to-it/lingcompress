import pandas as pd
import pytest

from src.v2 import corpus, train
from src.v2.runner import best_lr, estimate, plan, run_stage
from src.v2.spec import JOB_KEYS, RESULT_COLUMNS_V2, SIZES_V2, Job


def _fake_row(job, acc=0.5, loss=1.0):
    row = dict.fromkeys(RESULT_COLUMNS_V2, 0.0)
    for k in JOB_KEYS:
        row[k] = getattr(job, k)
    row["eval_persons"] = job.eval_persons
    row["acc_mean"] = acc
    row["final_loss"] = loss
    return row


def test_a_lr_full_count():
    jobs = plan("A", "A_lr", "full")
    # src.v4.spec4 adds 0.45M and 0.52M to SIZES_V2 when imported (e.g. by tests/v4 in the same session)
    assert len([s for s in SIZES_V2 if s not in ("0.45M", "0.52M")]) == 9
    assert len(jobs) == 4 * len(SIZES_V2) * 3
    assert {j.stage for j in jobs} == {"A_lr"}
    assert all(j.n_persons == 20000 and j.eval_persons == 5000 for j in jobs)


def test_quick_stage_names_sizes_and_lr():
    jobs = plan("A", "A_lr", "quick")
    assert {j.stage for j in jobs} == {"A_lr_quick"}
    assert {j.size for j in jobs} == {"0.15M", "0.60M", "1.49M"}
    assert {j.lr_mult for j in jobs} == {0.5, 1.0}
    assert len(jobs) == 4 * 3 * 2
    assert all(j.n_persons == 2000 and j.eval_persons == 500 for j in jobs)


def test_quick_exposures_of_a_exp():
    jobs = plan("A", "A_exp", "quick")
    assert {j.exposures for j in jobs} == {30, 80}


def test_plan_rejects_stage_of_other_phase():
    with pytest.raises(ValueError):
        plan("B", "A_lr", "full")


def test_best_lr_highest_acc_then_lower_loss():
    df = pd.DataFrame([
        dict(corpus="italiano_pieno", size="0.15M", lr_mult=0.5, acc_mean=0.80, final_loss=1.2),
        dict(corpus="italiano_pieno", size="0.15M", lr_mult=1.0, acc_mean=0.90, final_loss=1.5),
        dict(corpus="italiano_pieno", size="0.15M", lr_mult=2.0, acc_mean=0.90, final_loss=1.1),
        dict(corpus="interlingua", size="0.15M", lr_mult=0.5, acc_mean=0.70, final_loss=2.0),
    ])
    assert best_lr(df) == {("italiano_pieno", "0.15M"): 2.0, ("interlingua", "0.15M"): 0.5}


def test_lr_lookup_mapping_and_fallback():
    best = {("italiano_pieno", "0.15M"): 2.0, ("interlingua", "0.15M"): 0.5}
    abl = plan("A", "A_abl", "full", best, sizes=("0.15M",))
    lr = {j.corpus: j.lr_mult for j in abl}
    assert lr == {"interlingua_v1enc": 0.5, "italiano_pieno@nodigits": 2.0}

    shared = plan("A", "A_shared", "full", best, sizes=("0.15M", "0.20M"))
    lr = {(j.corpus, j.size): j.lr_mult for j in shared}
    assert lr[("italiano_pieno@shared", "0.15M")] == 2.0
    assert lr[("interlingua@shared", "0.15M")] == 0.5
    # taglia non provata: lr della taglia provata più vicina
    assert lr[("italiano_pieno@shared", "0.20M")] == 2.0
    assert plan("A", "A_main", "full", {}, sizes=("0.20M",))[0].lr_mult == 1.0  # nessuno sweep: 1x

    multi = plan("B", "B_main", "full", best, sizes=("0.15M",))
    assert {j.lr_mult for j in multi if j.corpus == "multi_pieno"} == {2.0}
    assert {j.lr_mult for j in plan("B", "B_main", "full")} == {1.0}


def test_run_stage_skips_done_jobs(tmp_path, monkeypatch):
    calls = []
    prepared = []

    def fake_run_job(job, persons):
        calls.append(job.run_id)
        return _fake_row(job)

    monkeypatch.setattr(train, "run_job", fake_run_job)
    monkeypatch.setattr(train, "prepare", lambda spec, persons, n: prepared.append((spec.name, n)))
    monkeypatch.setattr(corpus, "load_persons", lambda n: [])

    jobs = plan("A", "A_lr", "quick", sizes=("0.15M",), corpora=("italiano_pieno",))
    assert len(jobs) == 2
    csv = tmp_path / "A_lr_quick.csv"

    run_stage(jobs[:1], csv)
    assert len(calls) == 1

    df = run_stage(jobs, csv)
    assert len(calls) == 2
    assert len(df) == 2
    assert list(pd.read_csv(csv).columns) == list(RESULT_COLUMNS_V2)
    assert prepared == [("italiano_pieno", 51), ("italiano_pieno", 51)]


def test_run_stage_reuses_lr_stage_rows(tmp_path, monkeypatch):
    calls = []

    def fake_run_job(job, persons):
        calls.append(job.run_id)
        return _fake_row(job)

    monkeypatch.setattr(train, "run_job", fake_run_job)
    monkeypatch.setattr(train, "prepare", lambda spec, persons, n: None)
    monkeypatch.setattr(corpus, "load_persons", lambda n: [])

    lr_csv = tmp_path / "A_lr_quick.csv"
    run_stage(plan("A", "A_lr", "quick", sizes=("0.15M",), corpora=("italiano_pieno",)), lr_csv)
    assert len(calls) == 2

    best = {("italiano_pieno", "0.15M"): 1.0}
    main_jobs = plan("A", "A_main", "quick", best, sizes=("0.15M",), corpora=("italiano_pieno",))
    main_csv = tmp_path / "A_main_quick.csv"
    run_stage(main_jobs, main_csv, reuse=(lr_csv,))

    assert len(calls) == 2
    rows = pd.read_csv(main_csv)
    assert len(rows) == 1
    assert rows.iloc[0]["stage"] == "A_main_quick"
    assert rows.iloc[0]["lr_mult"] == 1.0


def test_estimate_positive_and_exact_on_fake_quick(tmp_path):
    rows = []
    for size in ("0.15M", "0.60M", "1.49M"):
        s = SIZES_V2[size]
        rows.append(dict(corpus="italiano_pieno", size=size, params_nonemb=12 * s.n_layer * s.n_embd ** 2,
                         steady_tokens_per_s=2000.0, tokens_per_bio=60, bios_per_step=512,
                         eval_time_s=10.0, eval_persons=500))
    quick = tmp_path / "A_lr_quick.csv"
    pd.DataFrame(rows).to_csv(quick, index=False)

    job = Job("A_lr", "italiano_pieno", "passi", 200, "0.15M", 1.0, 0, 20000, 5000)
    # 200 * 20000 * 60 / 2000 = 120000 s di training, 10/500 * 5000 = 100 s di eval
    assert estimate([job], quick) == pytest.approx(120100)

    esp = plan("A", "A_esp", "full", sizes=("0.20M", "4M"))  # telegrafico assente: fallback italiano_pieno
    assert estimate(esp, quick) > 0
