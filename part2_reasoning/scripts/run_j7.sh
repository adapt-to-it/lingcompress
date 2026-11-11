#!/bin/sh
# Fase J di v7: training (un adapter per processo) e valutazioni (un processo per adapter e condizione).
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/Scripts/python.exe}  # venv at the repository root; override with PY=...
export HF_HUB_OFFLINE=1
while true; do
  $PY run_v7.py train7 --models small large
  rc=$?
  [ $rc -eq 3 ] && break
  [ $rc -ne 0 ] && { echo "TRAIN FAILED rc=$rc"; exit 1; }
done
echo "TRAIN DONE"
for m in small large; do
  if [ $m = small ]; then seeds="0 1 2"; else seeds="0 1"; fi
  for c in 9 10 11 12; do
    for s in $seeds; do
      $PY run_v7.py J --models $m --cond $c --seed $s || { echo "EVAL FAILED $m c$c s$s"; exit 1; }
    done
  done
done
echo "J DONE"
