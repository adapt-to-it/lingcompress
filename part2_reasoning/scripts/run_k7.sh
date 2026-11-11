#!/bin/sh
# Fase K di v7: Qwen3-4B-Base QLoRA, adapter C5 e A1 (un adapter per processo), poi condizioni 5 e 10.
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/Scripts/python.exe}  # venv at the repository root; override with PY=...
export HF_HUB_OFFLINE=1
FMT=${1:-A1}
while true; do
  $PY run_v7.py xl_train --fmt $FMT
  rc=$?
  [ $rc -eq 3 ] && break
  [ $rc -ne 0 ] && { echo "TRAIN FAILED rc=$rc"; exit 1; }
done
echo "TRAIN DONE"
if [ $FMT = A1 ]; then C=10; else C=12; fi
for c in 5 $C; do
  $PY run_v7.py J --models xl --cond $c --seed 0 || { echo "EVAL FAILED xl c$c"; exit 1; }
done
echo "K DONE"
