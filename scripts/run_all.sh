#!/usr/bin/env bash
# Reproduce every number in the README (~2-3 h on an Apple M4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# Flush prints straight to the log files so progress is visible.
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs

$PY -m racism_detector.data
$PY -m racism_detector.baseline > outputs/logs/baseline.log 2>&1

# Single-task (aux weight 0) vs multi-task (aux weight 1), 3 seeds each.
# Seed 1 checkpoints are kept for racism_detector.predict.
for seed in 1 2 3; do
  for w in 0 1; do
    save=""
    if [[ $seed == 1 ]]; then save="--save-model"; fi
    $PY -m racism_detector.train --aux-weight "$w" --seed "$seed" $save \
      > "outputs/logs/aux${w}_seed${seed}.log" 2>&1
  done
done

$PY -m racism_detector.report
