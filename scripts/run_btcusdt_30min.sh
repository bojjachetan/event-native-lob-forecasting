#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/fetch_and_run_tardis_trial.py \
  --date 2020-02-01 \
  --trial-minutes 30 \
  --train-minutes 10 \
  --test-minutes 2 \
  --embargo-minutes 5 \
  --step-minutes 2 \
  --min-train-events 2000 \
  --min-test-events 500 \
  --ctgnn-epochs 5 \
  --baseline-epochs 4 \
  --batch-size 32 \
  --chunk-size 32 \
  --seeds 42,43,44 \
  --device cpu \
  --no-train-on-spine \
  --supervision-mode all_events
