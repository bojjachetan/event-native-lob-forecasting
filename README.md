# Continuous-Time LOB Modeling

This repository contains the research pipeline I used for event-native limit order book forecasting. The main goal is to compare a continuous-time graph memory model with discrete-time neural baselines, while keeping the evaluation strictly chronological and leakage-aware.

The basic idea is straightforward. Instead of first squeezing the order book into fixed-time snapshots, the model reads actual book events in the order they happened, updates graph memory asynchronously, and forecasts forward realized volatility from the reconstructed book state.

## What Is Included

- Top-10 limit order book reconstruction from exchange depth and trade streams
- Event-level features and forward realized-volatility targets built from reconstructed mid-prices
- Purged walk-forward splits with an embargo period
- CT-GNN model with TGN-style memory and marked next-event losses
- DeepLOB and StaticGCN baselines aligned to representative continuous-event timestamps
- Persistence, rolling mean, and Ridge baselines, with leakage checks
- Audits for event integrity, target construction, split correctness, and baseline alignment
- Tests for split validity, target construction, memory ordering, deterministic behavior, tensor shapes, and Ridge feature safety

## Repository Layout

- `src/`: reconstruction, target construction, training, evaluation, and audit code
- `scripts/`: entry points for the BTCUSDT experiment
- `configs/`: default experiment configuration
- `tests/`: protocol and shape checks
- `report/`: final report PDF
- `figures/`: figures used in the report
- `results/`: CSV tables used in the report
- `artifacts/`: audit report, Ridge audit summary, and run manifest

## Main Result

The included BTCUSDT experiment uses:

- Symbol: `BTCUSDT`
- Date: `2020-02-01`
- Window: 30 minutes
- Splits: 8 purged walk-forward folds
- Embargo: 5 minutes
- Seeds: `42,43,44`
- Backend: CPU
- Supervision: all events

The main result is intentionally narrow. CT-GNN improves the rank-ordering of future realized volatility compared with the neural discretized baselines. At the same time, Ridge remains a strong clean linear baseline, and CT-GNN does not dominate absolute-error calibration.

## Setup

```bash
pip install -r requirements.txt
```

`torch-geometric` is needed for CT-GNN and StaticGCN.

## Data

The default raw-data layout is:

```text
data/raw/binance/BTCUSDT_snapshot.json.gz
data/raw/binance/BTCUSDT_depth_buffer.jsonl.gz
data/raw/binance/BTCUSDT_depth.jsonl.gz
data/raw/binance/BTCUSDT_aggtrades.jsonl.gz
```

The trial script can download public first-of-month Tardis CSV files and convert them into this schema.

## Run

To run the 30-minute BTCUSDT experiment:

```bash
scripts/run_btcusdt_30min.sh
```

To run a smaller mechanics check:

```bash
python scripts/fetch_and_run_tardis_trial.py \
  --date 2020-02-01 \
  --trial-minutes 1.5 \
  --train-minutes 0.5 \
  --test-minutes 0.25 \
  --embargo-minutes 0.25 \
  --step-minutes 0.25 \
  --min-train-events 100 \
  --min-test-events 50 \
  --ctgnn-epochs 0 \
  --baseline-epochs 1 \
  --batch-size 16 \
  --chunk-size 16 \
  --seeds 42 \
  --device cpu
```

Outputs are written under `trial_runs/`, which is intentionally ignored by Git.

## Outputs

The committed result files are:

```text
results/paper_main_table.csv
results/paper_rank_table.csv
results/paper_error_table.csv
artifacts/audit_report.json
artifacts/ridge_audit_summary.json
artifacts/run_manifest.json
```

A fresh run writes:

```text
data/events.parquet
data/targets.parquet
data/split_manifest.parquet
outputs/eval/ctgnn/summary_table.csv
outputs/baselines/deeplob/baseline_metrics_summary.json
outputs/baselines/static_gcn/baseline_metrics_summary.json
outputs/baselines/simple/ridge_audit_summary.json
outputs/aligned/summary_table.csv
outputs/audit/audit_report.json
run_manifest.json
```

## Integrity Constraints

The pipeline is built around a few constraints that should not be relaxed:

- Real reconstructed mid-prices are used for realized-volatility targets.
- No simulated depth is used.
- No proxy volatility labels are used.
- Train/test splits are chronological and purged.
- Discrete baselines are evaluated only at timestamps aligned with continuous events.
- CT-GNN losses are computed before memory is updated with the supervised event.

## Tests

```bash
pytest -q
```

The tests are small by design. They are meant to catch protocol mistakes, not to prove model performance.
