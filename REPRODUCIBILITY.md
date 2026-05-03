# Reproducibility

The main experiment is run through:

```bash
scripts/run_btcusdt_30min.sh
```

The script downloads/adapts the public Tardis BTCUSDT data for `2020-02-01`, reconstructs the top-10 book, builds event targets, creates purged walk-forward splits, trains CT-GNN and baselines, audits the run, and writes aligned result tables.

## Backend

Use CPU for this workload:

```bash
--device cpu
```

On the local Apple Silicon machine used for this project, CPU was faster than MPS for the current PyTorch Geometric memory-update pattern.

## Protocol

The main run uses:

- `--trial-minutes 30`
- `--train-minutes 10`
- `--test-minutes 2`
- `--embargo-minutes 5`
- `--step-minutes 2`
- `--seeds 42,43,44`
- `--device cpu`
- `--no-train-on-spine`
- `--supervision-mode all_events`

The generated `run_manifest.json` records command-line arguments, environment information, output paths, fold counts, seeds, and runtime metadata.

## Checks

Run the test suite with:

```bash
pytest -q
```

For each experiment run, check:

- `outputs/audit/audit_report.json`
- `outputs/baselines/simple/ridge_audit_summary.json`
- `outputs/aligned/summary_table.csv`
- `run_manifest.json`
