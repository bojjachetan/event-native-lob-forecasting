# Dataset Card

## Source

Real exchange market data:
- Binance depth snapshot for bootstrap
- Binance diff-depth stream for book updates
- Binance aggTrade stream for aggressive-trade alignment

## Unit Of Observation

The core dataset is event-native. Each row corresponds to one real visible top-10 LOB event after reconstruction.

Columns include:
- event timestamp in microseconds
- event type: add, cancel, execute
- side and visible level
- event price and event size
- post-event visible top-10 book state
- derived event-native microstructure features

## Target Definition

Forward realized volatility is computed from the true reconstructed mid-price path, not from proxy labels.

For each event timestamp `t_i` and horizon `H`:
- `rv_H_var(i)` is the sum of squared log-mid returns between `i+1` and the last event inside `t_i + H`
- `rv_H(i)` is the square root of that quantity

Default horizons:
- 1 second
- 5 seconds
- 10 seconds

## Temporal Protocol

- chronological stable sorting by exchange timestamp
- purged walk-forward splits with embargo
- no random train/test mixing
- discrete-time baselines inherit labels from the exact representative continuous event

## Known Scope

This repository is designed around visible top-10 reconstruction and the default config targets BTCUSDT, but the pipeline is generic to any symbol for which the same raw exchange artifacts are available.
