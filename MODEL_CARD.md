# Model Card

## Primary Model

`CTGNN` combines:
- asynchronous `TGNMemory` state updates
- explicit market-structure embeddings
- factorized marked next-event modeling
- full-book masked attention over all visible nodes plus the execution sink
- downstream forecasting heads for forward realized volatility

## Forecast Timing

The model scores marked next-event losses from the pre-update memory state and downstream volatility from the post-event state at the same event timestamp. This keeps the event model causal while making the downstream forecast condition on information available immediately after the observed event.

## Baselines

Two discrete-time baselines are included:
- `DeepLOBBaseline` on 100ms representative snapshots
- `StaticGCNBaseline` on 1-second representative snapshots

Both baselines:
- inherit targets from exact representative event timestamps
- train and test only within purged fold boundaries
- use purged inner validation for model selection

## Metrics

Primary downstream metrics:
- RMSE
- MAE
- Spearman rank correlation

Primary event-modeling metrics:
- event NLL
- event-type accuracy
- location accuracy
