import numpy as np

from src.make_splits import generate_purged_walk_forward_splits


def test_purged_walk_forward_has_five_minute_embargo():
    t_us = np.arange(0, 30 * 60 * 1_000_000, 1_000_000, dtype=np.int64)
    folds = list(
        generate_purged_walk_forward_splits(
            t_us=t_us,
            train_window_us=10 * 60 * 1_000_000,
            test_window_us=2 * 60 * 1_000_000,
            embargo_us=5 * 60 * 1_000_000,
            step_us=2 * 60 * 1_000_000,
            min_train_events=1,
            min_test_events=1,
        )
    )
    assert folds
    for fold in folds:
        assert fold.train_start_us < fold.train_end_us <= fold.embargo_start_us
        assert fold.embargo_start_us < fold.embargo_end_us <= fold.test_start_us
        assert fold.train_end_us + 5 * 60 * 1_000_000 <= fold.test_start_us
        assert len(np.intersect1d(fold.train_indices, fold.test_indices)) == 0
