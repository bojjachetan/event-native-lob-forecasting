import pytest

from src.train_simple_baselines import assert_ridge_feature_columns_are_safe


def test_ridge_feature_guard_rejects_target_and_future_columns():
    with pytest.raises(AssertionError):
        assert_ridge_feature_columns_are_safe(["size", "rv_1s"])
    with pytest.raises(AssertionError):
        assert_ridge_feature_columns_are_safe(["spread_bps", "future_mid"])
    with pytest.raises(AssertionError):
        assert_ridge_feature_columns_are_safe(["book_imbalance_l1", "target_end_t_us"])


def test_ridge_feature_guard_allows_current_lob_features():
    assert_ridge_feature_columns_are_safe(
        [
            "signed_event_size",
            "spread_bps",
            "same_level_imbalance",
            "bid_px_1",
            "bid_sz_1",
            "ask_px_1",
            "ask_sz_1",
        ]
    )
