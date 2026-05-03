import numpy as np
import torch

from src.make_splits import build_split_manifest
from src.train import StandardTargetScaler, set_seed


def test_same_seed_repeats_first_random_value_cpu():
    set_seed(42)
    a = torch.randn(1).item()
    set_seed(42)
    b = torch.randn(1).item()
    assert a == b


def test_same_seed_produces_same_split_manifest():
    t_us = np.arange(0, 1_000_000 * 100, 1_000_000, dtype=np.int64)
    a = build_split_manifest(t_us, 20_000_000, 5_000_000, embargo_us=5_000_000, step_us=5_000_000, min_train_events=1, min_test_events=1)
    b = build_split_manifest(t_us, 20_000_000, 5_000_000, embargo_us=5_000_000, step_us=5_000_000, min_train_events=1, min_test_events=1)
    assert a.equals(b)


def test_target_scaler_inverse_and_train_scope():
    y_train = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    scaler = StandardTargetScaler.fit(y_train)
    z = scaler.transform(y_train)
    restored = scaler.inverse_transform(z)
    assert torch.allclose(restored, y_train)
    assert scaler.to_dict()["fit_scope"] == "train_fold_only"
