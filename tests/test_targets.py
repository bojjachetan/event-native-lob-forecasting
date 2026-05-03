import inspect

import numpy as np

from src.data.build_targets import _forward_realized_variance
import src.data.build_targets as build_targets


def test_forward_rv_uses_real_mid_path():
    t_us = np.array([0, 500_000, 1_000_000, 1_500_000], dtype=np.int64)
    mid = np.array([100.0, 101.0, 100.5, 101.5], dtype=np.float64)
    rv = _forward_realized_variance(t_us=t_us, mid=mid, horizons_s=(1,))
    dlog = np.diff(np.log(mid))
    expected0 = np.sqrt(dlog[0] ** 2 + dlog[1] ** 2)
    assert np.isclose(rv["rv_1s"][0], expected0)


def test_no_proxy_walk_in_target_builder_source():
    source = inspect.getsource(build_targets)
    lowered = source.lower()
    assert "random walk" not in lowered
    assert "proxy" not in lowered
    assert "mid" in lowered
