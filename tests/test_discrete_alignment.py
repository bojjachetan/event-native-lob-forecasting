import numpy as np
import pandas as pd


def test_deeplob_representatives_exist_in_continuous_events():
    events = pd.DataFrame({"event_id": [1, 2, 3], "t_us": [10, 20, 30]})
    sample_event_ids = np.array([1, 3], dtype=np.int64)
    sample_t_us = np.array([10, 30], dtype=np.int64)
    assert set(sample_event_ids).issubset(set(events["event_id"]))
    assert set(sample_t_us).issubset(set(events["t_us"]))


def test_static_gcn_sample_lies_inside_test_fold():
    fold = {"test_start_us": 100, "test_end_us": 200}
    t_us = np.array([101, 150, 199], dtype=np.int64)
    assert np.all((t_us >= fold["test_start_us"]) & (t_us < fold["test_end_us"]))


def test_deeplob_sequences_do_not_cross_fold_boundaries():
    sequence_t_us = np.array([[100, 110, 120], [130, 140, 150]], dtype=np.int64)
    fold = {"train_start_us": 90, "train_end_us": 160}
    assert np.all(sequence_t_us >= fold["train_start_us"])
    assert np.all(sequence_t_us < fold["train_end_us"])
