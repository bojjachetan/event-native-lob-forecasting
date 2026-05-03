import pytest
import torch

from src.models.baselines import DeepLOBBaseline, StaticGCNBaseline
from src.train import TrainConfig, build_model


def test_ctgnn_output_volatility_shape():
    pytest.importorskip("torch_geometric")
    cfg = TrainConfig(events_path="x", targets_path="y", memory_dim=16, time_dim=8, structure_embed_dim=8, structure_dim=16, raw_msg_dim=16, msg_hidden_dim=16, marked_hidden_dim=16, readout_dim=32, readout_heads=4)
    model = build_model(cfg)
    out = model(node_populated_mask=torch.ones(2, 21, dtype=torch.bool), compute_marked=False, enable_price_move_head=False)
    assert out["volatility"].shape == (2, 3)


def test_deeplob_output_shape():
    model = DeepLOBBaseline(input_dim=40, conv_channels=8, inception_branch_channels=4, lstm_hidden_dim=8, regression_hidden_dim=16)
    y = model(torch.randn(2, 20, 40))
    assert y.shape == (2, 3)


def test_static_gcn_output_shape():
    pytest.importorskip("torch_geometric")
    model = StaticGCNBaseline(node_feat_dim=5, num_levels=10, hidden_dim=8, num_gcn_layers=1, regression_hidden_dim=16)
    y = model(torch.randn(2, 20, 5))
    assert y.shape == (2, 3)
