import inspect

import src.train as train


def test_supervised_loss_order_is_pre_update():
    source = inspect.getsource(train.compute_event_losses)
    assert "update_memory" not in source
    loop_source = inspect.getsource(train.train_one_epoch)
    supervised_loss_pos = loop_source.find("losses = compute_event_losses")
    post_loss_replay_pos = loop_source.find("replay_event_no_grad", supervised_loss_pos)
    assert supervised_loss_pos >= 0
    assert post_loss_replay_pos > supervised_loss_pos


def test_non_supervised_replay_does_not_compute_loss():
    source = inspect.getsource(train.train_one_epoch)
    unsup_branch = source[source.find("if not is_supervised") : source.find("losses = compute_event_losses")]
    assert "replay_event_no_grad" in unsup_branch
    assert "compute_event_losses" not in unsup_branch
