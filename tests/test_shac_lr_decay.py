import pytest


def test_lr_decay_updates_defaults_to_full_declared_budget():
    from src.algorithms.shac.algorithm import resolve_lr_decay_updates

    assert resolve_lr_decay_updates(
        total_steps=393_216,
        steps_per_actor_update=3_072,
        requested_updates=None,
    ) == 128


def test_lr_decay_updates_can_be_pinned_independently_of_run_budget():
    from src.algorithms.shac.algorithm import resolve_lr_decay_updates

    assert resolve_lr_decay_updates(
        total_steps=393_216,
        steps_per_actor_update=3_072,
        requested_updates=16,
    ) == 16


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_lr_decay_updates_rejects_invalid_overrides(value):
    from src.algorithms.shac.algorithm import resolve_lr_decay_updates

    with pytest.raises(ValueError, match="positive integer"):
        resolve_lr_decay_updates(
            total_steps=393_216,
            steps_per_actor_update=3_072,
            requested_updates=value,
        )
