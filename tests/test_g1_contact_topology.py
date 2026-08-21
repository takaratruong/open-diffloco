import jax.numpy as jp
import numpy as np
import pytest

from src.envs.g1_tracking.contact_topology import (
    contact_topology_event,
    grouped_foot_support,
)


def test_grouped_support_uses_all_geometries_and_ignores_inactive_rows() -> None:
    geom_bodyid = jp.asarray([0, 7, 7, 13, 18])
    contact_geom = jp.asarray([[1, 0], [2, 4], [3, 0], [4, 0]])
    efc_address = jp.asarray([0, -1, 3, 6])

    actual = grouped_foot_support(
        contact_geom,
        efc_address,
        geom_bodyid,
        jp.asarray([7, 13]),
    )

    np.testing.assert_array_equal(actual, np.asarray([True, True]))


def test_grouped_support_ignores_active_nonfoot_contacts() -> None:
    actual = grouped_foot_support(
        jp.asarray([[0, 3], [3, 4]]),
        jp.asarray([0, 3]),
        jp.asarray([0, 7, 13, 18, 19]),
        jp.asarray([7, 13]),
    )

    np.testing.assert_array_equal(actual, np.asarray([False, False]))


@pytest.mark.parametrize(
    ("previous", "current"),
    (
        ([False, True], [True, True]),
        ([True, True], [False, True]),
        ([True, False], [False, True]),
    ),
)
def test_topology_event_detects_touchdown_and_liftoff(
    previous: list[bool], current: list[bool]
) -> None:
    assert bool(
        contact_topology_event(
            jp.asarray(previous), jp.asarray(current), done=False
        )
    )


def test_topology_event_ignores_unchanged_support_and_reset() -> None:
    support = jp.asarray([False, True])
    double = jp.asarray([True, True])

    assert not bool(contact_topology_event(support, support, done=False))
    assert not bool(contact_topology_event(support, double, done=True))


@pytest.mark.parametrize(
    ("contact_geom", "efc_address", "foot_body_ids"),
    (
        (jp.zeros((2, 3), dtype=jp.int32), jp.zeros(2), jp.asarray([7, 13])),
        (jp.zeros((2, 2), dtype=jp.int32), jp.zeros(3), jp.asarray([7, 13])),
        (jp.zeros((2, 2), dtype=jp.int32), jp.zeros(2), jp.asarray([7])),
    ),
)
def test_grouped_support_rejects_incompatible_shapes(
    contact_geom, efc_address, foot_body_ids
) -> None:
    with pytest.raises(ValueError, match="contact topology inputs"):
        grouped_foot_support(
            contact_geom,
            efc_address,
            jp.asarray([0, 7, 13]),
            foot_body_ids,
        )


def test_topology_event_rejects_non_pair_signatures() -> None:
    with pytest.raises(ValueError, match="shape.*2"):
        contact_topology_event(
            jp.asarray([True]), jp.asarray([False]), done=False
        )
