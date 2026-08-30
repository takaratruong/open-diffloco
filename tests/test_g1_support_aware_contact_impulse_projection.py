import numpy as np

from tools import analyze_g1_support_aware_contact_impulse_projection as support_projection


def test_bilateral_comparison_handles_partial_interval_mask():
    support_projected = {
        "full-a": np.ones((3, 6), dtype=np.float64),
        "full-b": np.ones((3, 6), dtype=np.float64),
    }
    support_costs = {
        "full-a": np.full(3, 6.0),
        "full-b": np.full(3, 6.0),
    }
    bilateral_arrays = {
        "projected_full_a": np.zeros((3, 6), dtype=np.float64),
        "projected_full_b": np.zeros((3, 6), dtype=np.float64),
        "discarded_residual_full_a": np.zeros((3, 6), dtype=np.float64),
        "discarded_residual_full_b": np.zeros((3, 6), dtype=np.float64),
    }
    masks = {
        "overall": np.ones(3, dtype=bool),
        "partial": np.asarray((True, False, True)),
    }

    summary, minimum_increment = support_projection._bilateral_comparison(
        support_projected,
        support_costs,
        bilateral_arrays,
        np.ones(6, dtype=np.float64),
        masks,
    )

    assert minimum_increment == 6.0
    assert summary["full-a"]["partial"]["window_count"] == 2
    assert summary["full-a"]["partial"]["unchanged_from_bilateral_count"] == 0
