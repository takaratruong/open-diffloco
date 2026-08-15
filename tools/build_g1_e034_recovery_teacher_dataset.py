"""Build immutable state/action supervision from the E034 recovery oracle."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


E034_SURVIVAL = (
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    30,
    32,
    26,
    21,
    20,
    18,
    16,
    15,
    14,
    12,
    11,
    9,
)

_SHAPES = {
    "actor_obs": (24, 32, 3280),
    "phase": (24, 32),
    "parent_action": (24, 32, 29),
    "correction": (24, 32, 29),
    "raw_action": (24, 32, 29),
    "effective_action": (24, 32, 29),
    "alive": (24, 32),
    "terminal": (24, 32),
    "reward": (24, 32),
    "normalized_termination_errors": (24, 32, 4),
}


def _survival(terminals: np.ndarray) -> list[int]:
    output = []
    for row in terminals:
        indices = np.flatnonzero(row)
        output.append(int(indices[0]) if indices.size else 32)
    return output


def validate_teacher_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Fail closed unless arrays exactly reproduce the E034 replay contract."""
    for name, shape in _SHAPES.items():
        if name not in arrays:
            raise ValueError(f"teacher dataset is missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"teacher dataset {name} shape does not match")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"teacher dataset {name} must be finite")

    terminal = np.asarray(arrays["terminal"], dtype=bool)
    alive = np.asarray(arrays["alive"], dtype=bool)
    survival = _survival(terminal)
    if survival != list(E034_SURVIVAL):
        raise ValueError("teacher dataset survival does not reproduce E034")
    expected_alive = np.ones((24, 32), dtype=bool)
    for row, survived in enumerate(E034_SURVIVAL):
        if survived < 32:
            expected_alive[row, survived + 1 :] = False
    if not np.array_equal(alive, expected_alive):
        raise ValueError("teacher dataset alive mask does not match terminals")

    raw = np.asarray(arrays["raw_action"])
    effective = np.asarray(arrays["effective_action"])
    if not np.array_equal(effective, np.clip(raw, -1.0, 1.0)):
        raise ValueError("teacher effective actions do not match final clipping")
    if not np.allclose(
        raw,
        np.asarray(arrays["parent_action"])
        + np.asarray(arrays["correction"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("teacher raw action does not equal parent plus correction")

    successful = np.asarray(E034_SURVIVAL) == 32
    action_mask = np.broadcast_to(alive[..., None], raw.shape)
    clipped = np.abs(raw) > 1.0

    def clip_fraction(start_mask: np.ndarray) -> float:
        selected = action_mask & start_mask[:, None, None]
        return float(np.mean(clipped[selected]))

    return {
        "survival": survival,
        "successful_starts": int(np.sum(successful)),
        "teacher_rows": int(np.sum(successful) * 32),
        "all_clip_fraction": clip_fraction(np.ones(24, dtype=bool)),
        "recovered_clip_fraction": clip_fraction(successful),
        "failed_clip_fraction": clip_fraction(~successful),
    }
