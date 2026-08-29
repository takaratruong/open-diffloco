"""Analyze preterminal windows in retained and learned-wrench G1 rollouts.

This is an artifact-only analysis: it reads immutable ``evaluation.npz`` files
and never constructs or steps a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROTOCOL_VERSION = "g1-learned-wrench-failure-window-v1"
RUN_LABELS = (
    "e002",
    "full-a",
    "full-b",
    "vertical-a",
    "vertical-b",
    "novertical-a",
    "novertical-b",
)
CONTROL_LABELS = ("full-a", "full-b")
TREATMENT_LABELS = (
    "vertical-a",
    "vertical-b",
    "novertical-a",
    "novertical-b",
)
FAMILIES = {
    "vertical-only": ("vertical-a", "vertical-b"),
    "no-vertical": ("novertical-a", "novertical-b"),
}
DISPLAY_LABELS = {
    "e002": "E002 unassisted",
    "full-a": "full wrench A",
    "full-b": "full wrench B",
    "vertical-a": "vertical only A",
    "vertical-b": "vertical only B",
    "novertical-a": "no vertical A",
    "novertical-b": "no vertical B",
}
EXPECTED_MASKS = {
    "full-a": np.ones(6, dtype=np.float64),
    "full-b": np.ones(6, dtype=np.float64),
    "vertical-a": np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float64),
    "vertical-b": np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float64),
    "novertical-a": np.asarray([1, 1, 0, 1, 1, 1], dtype=np.float64),
    "novertical-b": np.asarray([1, 1, 0, 1, 1, 1], dtype=np.float64),
}
TERMINATION_CHANNELS = (
    "anchor_z_error",
    "anchor_xy_error",
    "gravity_z_error",
    "distal_z_error",
)
TERMINATION_THRESHOLDS = np.asarray([0.25, 1.3, 0.8, 0.4])
WRENCH_COMPONENTS = ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")
ONSET_FRACTION = 0.5
PRE_ONSET_TRANSITIONS = 20
DUPLICATE_ONSET_TOLERANCE = 8
SHARED_ONSET_TOLERANCE = 12
SUPPORT_PROXIMITY_TOLERANCE = 5


@dataclass(frozen=True)
class RunData:
    """Validated arrays for one immutable rollout."""

    label: str
    path: Path
    sha256: str
    arrays: dict[str, np.ndarray]
    column_index: dict[str, int]

    @property
    def rows(self) -> int:
        return int(self.arrays["values"].shape[0])


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def load_run(
    *,
    label: str,
    path: Path,
    expected_sha256: str,
    expected_reference_transitions: int,
) -> RunData:
    """Load and validate one hash-bound evaluation archive."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label} archive: {resolved}")
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    common = {
        "columns",
        "values",
        "actor_joint_names",
        "joint_position",
        "joint_velocity",
        "reference_joint_position",
        "position_target",
        "qpos",
        "qvel",
        "constraint_force_yaw",
        "foot_support",
        "centroidal_momentum",
        "reference_centroidal_momentum",
    }
    missing = sorted(common - set(arrays))
    if missing:
        raise ValueError(f"{label} is missing arrays: {missing}")

    columns = arrays["columns"].tolist()
    required_columns = {
        "step",
        "phase",
        "done",
        "terminal",
        "transition_phase",
        *(f"termination_{name}" for name in TERMINATION_CHANNELS),
    }
    if not required_columns.issubset(columns):
        raise ValueError(f"{label} has an incompatible values schema")
    column_index = {name: columns.index(name) for name in required_columns}
    values = np.asarray(arrays["values"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] < 1:
        raise ValueError(f"{label} values must be a nonempty row-aligned matrix")
    rows = values.shape[0]
    _require_finite(f"{label} values", values)

    expected_vector_shapes = {
        "joint_position": (rows, 29),
        "joint_velocity": (rows, 29),
        "reference_joint_position": (rows, 29),
        "position_target": (rows, 29),
        "qpos": (rows, 36),
        "qvel": (rows, 35),
        "constraint_force_yaw": (rows, 3),
        "foot_support": (rows, 2),
        "centroidal_momentum": (rows + 1, 6),
        "reference_centroidal_momentum": (rows + 1, 6),
    }
    for name, shape in expected_vector_shapes.items():
        array = np.asarray(arrays[name])
        if array.shape != shape:
            raise ValueError(f"{label} {name} has shape {array.shape}, expected {shape}")
        _require_finite(f"{label} {name}", array)
    if arrays["actor_joint_names"].shape != (29,):
        raise ValueError(f"{label} must contain 29 actor joint names")

    expected_rows = np.arange(rows, dtype=np.float64)
    np.testing.assert_array_equal(values[:, column_index["step"]], expected_rows)
    np.testing.assert_array_equal(values[:, column_index["phase"]], expected_rows)
    np.testing.assert_array_equal(
        values[:, column_index["transition_phase"]], expected_rows + 1.0
    )
    if np.any(values[:-1, column_index["done"]] > 0.5):
        raise ValueError(f"{label} has an intermediate reset")
    if values[-1, column_index["done"]] <= 0.5:
        raise ValueError(f"{label} does not end at a done boundary")

    terminal = values[:, column_index["terminal"]] > 0.5
    if label in CONTROL_LABELS:
        if rows != expected_reference_transitions or terminal.any():
            raise ValueError(f"{label} is not a successful full-reference control")
    else:
        if rows >= expected_reference_transitions or not terminal[-1] or terminal[:-1].any():
            raise ValueError(f"{label} is not a single pre-reference terminal rollout")

    if label != "e002":
        wrench_arrays = {
            "learned_torso_wrench",
            "learned_torso_wrench_unmasked",
            "learned_torso_wrench_component_mask",
        }
        missing_wrench = sorted(wrench_arrays - set(arrays))
        if missing_wrench:
            raise ValueError(f"{label} is missing wrench arrays: {missing_wrench}")
        applied = np.asarray(arrays["learned_torso_wrench"], dtype=np.float64)
        unmasked = np.asarray(
            arrays["learned_torso_wrench_unmasked"], dtype=np.float64
        )
        mask = np.asarray(
            arrays["learned_torso_wrench_component_mask"], dtype=np.float64
        )
        if applied.shape != (rows, 6) or unmasked.shape != (rows, 6):
            raise ValueError(f"{label} wrench telemetry must have shape ({rows}, 6)")
        _require_finite(f"{label} applied wrench", applied)
        _require_finite(f"{label} unmasked wrench", unmasked)
        np.testing.assert_array_equal(mask, EXPECTED_MASKS[label])
        np.testing.assert_array_equal(applied, unmasked * mask)

    return RunData(
        label=label,
        path=resolved,
        sha256=actual_sha256,
        arrays=arrays,
        column_index=column_index,
    )


def normalized_termination_error(run: RunData) -> np.ndarray:
    """Return each termination channel as a fraction of its hard threshold."""
    indices = [
        run.column_index[f"termination_{name}"] for name in TERMINATION_CHANNELS
    ]
    return run.arrays["values"][:, indices] / TERMINATION_THRESHOLDS


def sustained_onset(max_ratio: np.ndarray, fraction: float) -> int | None:
    """Find the first row at or above ``fraction`` that never recovers below it."""
    for row in np.flatnonzero(max_ratio >= fraction):
        if np.all(max_ratio[row:] >= fraction):
            return int(row)
    return None


def support_transition_rows(run: RunData) -> np.ndarray:
    """Return rows where the two-bit support topology changes."""
    support = run.arrays["foot_support"].astype(np.int8)
    code = support[:, 0] + 2 * support[:, 1]
    return np.flatnonzero(np.r_[False, code[1:] != code[:-1]])


def _json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def analyze_run(run: RunData, *, control_dt: float) -> dict[str, object]:
    """Measure the fixed preterminal window and its physical telemetry."""
    normalized = normalized_termination_error(run)
    max_ratio = np.max(normalized, axis=1)
    onset_rows = {
        "25pct": sustained_onset(max_ratio, 0.25),
        "50pct": sustained_onset(max_ratio, ONSET_FRACTION),
        "75pct": sustained_onset(max_ratio, 0.75),
    }
    onset_row = onset_rows["50pct"]
    transition_values = run.arrays["values"][
        :, run.column_index["transition_phase"]
    ].astype(np.int64)
    support_rows = support_transition_rows(run)
    support_transitions = transition_values[support_rows]
    terminal_ratios = normalized[-1]
    dominant_index = int(np.argmax(terminal_ratios))
    momentum_error = (
        run.arrays["centroidal_momentum"][1:]
        - run.arrays["reference_centroidal_momentum"][1:]
    )

    result: dict[str, object] = {
        "path": str(run.path),
        "sha256": run.sha256,
        "steps": run.rows,
        "terminal": bool(
            run.arrays["values"][-1, run.column_index["terminal"]] > 0.5
        ),
        "terminal_threshold_ratios": {
            name: float(terminal_ratios[index])
            for index, name in enumerate(TERMINATION_CHANNELS)
        },
        "dominant_terminal_channel": TERMINATION_CHANNELS[dominant_index],
        "support_transition_phases": [int(value) for value in support_transitions],
        "tail20_centroidal_abs_error_mean": _json_vector(
            np.mean(np.abs(momentum_error[-min(20, run.rows) :]), axis=0)
        ),
        "terminal_root_height": float(run.arrays["qpos"][-1, 2]),
    }
    for level, row in onset_rows.items():
        result[f"sustained_{level}_onset_transition"] = (
            None if row is None else int(transition_values[row])
        )
    if onset_row is None:
        result["pre_onset_window"] = None
        return result

    lower = max(0, onset_row - PRE_ONSET_TRANSITIONS)
    upper = onset_row
    onset_transition = int(transition_values[onset_row])
    if support_transitions.size:
        distances = np.abs(support_transitions - onset_transition)
        nearest_index = int(np.argmin(distances))
        nearest_transition = int(support_transitions[nearest_index])
        nearest_distance = int(distances[nearest_index])
    else:
        nearest_transition = None
        nearest_distance = None

    joint_reference_error = np.mean(
        np.abs(
            run.arrays["joint_position"][lower:upper]
            - run.arrays["reference_joint_position"][lower:upper]
        ),
        axis=0,
    )
    joint_pd_error = np.mean(
        np.abs(
            run.arrays["position_target"][lower:upper]
            - run.arrays["joint_position"][lower:upper]
        ),
        axis=0,
    )
    window_momentum_error = momentum_error[lower:upper]
    result["pre_onset_window"] = {
        "start_transition": int(transition_values[lower]),
        "end_transition_inclusive": int(transition_values[upper - 1]),
        "row_count": int(upper - lower),
        "transitions_until_terminal": int(run.rows - onset_transition),
        "nearest_support_transition": nearest_transition,
        "nearest_support_transition_distance": nearest_distance,
        "support_transition_within_tolerance": bool(
            nearest_distance is not None
            and nearest_distance <= SUPPORT_PROXIMITY_TOLERANCE
        ),
        "centroidal_abs_error_mean": _json_vector(
            np.mean(np.abs(window_momentum_error), axis=0)
        ),
        "joint_reference_abs_error_mean": _json_vector(joint_reference_error),
        "joint_pd_target_abs_error_mean": _json_vector(joint_pd_error),
    }
    if run.label != "e002":
        mask = EXPECTED_MASKS[run.label]
        unmasked = run.arrays["learned_torso_wrench_unmasked"][lower:upper]
        missing = unmasked * (1.0 - mask)
        result["pre_onset_window"]["unmasked_wrench_world_mean"] = _json_vector(
            np.mean(unmasked, axis=0)
        )
        result["pre_onset_window"]["missing_wrench_world_mean"] = _json_vector(
            np.mean(missing, axis=0)
        )
        result["pre_onset_window"]["missing_wrench_world_vector_impulse"] = (
            _json_vector(np.sum(missing, axis=0) * control_dt)
        )
        result["pre_onset_window"]["missing_wrench_world_absolute_impulse"] = (
            _json_vector(np.sum(np.abs(missing), axis=0) * control_dt)
        )
    return result


def family_localization(
    analyses: dict[str, dict[str, object]], labels: tuple[str, str]
) -> dict[str, object]:
    """Summarize duplicate onset agreement and consensus pre-onset interval."""
    onsets = [
        int(analyses[label]["sustained_50pct_onset_transition"]) for label in labels
    ]
    windows = [analyses[label]["pre_onset_window"] for label in labels]
    starts = [int(window["start_transition"]) for window in windows]
    ends = [int(window["end_transition_inclusive"]) for window in windows]
    intersection_start = max(starts)
    intersection_end = min(ends)
    return {
        "labels": list(labels),
        "onset_transitions": onsets,
        "onset_spread": max(onsets) - min(onsets),
        "duplicates_localized_within_tolerance": bool(
            max(onsets) - min(onsets) <= DUPLICATE_ONSET_TOLERANCE
        ),
        "consensus_pre_onset_interval": (
            [intersection_start, intersection_end]
            if intersection_start <= intersection_end
            else None
        ),
        "support_proximity_passes": [
            bool(window["support_transition_within_tolerance"]) for window in windows
        ],
    }


def joint_clues(
    runs: dict[str, RunData], analyses: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Rank descriptive treatment excess over successful controls at each phase."""
    joint_names = runs["full-a"].arrays["actor_joint_names"].tolist()
    result: dict[str, object] = {}
    for family, labels in FAMILIES.items():
        reference_excesses = []
        pd_excesses = []
        for label in labels:
            window = analyses[label]["pre_onset_window"]
            start = int(window["start_transition"]) - 1
            end = int(window["end_transition_inclusive"])
            run = runs[label]
            treatment_reference = np.mean(
                np.abs(
                    run.arrays["joint_position"][start:end]
                    - run.arrays["reference_joint_position"][start:end]
                ),
                axis=0,
            )
            treatment_pd = np.mean(
                np.abs(
                    run.arrays["position_target"][start:end]
                    - run.arrays["joint_position"][start:end]
                ),
                axis=0,
            )
            control_reference = np.mean(
                [
                    np.mean(
                        np.abs(
                            runs[control].arrays["joint_position"][start:end]
                            - runs[control].arrays["reference_joint_position"][
                                start:end
                            ]
                        ),
                        axis=0,
                    )
                    for control in CONTROL_LABELS
                ],
                axis=0,
            )
            control_pd = np.mean(
                [
                    np.mean(
                        np.abs(
                            runs[control].arrays["position_target"][start:end]
                            - runs[control].arrays["joint_position"][start:end]
                        ),
                        axis=0,
                    )
                    for control in CONTROL_LABELS
                ],
                axis=0,
            )
            reference_excesses.append(treatment_reference - control_reference)
            pd_excesses.append(treatment_pd - control_pd)
        reference_excess = np.mean(reference_excesses, axis=0)
        pd_excess = np.mean(pd_excesses, axis=0)

        def ranked(values: np.ndarray) -> list[dict[str, object]]:
            return [
                {"joint": joint_names[index], "excess_radians": float(values[index])}
                for index in np.argsort(values)[::-1][:8]
            ]

        result[family] = {
            "reference_tracking_excess_top8": ranked(reference_excess),
            "pd_target_error_excess_top8": ranked(pd_excess),
            "reference_tracking_excess_all": _json_vector(reference_excess),
            "pd_target_error_excess_all": _json_vector(pd_excess),
        }
    return {"actor_joint_names": joint_names, "families": result}


def write_plot(
    path: Path,
    runs: dict[str, RunData],
    analyses: dict[str, dict[str, object]],
    clues: dict[str, object],
    *,
    verdict: str,
) -> None:
    """Write one compact plot containing every registered localization output."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = {
        "e002": "#d62728",
        "full-a": "#2ca02c",
        "full-b": "#98df8a",
        "vertical-a": "#ff7f0e",
        "vertical-b": "#ffbb78",
        "novertical-a": "#1f77b4",
        "novertical-b": "#9ecae1",
    }
    figure, axes = plt.subplots(3, 2, figsize=(15, 13))
    ax_ratio, ax_height, ax_momentum, ax_support, ax_impulse, ax_joint = axes.ravel()
    focus_end = max(runs[label].rows for label in ("e002", *TREATMENT_LABELS)) + 2

    for label in RUN_LABELS:
        run = runs[label]
        transitions = run.arrays["values"][
            :, run.column_index["transition_phase"]
        ]
        ratios = np.max(normalized_termination_error(run), axis=1)
        ax_ratio.plot(
            transitions,
            ratios,
            color=colors[label],
            linewidth=1.8,
            label=DISPLAY_LABELS[label],
        )
        ax_height.plot(
            transitions,
            run.arrays["qpos"][:, 2],
            color=colors[label],
            linewidth=1.8,
            label=DISPLAY_LABELS[label],
        )
        momentum_error = (
            run.arrays["centroidal_momentum"][1:]
            - run.arrays["reference_centroidal_momentum"][1:]
        )
        ax_momentum.plot(
            transitions,
            np.linalg.norm(momentum_error[:, :3], axis=1),
            color=colors[label],
            linewidth=1.8,
            label=DISPLAY_LABELS[label],
        )
        onset = analyses[label]["sustained_50pct_onset_transition"]
        if onset is not None:
            row = int(onset) - 1
            ax_ratio.scatter(
                [onset], [ratios[row]], color=colors[label], edgecolor="black", zorder=4
            )
    ax_ratio.axhline(ONSET_FRACTION, color="black", linestyle="--", linewidth=1)
    ax_ratio.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax_ratio.set(
        title="Sustained preterminal onset",
        ylabel="max termination error / hard threshold",
        xlim=(1, focus_end),
    )
    ax_ratio.legend(ncol=2, fontsize=8)
    ax_height.set(
        title="Root height",
        ylabel="world z [m]",
        xlim=(1, focus_end),
    )
    ax_momentum.set(
        title="Centroidal linear-momentum error",
        ylabel="L2 error",
        xlabel="reference transition",
        xlim=(1, focus_end),
    )
    for axis in (ax_ratio, ax_height, ax_momentum):
        axis.grid(alpha=0.25)

    support_matrix = np.full((len(RUN_LABELS), focus_end), np.nan)
    for row, label in enumerate(RUN_LABELS):
        support = runs[label].arrays["foot_support"].astype(np.int8)
        code = support[:, 0] + 2 * support[:, 1]
        visible_rows = min(len(code), focus_end - 1)
        support_matrix[row, 1 : visible_rows + 1] = code[:visible_rows]
    cmap = ListedColormap(["#222222", "#e377c2", "#17becf", "#bcbd22"])
    cmap.set_bad("white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    image = ax_support.imshow(
        np.ma.masked_invalid(support_matrix),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        norm=norm,
        extent=(0, focus_end, len(RUN_LABELS) - 0.5, -0.5),
    )
    ax_support.set_yticks(range(len(RUN_LABELS)), [DISPLAY_LABELS[x] for x in RUN_LABELS])
    ax_support.set(
        title="Recorded foot-support topology",
        xlabel="reference transition",
        xlim=(1, focus_end),
    )
    colorbar = figure.colorbar(image, ax=ax_support, ticks=[0, 1, 2, 3], fraction=0.046)
    colorbar.ax.set_yticklabels(["none", "left", "right", "both"])

    impulse_rows = []
    impulse_labels = []
    for label in TREATMENT_LABELS:
        window = analyses[label]["pre_onset_window"]
        impulse_rows.append(window["missing_wrench_world_vector_impulse"][:3])
        impulse_labels.append(DISPLAY_LABELS[label])
    impulses = np.asarray(impulse_rows)
    x = np.arange(len(impulse_labels))
    width = 0.24
    for component, color in zip(range(3), ("#d62728", "#9467bd", "#2ca02c")):
        ax_impulse.bar(
            x + (component - 1) * width,
            impulses[:, component],
            width,
            label=("Fx", "Fy", "Fz")[component],
            color=color,
        )
    ax_impulse.axhline(0.0, color="black", linewidth=0.8)
    ax_impulse.set_xticks(x, impulse_labels, rotation=18, ha="right")
    ax_impulse.set(
        title=f"Missing force impulse in {PRE_ONSET_TRANSITIONS}-transition window",
        ylabel="world impulse [N s]",
    )
    ax_impulse.legend()
    ax_impulse.grid(axis="y", alpha=0.25)

    family_clues = clues["families"]
    names = clues["actor_joint_names"]
    values_by_family = {
        family: np.asarray(details["pd_target_error_excess_all"])
        for family, details in family_clues.items()
    }
    union_rank = np.argsort(
        np.maximum(values_by_family["vertical-only"], values_by_family["no-vertical"])
    )[::-1][:8]
    y = np.arange(len(union_rank))
    ax_joint.barh(
        y - 0.18,
        values_by_family["vertical-only"][union_rank],
        0.36,
        color="#ff7f0e",
        label="vertical only",
    )
    ax_joint.barh(
        y + 0.18,
        values_by_family["no-vertical"][union_rank],
        0.36,
        color="#1f77b4",
        label="no vertical",
    )
    ax_joint.set_yticks(y, [names[index] for index in union_rank])
    ax_joint.invert_yaxis()
    ax_joint.axvline(0.0, color="black", linewidth=0.8)
    ax_joint.set(
        title="Pre-onset PD tracking demand above successful controls",
        xlabel="excess mean |target - joint| [rad]",
    )
    ax_joint.legend()
    ax_joint.grid(axis="x", alpha=0.25)

    figure.suptitle(f"G1 learned-wrench failure-window analysis: {verdict}", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in RUN_LABELS:
        option = label.replace("-", "_")
        parser.add_argument(f"--{label}-npz", dest=f"{option}_npz", type=Path, required=True)
        parser.add_argument(f"--{label}-sha256", dest=f"{option}_sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-reference-transitions", type=int, default=271)
    parser.add_argument("--control-dt", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_reference_transitions < 1:
        raise ValueError("expected reference transitions must be positive")
    if not np.isfinite(args.control_dt) or args.control_dt <= 0.0:
        raise ValueError("control dt must be finite and positive")
    runs = {}
    for label in RUN_LABELS:
        option = label.replace("-", "_")
        runs[label] = load_run(
            label=label,
            path=getattr(args, f"{option}_npz"),
            expected_sha256=getattr(args, f"{option}_sha256"),
            expected_reference_transitions=args.expected_reference_transitions,
        )
    joint_names = runs["full-a"].arrays["actor_joint_names"]
    for label, run in runs.items():
        np.testing.assert_array_equal(
            run.arrays["actor_joint_names"],
            joint_names,
            err_msg=f"{label} joint ordering differs from full-a",
        )

    analyses = {
        label: analyze_run(run, control_dt=args.control_dt)
        for label, run in runs.items()
    }
    family_results = {
        family: family_localization(analyses, labels)
        for family, labels in FAMILIES.items()
    }
    treatment_onsets = [
        int(analyses[label]["sustained_50pct_onset_transition"])
        for label in TREATMENT_LABELS
    ]
    duplicates_localized = all(
        result["duplicates_localized_within_tolerance"]
        for result in family_results.values()
    )
    shared_onset = (
        max(treatment_onsets) - min(treatment_onsets) <= SHARED_ONSET_TOLERANCE
    )
    unanimous_support_proximity = all(
        all(result["support_proximity_passes"])
        for result in family_results.values()
    )
    if not duplicates_localized:
        verdict = "not-localized"
    elif shared_onset and unanimous_support_proximity:
        verdict = "shared-stance-transfer-window"
    else:
        verdict = "component-specific-runaway-not-shared-stance-transfer"

    vertical_interval = family_results["vertical-only"][
        "consensus_pre_onset_interval"
    ]
    novertical_interval = family_results["no-vertical"][
        "consensus_pre_onset_interval"
    ]
    interval_gap = None
    if vertical_interval is not None and novertical_interval is not None:
        interval_gap = int(novertical_interval[0] - vertical_interval[1] - 1)
    clues = joint_clues(runs, analyses)
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_only": True,
        "simulator_constructed": False,
        "policy_evaluated": False,
        "training_run": False,
        "constants": {
            "control_dt": args.control_dt,
            "onset_fraction": ONSET_FRACTION,
            "pre_onset_transitions": PRE_ONSET_TRANSITIONS,
            "duplicate_onset_tolerance": DUPLICATE_ONSET_TOLERANCE,
            "shared_onset_tolerance": SHARED_ONSET_TOLERANCE,
            "support_proximity_tolerance": SUPPORT_PROXIMITY_TOLERANCE,
            "termination_channels": list(TERMINATION_CHANNELS),
            "termination_thresholds": _json_vector(TERMINATION_THRESHOLDS),
            "wrench_component_order": list(WRENCH_COMPONENTS),
        },
        "runs": analyses,
        "families": family_results,
        "pooled_treatment_onset_spread": max(treatment_onsets)
        - min(treatment_onsets),
        "consensus_pre_onset_interval_gap": interval_gap,
        "all_duplicate_families_localized": duplicates_localized,
        "shared_treatment_onset": shared_onset,
        "unanimous_support_proximity": unanimous_support_proximity,
        "joint_clues": clues,
        "verdict": verdict,
        "claim_scope": {
            "established": [
                "Each masked treatment has a duplicate-consistent sustained preterminal onset.",
                "The vertical-only and no-vertical onset windows are not one shared narrow phase window.",
                "Recorded support transitions do not unanimously coincide with the fixed onset gate.",
                "The retained unassisted E002 rollout ends through the same height-error family observed when vertical learned force is absent.",
            ],
            "not_established": [
                "A particular foot contact or stance transfer causally triggers failure.",
                "Any single horizontal-force or torque axis is sufficient.",
                "The ranked joint deviations are causal training targets.",
                "A learned torso wrench is part of the intended final controller.",
            ],
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "failure_window_analysis.json"
    plot_path = output_dir / "failure_window_analysis.png"
    write_plot(plot_path, runs, analyses, clues, verdict=verdict)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "plot": str(plot_path), "verdict": verdict}))


if __name__ == "__main__":
    main()
