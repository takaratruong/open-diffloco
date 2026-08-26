"""Recompute the hash-bound E006 capture-point value discriminator."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from src.envs.g1_tracking.centroidal_momentum import (
    reference_capture_points,
    standing_com_height,
)
from src.envs.g1_tracking.controller import load_rmr_controller
from src.envs.g1_tracking.reference import load_mujoco_reference
from tools.prepare_g1_rmr_reference import sha256_file
from tools.run_g1_zero_assistance_consolidation import _write_json_atomically


EXPECTED_TRACE_SHA256 = {
    "e026": "b2734fda552ffa47c22d37fab8dff856dbf849936bb900ba57aafaab908ccb49",
    "e004": "72fce58f63562dbda04a70f5d7ff022beaba0de2b04757759c652885e71f7adf",
    "e005": "b1cc58688dacddbbbbdc4d8c1782628cdfc0edc191410026b1259aed9e37973a",
}
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)
EXPECTED_CONTROLLER_SHA256 = (
    "f832285356d8fc10b226b6bbf557520d5323c7c9022ae6dbd00c683b06e5b7ee"
)


@dataclass(frozen=True)
class CaptureMetrics:
    """Registered normalized capture-point summary."""

    rms: float
    p99: float
    final: float
    component_rms: tuple[float, float]


def _validate_metrics(metrics: CaptureMetrics) -> None:
    values = np.asarray(
        (metrics.rms, metrics.p99, metrics.final, *metrics.component_rms),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("capture metrics must be finite and nonnegative")


def classify_capture_discriminator(
    *,
    assisted: CaptureMetrics,
    e026: CaptureMetrics,
    e005: CaptureMetrics,
) -> str:
    """Require both tail and distributional separation from both failures."""
    for metrics in (assisted, e026, e005):
        _validate_metrics(metrics)
    p99_separates = assisted.p99 <= 0.8 * min(e026.p99, e005.p99)
    final_separates = assisted.final <= 0.8 * min(e026.final, e005.final)
    return (
        "capture-signal-valid"
        if p99_separates and final_separates
        else "capture-signal-not-discriminating"
    )


def _capture_points_from_trace(
    model: mujoco.MjModel,
    trace_path: Path,
) -> np.ndarray:
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
        momentum = np.asarray(trace["centroidal_momentum"], dtype=np.float64)
    if (
        qpos.shape != (106, model.nq)
        or qvel.shape != (106, model.nv)
        or momentum.shape != (107, 6)
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or not np.isfinite(momentum).all()
    ):
        raise ValueError("capture trace does not match the 106-transition contract")
    root_body_id = int(model.body_rootid[1])
    total_mass = float(model.body_subtreemass[root_body_id])
    gravity = float(np.linalg.norm(model.opt.gravity))
    data = mujoco.MjData(model)
    rows = []
    for index, position in enumerate(qpos):
        data.qpos[:] = position
        data.qvel[:] = qvel[index]
        mujoco.mj_forward(model, data)
        com = data.subtree_com[root_body_id]
        omega = math.sqrt(gravity / float(com[2]))
        rows.append(com[:2] + momentum[index, :2] / total_mass / omega)
    return np.asarray(rows)


def compute_capture_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    standing_height: float,
) -> CaptureMetrics:
    """Compute the registered common-prefix capture summary."""
    actual_values = np.asarray(actual, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if (
        actual_values.shape != (106, 2)
        or reference_values.shape != actual_values.shape
        or not np.isfinite(actual_values).all()
        or not np.isfinite(reference_values).all()
        or not math.isfinite(standing_height)
        or standing_height <= 0.0
    ):
        raise ValueError("capture arrays do not match the registered prefix")
    error = (actual_values - reference_values) / standing_height
    norm = np.linalg.norm(error, axis=-1)
    metrics = CaptureMetrics(
        rms=float(np.sqrt(np.mean(np.square(norm)))),
        p99=float(np.quantile(norm, 0.99)),
        final=float(norm[-1]),
        component_rms=tuple(
            float(value) for value in np.sqrt(np.mean(np.square(error), axis=0))
        ),
    )
    _validate_metrics(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e026", type=Path, required=True)
    parser.add_argument("--e004", type=Path, required=True)
    parser.add_argument("--e005", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    traces = {"e026": args.e026, "e004": args.e004, "e005": args.e005}
    for label, path in traces.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_TRACE_SHA256[label]:
            raise ValueError(f"{label} trace SHA-256 does not match")
    for path, expected, label in (
        (args.model, EXPECTED_MODEL_SHA256, "model"),
        (args.controller, EXPECTED_CONTROLLER_SHA256, "controller"),
        (args.reference, EXPECTED_REFERENCE_SHA256, "reference"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    controller = load_rmr_controller(model, args.controller)
    reference = load_mujoco_reference(
        model, args.reference, controller=controller
    )
    root_body_id = int(model.body_rootid[1])
    reference_capture = reference_capture_points(
        model,
        reference.qpos[:106],
        reference.qvel[:106],
        root_body_id,
    )
    height = standing_com_height(model, model.qpos0, root_body_id)
    metrics = {
        label: compute_capture_metrics(
            _capture_points_from_trace(model, path),
            reference_capture,
            standing_height=height,
        )
        for label, path in traces.items()
    }
    outcome = classify_capture_discriminator(
        assisted=metrics["e004"], e026=metrics["e026"], e005=metrics["e005"]
    )
    report = {
        "valid": True,
        "protocol": "g1-capture-point-discriminator-v1",
        "outcome": outcome,
        "standing_height": height,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "trace_sha256": EXPECTED_TRACE_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
    }
    _write_json_atomically(args.output.resolve(), report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
