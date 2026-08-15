"""Build the targeted E023 LAFAN bank and compact recovery support."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np

from src.algorithms.shac.progressive_recovery_expert import (
    build_recovery_support,
)
from src.algorithms.shac.residual_preview_adapter import (
    FrozenPreviewResidualParams,
)
from src.envs.g1_tracking.environment import DEFAULT_CONTROLLER_PATH
from tools.build_g1_e023_carried_reset_bank import (
    E023_TOTAL_STEPS,
    validate_code_commit,
    validate_e023_hparams,
)
from tools.build_g1_e023_lafan_carried_reset_bank import (
    EXPECTED_REFERENCE_SHA256,
    LAFAN_SOURCE_PHASES,
    PROTOCOL as SOURCE_BANK_PROTOCOL,
)
from tools.build_g1_history_carried_reset_bank import (
    _write_json_atomically,
    _write_npz_atomically,
)
from tools.run_g1_root_recovery_continuation import validate_runtime_assets


PROTOCOL = "g1-progressive-recovery-support-v1"
TARGETED_BANK_PROTOCOL = "g1-e023-lafan-phase0-targeted-bank-v1"
SOURCE_ROWS = 120
ROWS_PER_SOURCE = 24
HISTORY_LEN = 10
FRAME_DIM = 328
TAPER = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_layout(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if "source_start_phase" not in arrays:
        raise ValueError("source bank is missing source_start_phase")
    source_start = np.asarray(arrays["source_start_phase"])
    expected = np.repeat(np.asarray(LAFAN_SOURCE_PHASES), ROWS_PER_SOURCE)
    if source_start.shape != (SOURCE_ROWS,) or not np.array_equal(
        source_start, expected
    ):
        raise ValueError("source bank must contain five exact 24-row bands")
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype == object or not np.isfinite(array).all():
            raise ValueError(f"source bank array {name} must be finite numeric data")
        if array.ndim > 0 and array.shape[0] not in (SOURCE_ROWS, 4):
            raise ValueError(f"source bank array {name} has an invalid row count")
    return source_start


def build_targeted_bank(
    source: Mapping[str, np.ndarray], *, source_phase: int = 0
) -> dict[str, np.ndarray]:
    """Slice one immutable 24-row source band without changing its values."""
    source_start = _validate_source_layout(source)
    if source_phase not in LAFAN_SOURCE_PHASES:
        raise ValueError("targeted source phase is not registered")
    selected = source_start == source_phase
    if int(np.sum(selected)) != ROWS_PER_SOURCE:
        raise ValueError("targeted source phase must contain exactly 24 rows")
    targeted: dict[str, np.ndarray] = {}
    for name, values in source.items():
        array = np.asarray(values)
        targeted[name] = array[selected].copy() if array.shape[:1] == (SOURCE_ROWS,) else array.copy()
    if not np.array_equal(
        targeted["source_start_phase"],
        np.full(ROWS_PER_SOURCE, source_phase, dtype=source_start.dtype),
    ):
        raise ValueError("targeted bank source phase changed during slicing")
    return targeted


def build_support_artifact(
    positive_frames: np.ndarray,
    negative_frames: np.ndarray,
    positive_phases: np.ndarray,
    *,
    source_bank_sha256: str,
    checkpoint_sha256: str,
    reference_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build hash-bound arrays and a JSON-safe support report."""
    for value, label in (
        (source_bank_sha256, "source bank"),
        (checkpoint_sha256, "checkpoint"),
        (reference_sha256, "reference"),
    ):
        if len(value) != 64:
            raise ValueError(f"{label} SHA-256 must contain 64 hex characters")
    support, report = build_recovery_support(
        positive_frames,
        negative_frames,
        positive_phases,
        taper=TAPER,
        minimum_positive_coverage=20,
    )
    arrays = {
        "anchors": np.asarray(support.anchors, dtype=np.float32),
        "radius": np.asarray(support.radius, dtype=np.float32),
        "phase_min": np.asarray(support.phase_min, dtype=np.int32),
        "phase_max": np.asarray(support.phase_max, dtype=np.int32),
        "taper": np.asarray(support.taper, dtype=np.int32),
        "positive_leave_one_out_distances": np.asarray(
            report["positive_leave_one_out_distances"], dtype=np.float32
        ),
        "protected_negative_distances": np.asarray(
            report["protected_negative_distances"], dtype=np.float32
        ),
    }
    summary = {
        **report,
        "protocol": PROTOCOL,
        "source_bank_sha256": source_bank_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "reference_sha256": reference_sha256,
    }
    return arrays, summary


def publish_artifacts(
    *,
    output_directory: Path,
    targeted_arrays: Mapping[str, np.ndarray],
    support_arrays: Mapping[str, np.ndarray],
    support_summary: dict[str, object],
) -> dict[str, object]:
    """Atomically publish NPZ evidence before manifest-last JSON."""
    output_directory = output_directory.resolve()
    targeted_path = output_directory / "e023_lafan_phase0_targeted_bank.npz"
    support_path = output_directory / "e023_lafan_phase0_recovery_support.npz"
    targeted_manifest_path = targeted_path.with_suffix(".json")
    support_manifest_path = support_path.with_suffix(".json")
    _write_npz_atomically(targeted_path, targeted_arrays)
    _write_npz_atomically(support_path, support_arrays)
    targeted_sha256 = _sha256(targeted_path)
    support_sha256 = _sha256(support_path)
    targeted_manifest = {
        "valid": True,
        "protocol": TARGETED_BANK_PROTOCOL,
        "rows": ROWS_PER_SOURCE,
        "source_phase": 0,
        "targeted_bank_path": str(targeted_path),
        "targeted_bank_sha256": targeted_sha256,
    }
    support_manifest = {
        **support_summary,
        "valid": True,
        "support_path": str(support_path),
        "support_sha256": support_sha256,
        "targeted_bank_path": str(targeted_path),
        "targeted_bank_sha256": targeted_sha256,
    }
    _write_json_atomically(targeted_manifest_path, targeted_manifest)
    _write_json_atomically(support_manifest_path, support_manifest)
    return {
        "targeted_bank_path": targeted_path,
        "targeted_bank_sha256": targeted_sha256,
        "targeted_bank_manifest_path": targeted_manifest_path,
        "support_path": support_path,
        "support_sha256": support_sha256,
        "support_manifest_path": support_manifest_path,
    }


def collect_support_frames(
    checkpoint_path: Path,
    hparams_path: Path,
    reference_path: Path,
    targeted_bank: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect normalized positives and protected E023 trajectory states."""
    import jax
    import jax.numpy as jnp

    from src.core.data_structures import Normalizer
    from src.envs.g1_tracking.solver_profiles import (
        get_solver_profile,
        solver_context,
    )
    from tools.evaluate_g1_flax_phase_grid import (
        evaluate_actor_action,
        prepare_phase_grid_action,
    )
    from tools.evaluate_g1_tracking import (
        _load_policy,
        build_compiled_step,
        configure_jax,
        make_evaluation_env,
    )

    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    contract = validate_e023_hparams(hparams)
    configure_jax()
    profile = get_solver_profile("g1-4x5")
    env = make_evaluation_env(
        str(contract["env_variant"]),
        solver_iterations=int(contract["solver_iterations"]),
        solver_ls_iterations=int(contract["solver_ls_iterations"]),
        reference_path=reference_path,
        reference_stride=int(contract["reference_stride"]),
        actor_history_len=int(contract["actor_history_len"]),
        actor_reference_lookahead_steps=tuple(
            contract["actor_reference_lookahead_steps"]
        ),
        actor_reference_preview_mode=str(contract["actor_reference_preview_mode"]),
        actor_observation_noise=False,
        domain_randomization=False,
        reference_reset_noise_scale=0.0,
        reference_residual_control=True,
        reference_residual_scale=1.0,
    )
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if isinstance(checkpoint.actor_params, FrozenPreviewResidualParams):
        raise ValueError("support parent must be the plain E023 actor")
    if int(np.asarray(checkpoint.step)) != E023_TOTAL_STEPS:
        raise ValueError("support parent checkpoint step does not match E023")
    actor, actor_params, normalizer_state = _load_policy(
        env, checkpoint_path, seed
    )
    normalizer = Normalizer(env.actor_frame_obs_dim)
    compiled_step = build_compiled_step(env)

    positive_histories = np.asarray(targeted_bank["actor_obs_history"])
    if positive_histories.shape != (ROWS_PER_SOURCE, HISTORY_LEN, FRAME_DIM):
        raise ValueError("targeted actor history shape does not match contract")
    positive_normalized = env.normalize_actor_obs(
        normalizer,
        normalizer_state,
        jnp.asarray(positive_histories.reshape(ROWS_PER_SOURCE, -1)),
    ).reshape(ROWS_PER_SOURCE, HISTORY_LEN, FRAME_DIM)
    positive_frames = np.asarray(positive_normalized[:, -1], dtype=np.float32)
    positive_phases = np.asarray(targeted_bank["phase"], dtype=np.int32)
    phase_min = int(np.min(positive_phases))
    phase_max = int(np.max(positive_phases))

    def action_fn(state):
        normalized = env.normalize_actor_obs(
            normalizer, normalizer_state, state.obs
        ).astype(jnp.float32)
        return prepare_phase_grid_action(
            evaluate_actor_action(actor, actor_params, normalized),
            clip_sampled_actor_actions=bool(contract["clip_sampled_actor_actions"]),
        ).astype(jnp.float64)

    protected_frames: list[np.ndarray] = []
    protected_phases: list[int] = []
    source_survival: list[int] = []
    with solver_context(profile):
        for source_phase in LAFAN_SOURCE_PHASES:
            state = env.reset_at_phase(
                jax.random.PRNGKey(seed),
                jnp.asarray(0.0),
                jnp.asarray(source_phase),
            )
            steps = 0
            for _ in range(int(env.reference_transitions) - source_phase):
                phase = int(state.info["phase"])
                normalized = env.normalize_actor_obs(
                    normalizer, normalizer_state, state.obs
                ).reshape(HISTORY_LEN, FRAME_DIM)
                protect = (source_phase == 0 and phase < phase_min) or (
                    source_phase != 0
                    and phase_min - TAPER <= phase <= phase_max + TAPER
                )
                if protect:
                    protected_frames.append(
                        np.asarray(normalized[-1], dtype=np.float32)
                    )
                    protected_phases.append(phase)
                state = compiled_step(state, action_fn(state))
                steps += 1
                if float(state.done) > 0.5:
                    break
            if float(state.info["terminal"]) <= 0.5:
                raise ValueError("protected-state source did not terminate")
            source_survival.append(steps)
    if not protected_frames:
        raise ValueError("protected-state collection produced no negatives")
    negative_frames = np.stack(protected_frames)
    report = {
        "protected_rows": int(negative_frames.shape[0]),
        "protected_phases": protected_phases,
        "source_survival": source_survival,
        "positive_phase_min": phase_min,
        "positive_phase_max": phase_max,
    }
    return positive_frames, negative_frames, positive_phases, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--hparams-sha256", required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--source-bank-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    inputs = (
        (args.checkpoint.resolve(), args.checkpoint_sha256, "checkpoint"),
        (args.hparams.resolve(), args.hparams_sha256, "hparams"),
        (args.reference_path.resolve(), args.reference_sha256, "reference"),
        (args.source_bank.resolve(), args.source_bank_sha256, "source bank"),
    )
    for path, expected, label in inputs:
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{label} SHA-256 does not match")
    if args.reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference is not the registered LAFAN walk")
    code_commit = validate_code_commit(repository, args.code_commit)
    hparams = json.loads(args.hparams.read_text(encoding="utf-8"))
    validate_e023_hparams(hparams)
    runtime_assets = validate_runtime_assets(
        Path(str(hparams["xml_path"])), Path(DEFAULT_CONTROLLER_PATH)
    )
    with np.load(args.source_bank, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    targeted = build_targeted_bank(source)
    positives, negatives, phases, collection = collect_support_frames(
        args.checkpoint,
        args.hparams,
        args.reference_path,
        targeted,
        seed=args.seed,
    )
    support_arrays, support_summary = build_support_artifact(
        positives,
        negatives,
        phases,
        source_bank_sha256=args.source_bank_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        reference_sha256=args.reference_sha256,
    )
    published = publish_artifacts(
        output_directory=args.output_directory,
        targeted_arrays=targeted,
        support_arrays=support_arrays,
        support_summary={
            **support_summary,
            **collection,
            **runtime_assets,
            "code_commit": code_commit,
            "source_bank_protocol": SOURCE_BANK_PROTOCOL,
            "seed": args.seed,
        },
    )
    print(json.dumps({key: str(value) for key, value in published.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
