"""Localize same-process G1 SHAC nondeterminism within one compiled update."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

from src.algorithms.shac.algorithm import numeric_tree_sha256, train
from src.envs.g1_tracking.solver_profiles import get_solver_profile, solver_context
from tools.run_g1_jave_inprocess_pair import build_branch_kwargs, sha256_file
from tools.run_g1_tracking_shac import configure_jax

SOURCE_STEP = 1_880_064
EXPECTED_SOURCE_SHA256 = (
    "42b8c1d2fc0eca353f12b87c70ce9d8d091bdd15a2be3dac0d25759d3bfd8cbb"
)
EXPECTED_SOURCE_STATE_SHA256 = (
    "20a9546f63e8789c39f8381d3b670b05c22e64d73e611ea9abb6c55c2968b54a"
)
EXPECTED_SOURCE_HPARAMS_SHA256 = (
    "ebc9cfc4f0a44517cff25accfe3ff40a157aa6f7d21da3e3629182fc2a7ea4f6"
)
EXPECTED_REFERENCE_SHA256 = (
    "5bf1c08990818b39d62b8e3977e2368abf74d71a0d9dbf2de7d8f2ea5c3ae934"
)
EXPECTED_MODEL_SHA256 = (
    "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
)
MODEL_PATH = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)
BOUNDARIES = (
    "random_inputs",
    "first_actor_action",
    "first_mjx_substep",
    "first_mjx_control_step",
    "first_env_step",
    "rollout",
    "actor_cagrad",
    "learned_dynamics",
    "critic",
)

FIRST_MJX_SUBSTEP_COMPONENTS = (
    "integrated_state",
    "acceleration_state",
    "constraint_force",
    "contact_state",
)

FIRST_MJX_SUBSTEP_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "qacc",
    "qacc_smooth",
    "qacc_warmstart",
    "qfrc_applied",
    "qfrc_smooth",
    "qfrc_constraint",
    "efc_force",
    "contact",
)


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite_tree(tree: object) -> bool:
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            return False
    return True


def build_probe_kwargs(
    profile_name: str,
    reference_path: str | Path,
    seed: int,
    resume_from: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reuse the exact weight-zero post-warm-up JAVE continuation route."""

    kwargs = build_branch_kwargs(
        profile_name,
        reference_path,
        seed,
        resume_from,
        enabled=False,
        active_updates=1,
    )
    kwargs["determinism_probe_output"] = str(output_path)
    return kwargs


def classify_probe(report: dict[str, object]) -> dict[str, object]:
    """Map a valid probe report to its preregistered localization outcome."""

    first_mismatch = report.get("first_mismatch_boundary")
    if first_mismatch is not None and first_mismatch not in BOUNDARIES:
        raise ValueError("probe named an unknown first mismatch boundary")
    if report.get("valid") is True:
        outcome = "compiled-update-exact"
    elif first_mismatch is not None:
        outcome = f"compiled-update-diverges-{first_mismatch.replace('_', '-')}"
    elif (
        report.get("full_state_exact") is False
        or report.get("metrics_exact") is False
    ):
        outcome = "compiled-update-diverges-unlocalized"
    else:
        raise ValueError("probe report has no classifiable exactness result")
    return {
        "protocol": "g1-compiled-update-determinism-selection-v5",
        "outcome": outcome,
        "scientific_valid": True,
        "first_mismatch_boundary": first_mismatch,
        "mismatching_first_mjx_substep_components": report.get(
            "mismatching_first_mjx_substep_components", []
        ),
        "mismatching_first_mjx_substep_fields": report.get(
            "mismatching_first_mjx_substep_fields", []
        ),
        "full_state_exact": report.get("full_state_exact"),
        "metrics_exact": report.get("metrics_exact"),
        "policy_retained": False,
    }


def validate_preflight(
    *,
    repository: Path,
    checkpoint: Path,
    reference: Path,
    code_commit: str,
) -> dict[str, object]:
    """Bind the exact executable, common branch, model, and reference."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hparams_path = checkpoint.parent / "hparams.json"
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    required_hparams = {
        "env_variant": "g1_tracking_rmr_50hz_action_parity",
        "tracking_root_velocity_weight": 1.0,
        "actor_bootstrap_scale": 1.0,
        "actor_frozen_controller_residual": True,
        "actor_residual_preview_adapter": True,
        "actor_cagrad": True,
        "gradient_accumulation_steps": 2,
        "num_envs": 256,
        "unroll_length": 24,
        "jave_vg_weight": 0.0,
        "jave_vg_warmup_steps": 0,
        "jave_start_step": 1_867_776,
        "jave_collect_transitions": True,
        "solver_profile": "g1-4x5",
        "solver_iterations": 4,
        "solver_ls_iterations": 5,
    }
    errors = []
    if head != code_commit:
        errors.append("code commit mismatch")
    if status:
        errors.append("code worktree is dirty")
    if sha256_file(checkpoint) != EXPECTED_SOURCE_SHA256:
        errors.append("common branch checkpoint hash mismatch")
    if sha256_file(hparams_path) != EXPECTED_SOURCE_HPARAMS_SHA256:
        errors.append("common branch hparams hash mismatch")
    if sha256_file(reference) != EXPECTED_REFERENCE_SHA256:
        errors.append("reference hash mismatch")
    if sha256_file(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        errors.append("model hash mismatch")
    if (
        int(state.step) != SOURCE_STEP
        or not _finite_tree(state)
        or numeric_tree_sha256(state) != EXPECTED_SOURCE_STATE_SHA256
    ):
        errors.append("common branch numeric state mismatch")
    if any(hparams.get(name) != value for name, value in required_hparams.items()):
        errors.append("common branch hparams mismatch")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "valid": True,
        "protocol": "g1-compiled-update-determinism-preflight-v5",
        "authoritative_entrypoint": (
            "python -m tools.run_g1_compiled_update_determinism"
        ),
        "execution_shape": (
            "one-process-one-gpu-one-compiled-callable-two-identical-inputs"
        ),
        "code_commit": head,
        "source_step": SOURCE_STEP,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
        "source_state_sha256": EXPECTED_SOURCE_STATE_SHA256,
        "source_hparams_sha256": EXPECTED_SOURCE_HPARAMS_SHA256,
        "reference": str(reference.resolve()),
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "model": str(MODEL_PATH),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "boundaries": list(BOUNDARIES),
        "first_mjx_substep_components": list(
            FIRST_MJX_SUBSTEP_COMPONENTS
        ),
        "first_mjx_substep_fields": list(FIRST_MJX_SUBSTEP_FIELDS),
    }


def validate_probe_artifacts(
    *,
    report_path: Path,
    run_directory: Path,
    returned_state: object,
) -> dict[str, object]:
    """Require a complete, internally consistent exactness report."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    hparams_path = run_directory / "hparams.json"
    hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    errors = []
    if report.get("protocol") != "shac-compiled-update-determinism-v5":
        errors.append("probe protocol mismatch")
    if report.get("input_step") != SOURCE_STEP:
        errors.append("probe input step mismatch")
    if report.get("input_state_sha256") != EXPECTED_SOURCE_STATE_SHA256:
        errors.append("probe input state mismatch")
    if report.get("compiled_callable_reused") is not True:
        errors.append("compiled callable reuse is unproven")
    if report.get("ordinary_training_loop_entered") is not False:
        errors.append("ordinary training loop was entered")
    boundary_reports = report.get("boundaries")
    if not isinstance(boundary_reports, dict) or set(boundary_reports) != set(
        BOUNDARIES
    ):
        errors.append("probe boundary set mismatch")
        boundary_reports = {}
    exactness = []
    for name in BOUNDARIES:
        boundary = boundary_reports.get(name)
        if (
            not isinstance(boundary, dict)
            or not isinstance(boundary.get("first"), list)
            or len(boundary["first"]) != 4
            or not isinstance(boundary.get("second"), list)
            or len(boundary["second"]) != 4
            or not isinstance(boundary.get("exact"), bool)
        ):
            errors.append(f"{name} boundary report is invalid")
        else:
            exactness.append(boundary["exact"])
    expected_first_mismatch = next(
        (name for name in BOUNDARIES if boundary_reports.get(name, {}).get("exact") is False),
        None,
    )
    if report.get("first_mismatch_boundary") != expected_first_mismatch:
        errors.append("first mismatch localization is inconsistent")
    component_reports = report.get("first_mjx_substep_components")
    if not isinstance(component_reports, dict) or set(component_reports) != set(
        FIRST_MJX_SUBSTEP_COMPONENTS
    ):
        errors.append("first MJX substep component set mismatch")
        component_reports = {}
    component_exactness = []
    for name in FIRST_MJX_SUBSTEP_COMPONENTS:
        component = component_reports.get(name)
        if (
            not isinstance(component, dict)
            or not isinstance(component.get("first"), list)
            or len(component["first"]) != 4
            or not isinstance(component.get("second"), list)
            or len(component["second"]) != 4
            or not isinstance(component.get("exact"), bool)
        ):
            errors.append(f"{name} first MJX substep component is invalid")
        else:
            component_exactness.append(component["exact"])
    expected_component_mismatches = [
        name
        for name in FIRST_MJX_SUBSTEP_COMPONENTS
        if component_reports.get(name, {}).get("exact") is False
    ]
    if report.get("mismatching_first_mjx_substep_components") != (
        expected_component_mismatches
    ):
        errors.append(
            "first MJX substep component mismatch list is inconsistent"
        )
    field_reports = report.get("first_mjx_substep_fields")
    if not isinstance(field_reports, dict) or set(field_reports) != set(
        FIRST_MJX_SUBSTEP_FIELDS
    ):
        errors.append("first MJX substep field set mismatch")
        field_reports = {}
    field_exactness = []
    for name in FIRST_MJX_SUBSTEP_FIELDS:
        field = field_reports.get(name)
        if (
            not isinstance(field, dict)
            or not isinstance(field.get("first"), list)
            or len(field["first"]) != 4
            or not isinstance(field.get("second"), list)
            or len(field["second"]) != 4
            or not isinstance(field.get("exact"), bool)
        ):
            errors.append(f"{name} first MJX substep field is invalid")
        else:
            field_exactness.append(field["exact"])
    expected_field_mismatches = [
        name
        for name in FIRST_MJX_SUBSTEP_FIELDS
        if field_reports.get(name, {}).get("exact") is False
    ]
    if report.get("mismatching_first_mjx_substep_fields") != (
        expected_field_mismatches
    ):
        errors.append("first MJX substep field mismatch list is inconsistent")
    if not isinstance(report.get("full_state_exact"), bool):
        errors.append("full state exactness is missing")
    if not isinstance(report.get("metrics_exact"), bool):
        errors.append("metrics exactness is missing")
    expected_valid = (
        len(exactness) == len(BOUNDARIES)
        and all(exactness)
        and len(component_exactness) == len(FIRST_MJX_SUBSTEP_COMPONENTS)
        and all(component_exactness)
        and len(field_exactness) == len(FIRST_MJX_SUBSTEP_FIELDS)
        and all(field_exactness)
        and report.get("full_state_exact") is True
        and report.get("metrics_exact") is True
    )
    if report.get("valid") is not expected_valid:
        errors.append("probe validity is inconsistent")
    for name in (
        "first_state_sha256",
        "second_state_sha256",
        "first_metrics_sha256",
        "second_metrics_sha256",
    ):
        value = report.get(name)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{name} is invalid")
    expected_output = str(report_path.resolve())
    required_hparams = {
        "determinism_probe": True,
        "determinism_probe_output": expected_output,
        "actor_bootstrap_scale": 1.0,
        "jave_vg_weight": 0.0,
        "jave_collect_transitions": True,
        "actor_cagrad": True,
        "gradient_accumulation_steps": 2,
        "num_envs": 256,
        "unroll_length": 24,
    }
    if any(hparams.get(name) != value for name, value in required_hparams.items()):
        errors.append("probe run hparams mismatch")
    returned_state_sha256 = numeric_tree_sha256(returned_state)
    if returned_state_sha256 != EXPECTED_SOURCE_STATE_SHA256:
        errors.append("probe advanced or changed its returned input state")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "valid": True,
        "protocol": "g1-compiled-update-determinism-validation-v5",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "run_directory": str(run_directory),
        "hparams": str(hparams_path),
        "hparams_sha256": sha256_file(hparams_path),
        "returned_state_sha256": returned_state_sha256,
        "probe": report,
    }


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    report_path = output_root / "determinism_probe.json"
    try:
        preflight = validate_preflight(
            repository=repository,
            checkpoint=args.resume_from.resolve(),
            reference=args.reference_path.resolve(),
            code_commit=args.code_commit,
        )
        _write_json_atomically(output_root / "preflight.json", preflight)
        configure_jax()
        kwargs = build_probe_kwargs(
            args.solver_profile,
            args.reference_path.resolve(),
            args.seed,
            args.resume_from.resolve(),
            report_path,
        )
        previous = Path.cwd()
        try:
            os.chdir(output_root)
            with solver_context(get_solver_profile(args.solver_profile)):
                returned_state, relative_run_directory = train(**kwargs)
        finally:
            os.chdir(previous)
        run_directory = (output_root / relative_run_directory).resolve()
        validation = validate_probe_artifacts(
            report_path=report_path,
            run_directory=run_directory,
            returned_state=returned_state,
        )
        _write_json_atomically(output_root / "validation.json", validation)
        selection = classify_probe(validation["probe"])
        selection.update(
            code_commit=args.code_commit,
            source_checkpoint_sha256=EXPECTED_SOURCE_SHA256,
            source_state_sha256=EXPECTED_SOURCE_STATE_SHA256,
            report=str(report_path),
            report_sha256=validation["report_sha256"],
            validation=str(output_root / "validation.json"),
            run_directory=str(run_directory),
        )
        _write_json_atomically(output_root / "selection.json", selection)
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _write_json_atomically(
            output_root / "selection.json",
            {
                "protocol": "g1-compiled-update-determinism-selection-v5",
                "outcome": "invalid-execution",
                "scientific_valid": False,
                "reason": f"{type(error).__name__}: {error}",
                "policy_retained": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-profile", choices=("g1-4x5",), required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed != 0:
        raise ValueError("compiled-update determinism probe seed must equal zero")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
