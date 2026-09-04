"""Correct E007's rank-3 contact reduction from its immutable raw trajectories.

This artifact-only successor performs no simulator step, policy evaluation,
derivative, or optimizer update.  It strict-loads the audited E007 trajectory
archive and recomputes only the derived comparison, boundary, and plot outputs
with the corrected row-wise contact reduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.g1_success_failure_visitation.run import (
    DIFFSIM_SURVIVAL,
    PHASES,
    PPO_SURVIVAL,
    TERMINATION_LIMITS,
    TRANSITION_METRICS,
    classify_frozen_controls,
    compare_traces,
    plot_comparison,
    read_json,
    repository_preflight,
    sha256_file,
    summarize_trace,
    write_json,
)


TRACE_FIELDS = (
    "phase",
    "qpos",
    "qvel",
    "model_action",
    "position_target",
    "last_action",
    "foot_support",
    "contact_pairs",
    "constraint_force_root",
    "reward",
    "done",
    "terminal",
    "metrics",
)
SOURCE_HASHES = {
    "experiment.yaml": "02ebf0fc31f92d454d770c34a7b61d2f9f43d36144d40b35171fb9b67f1231e9",
    "run.json": "e6465bf05845d6294f88d374e403e75f8c61ad4a32d77fa3105f6b9264e8f010",
    "seed-0/visitation/preflight.json": (
        "fefcbcccc3f40af1684684cf14c4525da5709e407f23eebf91ec9409bcef4f34"
    ),
    "seed-0/visitation/paired_trajectories.npz": (
        "dc4199fa5383e7caf31c89bb56c7d261af6561ce237d48e8e217276827dbc89b"
    ),
    "seed-0/visitation/selected_h1_boundaries.json": (
        "0a0c776c2f5549de8ceee1dbea428c2eb8e3a3d4cbe4d5e0eceb577b1ec76f64"
    ),
    "seed-0/visitation/visitation_comparison.png": (
        "dba6abd8a6eaf7dcc9d1bdb3b1c0d268d8438396ba00298c5b8987c620f72d72"
    ),
    "seed-0/visitation/summary.json": (
        "b26e6d1fb18ff9c56c3167b9e62a5e7910b816cbdf597662f1f924e7d0be5762"
    ),
    "seed-0/visitation/completion.json": (
        "f41fce42ef616222d2c7de7fe72e3aff21f0f047effc5991aab2dfdad586df68"
    ),
    "independent_evaluation.json": (
        "31239e1ab0db34db5ab4f4a667fd5bab6e082f51dbeac26a5f88ceca50a49c62"
    ),
}


def validate_source_run(source_root: Path) -> dict[str, Any]:
    """Require the exact E007 raw package and its expected two audit failures."""
    observed = {name: sha256_file(source_root / name) for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        raise ValueError("E007 source artifact hashes changed")
    run = read_json(source_root / "run.json")
    source_summary = read_json(source_root / "seed-0/visitation/summary.json")
    audit = read_json(source_root / "independent_evaluation.json")
    if (
        run.get("experiment") != "E-20260904-007"
        or run.get("return_code") != 0
        or run.get("timed_out") is not False
        or len(run.get("attempts", [])) != 1
        or source_summary.get("fresh_ppo_survival") != list(PPO_SURVIVAL)
        or source_summary.get("fresh_diffsim_survival") != [124, 135, 81, 92, 79]
        or audit.get("valid") is not False
        or audit.get("outcome") != "invalid-execution"
        or audit.get("checks_passed") != 19
        or audit.get("checks_total") != 21
        or audit.get("failed_checks")
        != ["comparison-arithmetic-recomputed", "boundary-selection-recomputed"]
        or audit.get("policy_retained") is not False
    ):
        raise ValueError("E007 source classification changed")
    return {
        "hashes": observed,
        "run_duration_seconds": run["duration_seconds"],
        "source_ppo_survival": source_summary["fresh_ppo_survival"],
        "source_diffsim_survival": source_summary["fresh_diffsim_survival"],
        "source_audit_failed_checks": audit["failed_checks"],
    }


def load_traces(path: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Strict-load E007's flattened trace archive without object arrays."""
    traces: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "ppo": {},
        "diffsim": {},
    }
    with np.load(path, allow_pickle=False) as archive:
        if (
            archive["metric_names"].tolist() != list(TRANSITION_METRICS)
            or not np.array_equal(archive["termination_limits"], TERMINATION_LIMITS)
            or archive["phases"].tolist() != list(PHASES)
        ):
            raise ValueError("E007 trajectory metadata changed")
        expected = {"metric_names", "termination_limits", "phases"}
        expected.update(
            f"{arm}_phase_{phase:03d}_{field}"
            for arm in traces
            for phase in PHASES
            for field in TRACE_FIELDS
        )
        if set(archive.files) != expected:
            raise ValueError("E007 trajectory keyset changed")
        for arm in traces:
            for phase in PHASES:
                prefix = f"{arm}_phase_{phase:03d}_"
                trace = {
                    field: np.asarray(archive[prefix + field]) for field in TRACE_FIELDS
                }
                if not all(
                    not np.issubdtype(value.dtype, np.number)
                    or np.isfinite(value).all()
                    for value in trace.values()
                ):
                    raise ValueError("E007 trajectory contains a nonfinite value")
                traces[arm][phase] = trace
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    args = parser.parse_args()

    source_root = args.source_run_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    source = validate_source_run(source_root)
    preflight = {
        "protocol": "g1-success-failure-visitation-reanalysis-preflight-v1",
        "valid": True,
        "code": repository_preflight(repository, args.code_commit),
        "source_run_root": str(source_root),
        "source": source,
        "seed": args.seed,
        "simulator_step_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "policy_retained": False,
    }
    write_json(output_root / "preflight.json", preflight)

    trajectory_path = source_root / "seed-0/visitation/paired_trajectories.npz"
    traces = load_traces(trajectory_path)
    summaries = {arm: [] for arm in traces}
    comparisons = []
    selected_boundaries = []
    for phase in PHASES:
        remaining = 271 - phase
        for arm in traces:
            summaries[arm].append(
                summarize_trace(traces[arm][phase], phase=phase, remaining=remaining)
            )
        comparison, selected = compare_traces(
            traces["ppo"][phase], traces["diffsim"][phase], phase=phase
        )
        comparisons.append(comparison)
        selected_boundaries.extend(
            {
                "start_phase": phase,
                "offset": offset,
                "absolute_phase": phase + offset,
                "ppo_trace_key": f"ppo_phase_{phase:03d}",
                "diffsim_trace_key": f"diffsim_phase_{phase:03d}",
            }
            for offset in selected
        )

    classification = classify_frozen_controls(summaries["ppo"], summaries["diffsim"])
    selected_path = output_root / "selected_h1_boundaries.json"
    write_json(
        selected_path,
        {
            "protocol": "g1-success-failure-selected-h1-boundaries-v2",
            "selection": "quartiles-plus-first-state-target-support-contact-divergence",
            "contact_reduction": "any-over-all-non-time-axes",
            "count": len(selected_boundaries),
            "boundaries": selected_boundaries,
            "source_trajectory_path": str(trajectory_path),
            "source_trajectory_sha256": sha256_file(trajectory_path),
        },
    )
    plot_path = output_root / "visitation_comparison.png"
    plot_comparison(plot_path, traces)
    summary = {
        "protocol": "g1-success-failure-visitation-reanalysis-v1",
        "valid": True,
        "outcome": classification["outcome"],
        "correction": (
            "reduce 31x31 contact-pair differences over both non-time axes; "
            "all immutable raw trajectory values are reused"
        ),
        "source_run_id": "E-20260904-007/20260904T160752Z",
        "source_trajectory_path": str(trajectory_path),
        "source_trajectory_sha256": sha256_file(trajectory_path),
        "phases": list(PHASES),
        "ppo": summaries["ppo"],
        "diffsim": summaries["diffsim"],
        "comparisons": comparisons,
        "selected_h1_boundary_count": len(selected_boundaries),
        "ppo_survival": classification["ppo_survival"],
        "diffsim_survival": classification["diffsim_survival"],
        "historical_diffsim_cpu_survival": list(DIFFSIM_SURVIVAL),
        "exact_reset_pairing": True,
        "nominal_unassisted": True,
        "simulator_step_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "selected_h1_boundaries.json": sha256_file(selected_path),
            "visitation_comparison.png": sha256_file(plot_path),
        },
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    completion = {
        "protocol": "g1-success-failure-visitation-reanalysis-completion-v1",
        "valid": True,
        "outcome": summary["outcome"],
        "simulator_step_computed": False,
        "policy_evaluation_computed": False,
        "policy_update_computed": False,
        "optimizer_update_retained": False,
        "policy_retained": False,
        "retained_policy": "E-20260826-002",
        "artifacts": {
            "preflight.json": sha256_file(output_root / "preflight.json"),
            "selected_h1_boundaries.json": sha256_file(selected_path),
            "visitation_comparison.png": sha256_file(plot_path),
            "summary.json": sha256_file(summary_path),
            "source_paired_trajectories.npz": sha256_file(trajectory_path),
        },
    }
    write_json(output_root / "completion.json", completion)
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "ppo_survival": classification["ppo_survival"],
                "diffsim_survival": classification["diffsim_survival"],
                "selected_h1_boundary_count": len(selected_boundaries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
