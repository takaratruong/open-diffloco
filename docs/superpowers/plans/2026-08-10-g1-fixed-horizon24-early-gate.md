# G1 Fixed-Horizon-24 Early Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable fixed-horizon-24 G1 SHAC runner and execute one bounded 393,216-transition experiment that either clears or stops at a replay-free robustness gate.

**Architecture:** Reuse `build_canonical_kwargs` as the single authority for every scientific setting, copy its dictionary, and replace only `unroll_length` and `total_steps`. Keep experiment orchestration in the existing DiffSim2Real YAML registry and guarded runner, then evaluate the final actor with the existing five-phase replay-free evaluator.

**Tech Stack:** Python 3.11, JAX 0.4.38, MuJoCo/MJX 3.9.0, Flax 0.10.2, pytest, Ruff, YAML experiment registry, `tools.runexp`.

## Global Constraints

- Change `unroll_length` from `12` to `24`.
- Change `total_steps` from `8_000_000` to `393_216` solely to enforce the bounded early gate.
- Preserve every other canonical scientific setting exactly, including seed 0, 256 environments, fixed-scan `g1-4x5`, rewards, actor, critic, optimizer, terminal critic, noise, randomization, curriculum, and action semantics.
- The dedicated parser exposes only solver profile, reference path, seed, and output root; it exposes no resume or scientific override.
- Do not resume E-20260810-003 and do not use replay, teleportation, or intermediate resets as completion evidence.
- Stop unless all artifacts and numeric leaves are finite, five-phase minimum survival is at least 19, and median survival is at least 48.
- Never bind any service to `0.0.0.0`.

---

### Task 1: Immutable fixed-horizon runner

**Files:**
- Create: `tests/test_g1_horizon24_shac_runner.py`
- Create: `tools/run_g1_horizon24_shac.py`

**Interfaces:**
- Consumes: `build_canonical_kwargs(profile_name: str, reference_path: str | Path, seed: int) -> dict`.
- Produces: `build_horizon24_kwargs(profile_name: str, reference_path: str | Path, seed: int) -> dict` and `build_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write the failing contract and parser tests**

```python
import unittest
from pathlib import Path


class G1Horizon24ShacRunnerTest(unittest.TestCase):
    def test_contract_differs_only_in_horizon_and_bounded_budget(self):
        from tools.run_canonical_g1_shac import build_canonical_kwargs
        from tools.run_g1_horizon24_shac import build_horizon24_kwargs

        canonical = build_canonical_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        candidate = build_horizon24_kwargs(
            "g1-4x5", Path("/tmp/dance.npz"), seed=42
        )
        self.assertEqual(set(candidate), set(canonical))
        differing = {
            name for name in canonical if canonical[name] != candidate[name]
        }
        self.assertEqual(differing, {"total_steps", "unroll_length"})
        self.assertEqual(candidate["total_steps"], 393_216)
        self.assertEqual(candidate["unroll_length"], 24)
        self.assertNotIn("resume_from", candidate)

    def test_parser_rejects_resume_and_scientific_overrides(self):
        from tools.run_g1_horizon24_shac import build_parser

        parser = build_parser()
        for arguments in (
            ["--resume-from", "/tmp/checkpoint.pkl"],
            ["--actor-lr", "0.001"],
            ["--total-steps", "8000000"],
            ["--unroll-length", "12"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_horizon24_shac_runner.py -q`

Expected: FAIL because `tools.run_g1_horizon24_shac` does not exist.

- [ ] **Step 3: Add the minimal dedicated runner**

```python
"""Run the bounded fixed-horizon-24 G1 SHAC experiment."""

import argparse
import os
from pathlib import Path

from src.algorithms.shac.algorithm import train
from src.envs.g1_tracking.environment import DEFAULT_REFERENCE_PATH
from src.envs.g1_tracking.solver_profiles import (
    SOLVER_PROFILES,
    get_solver_profile,
    solver_context,
)
from tools.run_canonical_g1_shac import build_canonical_kwargs
from tools.run_g1_tracking_shac import configure_jax


def build_horizon24_kwargs(
    profile_name: str, reference_path: str | Path, seed: int
) -> dict:
    """Copy the canonical contract and change only horizon and budget."""
    kwargs = build_canonical_kwargs(profile_name, reference_path, seed)
    kwargs["unroll_length"] = 24
    kwargs["total_steps"] = 393_216
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable G1 SHAC horizon-24 early-gate run."
    )
    parser.add_argument(
        "--solver-profile",
        required=True,
        choices=tuple(sorted(SOLVER_PROFILES)),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(DEFAULT_REFERENCE_PATH),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("g1_horizon24_runs")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_jax()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_horizon24_kwargs(
        args.solver_profile, args.reference_path.resolve(), args.seed
    )
    profile = get_solver_profile(args.solver_profile)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_root)
        with solver_context(profile):
            _, relative_save_dir = train(**kwargs)
    finally:
        os.chdir(previous_directory)
    print((output_root / relative_save_dir).resolve())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused and neighboring tests and verify GREEN**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest tests/test_g1_horizon24_shac_runner.py tests/test_canonical_g1_shac_runner.py -q`

Expected: 6 tests PASS.

- [ ] **Step 5: Run static verification**

Run: `/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tools/run_g1_horizon24_shac.py tests/test_g1_horizon24_shac_runner.py && /home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile tools/run_g1_horizon24_shac.py tests/test_g1_horizon24_shac_runner.py && git diff --check`

Expected: exit 0 with no Ruff, compilation, or whitespace errors.

- [ ] **Step 6: Commit and push the verified runner**

```bash
git add tools/run_g1_horizon24_shac.py tests/test_g1_horizon24_shac_runner.py docs/superpowers/plans/2026-08-10-g1-fixed-horizon24-early-gate.md
git commit -m "feat: add bounded G1 horizon-24 runner"
git push takaratruong research/g1-rmr-50hz-20260805
```

### Task 2: Register and execute the bounded experiment

**Files:**
- Create: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260810-004.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260810-004.md`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/02 Current State.md`

**Interfaces:**
- Consumes: the verified runner commit from Task 1 and reference SHA-256 `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- Produces: one schema-v1 approved experiment executed with `uv run python -m tools.runexp E-20260810-004`.

- [ ] **Step 1: Create E-20260810-004 from the established schema**

Register one causal change (`unroll_length: 12 -> 24`), parent `E-20260810-003`, hypothesis `H-SCI-001`, seed `[0]`, GPU count `1`, solver `g1-4x5-fixed-scan`, environment `conda:diffsim`, exact clean runner commit, and this command:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python tools/run_g1_horizon24_shac.py --solver-profile g1-4x5 --reference-path artifacts/E-20260808-000/reference/dance1_subject2_f122_422_50hz.npz --seed $DIFFSIM_SEED --output-root $DIFFSIM_SEED_DIR
```

Require `[policy_final.pkl, hparams.json, log.npy, diag_log.json]`. Register exactly three outcomes: `horizon24-advances`, `horizon24-finite-no-robust-gain`, and `invalid-execution`. Define advancement as finite completion plus replay-free five-phase minimum at least 19 and median at least 48.

- [ ] **Step 2: Validate and dry-run before approval**

Run: `uv run python tools/researchctl.py validate && uv run python -m tools.runexp E-20260810-004 --dry-run`

Expected: registry validation succeeds and the dry run prints one seed-0 command rooted under `runs/E-20260810-004/` without executing training.

- [ ] **Step 3: Build the Obsidian projection and commit registry state**

Run: `uv run python tools/researchctl.py build-vault && uv run python tools/researchctl.py validate && git diff --check`

Commit only the new experiment YAML, current state YAML, and generated E004/current-state notes; preserve `.obsidian` and canvas edits.

- [ ] **Step 4: Launch the guarded experiment detached on physical GPU 3**

Run: `tmux new-session -d -s e20260810-004 'cd /home/ubuntu/projects/diffsim2real-lab && uv run python -m tools.runexp E-20260810-004'`

Verify the run manifest, seed directory, process, first log output, and GPU utilization. Do not change code or inputs while it runs.

### Task 3: Evaluate and curate the hard stop gate

**Files:**
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/experiments/E-20260810-004.yaml`
- Modify: `/home/ubuntu/projects/diffsim2real-lab/research/state/current.yaml`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/experiments/E-20260810-004.md`
- Generated: `/home/ubuntu/projects/diffsim2real-lab/docs/vault/02 Current State.md`

**Interfaces:**
- Consumes: E004 final actor, required artifacts, reference, and existing `tools/evaluate_g1_phase_grid.py`.
- Produces: finite-state audit, replay-free survival vector for phases `0/100/200/300/400`, one preregistered verdict, and the next bounded decision.

- [ ] **Step 1: Verify normal completion and finite state**

Require the run process to exit zero, all four registered artifacts to exist, the final checkpoint and every numeric TrainState leaf to be finite, and artifact hashes to be recorded. Otherwise select `invalid-execution` and stop.

- [ ] **Step 2: Evaluate exact phase zero, then phases 100/200/300/400 concurrently**

Use nominal physics, the exact final checkpoint and reference hashes, solver profile `g1-4x5`, physical GPU 2 for phase zero, and physical GPUs `1/5/6/7` for the four remaining phases. Store the complete phase-grid output under the E004 seed directory.

- [ ] **Step 3: Apply the preregistered gate mechanically**

Select `horizon24-advances` only when minimum survival is at least 19 and median survival is at least 48. Otherwise select `horizon24-finite-no-robust-gain` and stop this branch; do not extend training based on scalar reward alone.

- [ ] **Step 4: Curate, validate, build, test, commit, and push**

Run: `uv run python tools/researchctl.py validate && uv run python tools/researchctl.py build-vault && uv run python -m pytest tests/test_registry.py -q -k 'not links_resolve' && git diff --check`

Update E004 and current state from the deterministic evidence, commit only owned registry/generated files, and push the lab `main` branch.
