# Task 1 Report: Pure bounded torso-wrench controller

## Scope

Implemented only the frozen E008 evaluation-only torso-wrench core in the
isolated `research/g1-backpack-oracle-20260811` worktree. No training code,
SHAC code, E010 artifacts, lab registry, GPU, checkpoint, or reference data was
modified or consumed.

## Files

- `src/evaluation/g1_torso_wrench_oracle.py`
  - Typed immutable physical parameters using the environment's
    `nominal_total_mass` and `base_gravity_mag`.
  - Name-based `torso_link` lookup, shortest WXYZ quaternion rotation vector,
    current-yaw-frame critical 2 Hz PD, world-frame conversion, stable norm
    caps, exact disabled zero, and targeted MuJoCo force/torque-row overwrite.
- `tests/test_g1_torso_wrench_oracle.py`
  - Seven unit tests covering the required controller behavior.

## RED evidence

Before production code existed, ran:

```bash
conda run -n diffsim python -m pytest tests/test_g1_torso_wrench_oracle.py -q
```

Result: `6 failed`; every test failed with the expected
`ModuleNotFoundError: No module named 'src.evaluation'`.

## GREEN evidence

After the minimal implementation, the same command initially isolated a test
fixture issue (the yaw-frame position-only case unintentionally requested an
orientation correction). The fixture was corrected to supply the same actual
and reference yaw quaternion. The final focused run was:

```text
.......                                                                  [100%]
7 passed in 4.72s
```

The seven tests cover:

1. exact zero six-vector for scale zero;
2. quaternion sign invariance and shortest rotation;
3. yaw-frame error calculation with world-frame wrench output;
4. force channels `:3` and torque channels `3:` on the name-resolved torso row;
5. environment-derived mass/gravity and name-derived torso ID;
6. replacement of a stale torso wrench on the next policy step; and
7. finite output plus force and torque norm caps under extreme finite errors.

Final focused verification also included the existing G1 tracking-controller
tests:

```text
..........                                                               [100%]
10 passed in 4.75s
```

## Static verification

Ran successfully:

```bash
conda run -n diffsim ruff check src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
conda run -n diffsim python -m py_compile src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
git diff --check
```

Ruff reported `All checks passed!`; `py_compile` and `git diff --check` exited
zero.

## Self-review

Reviewed the full untracked diff against the frozen task brief and approved
design. The implementation is confined to a new evaluation module and its
tests; it neither imports nor changes training or SHAC paths. It resolves
`torso_link` via MuJoCo name lookup, reads mass/gravity from the environment,
uses the specified 2 Hz critical-damping gains and `0.3 m` inertia/cap values,
and emits world force then torque in one six-channel row. The norm calculation
is scale-stable for extreme finite vectors and sanitizes nonfinite intermediates
to retain finite output. The existing focused tracking-controller baseline also
passed before the change: `3 passed in 0.37s`.

## Concerns

None for Task 1. The later evaluator must call `write_torso_wrench` before each
unchanged environment step, including the disabled path, so the target row is
overwritten every policy step as designed.
