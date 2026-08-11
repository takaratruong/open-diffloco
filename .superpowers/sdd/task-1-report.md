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

## Follow-up: float-limit PD overflow fix

### Scope and root cause

This follow-up changes only the Task 1 controller and its unit tests. A finite
`float32` position error near `1e38` overflowed while forming
`Kp * position_error`; `_finite_vector` then replaced the resulting infinity
with zero before the norm cap could preserve its direction. Direct reproduction
before the fix was:

```bash
conda run -n diffsim python -c '... reference_position=jp.array([1e38, 0.0, 0.0], dtype=jp.float32) ...'
```

```text
wrench [0. 0. 0. 0. 0. 0.]
force norm 0.0 force cap 10.0
```

The replacement first normalizes the finite proportional and derivative errors
by their shared maximum magnitude, forms the bounded-gain demand at that safe
scale, then restores only the capped magnitude. Thus a request that must
saturate never forms the overflowing raw force or torque.

### RED and GREEN evidence

Added `test_near_float_limit_force_error_remains_directionally_capped`, which
requires the `float32 [1e38, -1e38, 0]` request to remain finite, reach the
force cap, and keep its normalized direction. Before the fix:

```text
1 failed, 7 passed in 5.03s
```

The first normalization attempt exposed an operator-ordering issue
(`gain * error / scale` still overflowed before division). Evidence was:

```text
unsafe ordering [ inf -inf   0.]
safe ordering [ 157.91367 -157.91367    0.     ]
bounded [0. 0. 0.]
```

The final implementation divides before multiplying by the gain. During
self-review, an additional cancellation-residual regression was added to prove
that an uncapped low-norm PD demand is not double-scaled. Its RED run was:

```text
1 failed, 8 passed in 5.03s
```

After the one-line normalization-denominator correction, the oracle-only suite
passed:

```text
.........                                                                [100%]
9 passed in 4.80s
```

### Final verification

Ran exactly:

```bash
conda run -n diffsim python -m pytest tests/test_g1_torso_wrench_oracle.py tests/test_g1_tracking_controller.py -q
conda run -n diffsim ruff check src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
conda run -n diffsim python -m py_compile src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
git diff --check
```

Exact relevant output:

```text
............                                                             [100%]
12 passed in 4.77s

All checks passed!
```

`py_compile` and `git diff --check` exited zero.

### Follow-up self-review and concerns

The response helper is still pure and JAX-compatible, caps force and torque in
the yaw frame before returning them to world coordinates, and does not change
the frozen gain, cap, body-resolution, or rollout interfaces. The new residual
test guards against the only extra scaling edge case introduced by the
normalization. No concerns remain for this scoped follow-up.

## Follow-up: float-limit yaw-frame rotation fix

### Scope and root cause

This follow-up remains limited to the Task 1 oracle, its tests, and this
report. The earlier PD normalization happened after the initial world-to-yaw
rotation. With a valid 45-degree yaw and finite `float32` world error
`[3e38, 3e38, 0]`, the quaternion dot/cross intermediates overflowed and
corrupted the direction before the PD cap was reached. Direct reproduction
before this fix returned a full-cap but incorrect direction:

```text
wrench [-7.0710683  7.0710697  0.         0.         0.         0.       ]
force norm 10.000001 force cap 10.0
direction [-0.70710677  0.70710695  0.        ]
```

### RED and GREEN evidence

Added `test_float_limit_yaw_rotation_preserves_capped_world_direction`. It
requires the same 45-degree yaw and finite `[3e38, 3e38, 0]` position request
to yield a finite full-cap world force in `[+1, +1, 0] / sqrt(2)` direction,
while the actual/reference orientation is equal to isolate translation. The
RED run was:

```text
1 failed, 9 passed in 4.62s
```

The fix represents each initial reference-minus-actual vector as a normalized
finite difference plus a separate finite magnitude, rotates only that bounded
vector into the yaw frame, and passes both parts to the bounded PD response.
The capped yaw-frame force is small before its final world rotation. This also
handles finite opposite-sign reference/actual inputs without overflowing their
subtraction. The oracle-only GREEN run was:

```text
..........                                                               [100%]
10 passed in 5.08s
```

### Final verification

Ran exactly:

```bash
conda run -n diffsim python -m pytest tests/test_g1_torso_wrench_oracle.py tests/test_g1_tracking_controller.py -q
conda run -n diffsim ruff check src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
conda run -n diffsim python -m py_compile src/evaluation/g1_torso_wrench_oracle.py tests/test_g1_torso_wrench_oracle.py
git diff --check
```

Exact relevant output:

```text
.............                                                            [100%]
13 passed in 5.01s

All checks passed!
```

`py_compile` and `git diff --check` exited zero.

### Follow-up self-review and concerns

The fix affects only the pre-PD yaw-frame error representation. It leaves the
fixed gains, body resolution, world-frame force/torque order, cap values,
disabled-zero path, and rollout interface unchanged. The full oracle suite
passes the previous nominal, direct float-limit, and cancellation-residual
tests as well as the new non-identity-yaw regression. No concerns remain.
