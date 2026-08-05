# G1 Fixed-Mass Differentiable Adaptation Design

## Objective

Test whether the working frozen-RMR plus bounded-residual SHAC method can use
MJX physics gradients to adapt Unitree G1 motion tracking to a fixed,
sim-to-real-relevant dynamics mismatch. This is the next gate after nominal
training preserved strict visible walking across two seeds but improved nominal
metrics in only one seed.

## Alternatives Considered

### Fixed body mass and inertia scale

Multiply every non-world body mass and diagonal inertia by one scalar before
converting the MuJoCo model to MJX. This changes the plant while preserving
mass distribution, center-of-mass locations, geometry, contacts, controller,
observations, rewards, and reference motion. It is deterministic,
differentiable, physically coherent, and directly represents a mass-model
calibration error.

### Fixed actuator-authority scale

Scale the applied PD torque or its gains. This is easy to implement and relevant
to motor-model error, but it changes the low-level controller boundary rather
than only the simulated plant. A positive result would therefore be less clean
evidence that physics-gradient adaptation handles a dynamics mismatch.

### Fixed ground-friction scale

Scale sliding, torsional, and rolling friction. This is relevant to transfer,
but contact friction is discontinuity-sensitive and can make a first adaptation
result harder to interpret. It is a useful later robustness test after the
method passes a smoother model shift.

## Selected Design

Use a fixed `1.15` non-world body mass-and-inertia scale. Fifteen percent is
large enough to create plausible adaptation headroom without intentionally
moving the source controller outside its walking basin. The scalar applies to
`body_mass[1:]` and `body_inertia[1:]`; body zero is the MuJoCo world and remains
unchanged. Geometry, body positions, centers of mass, gravity, friction,
controller gains, torque limits, actions, observations, rewards, terminations,
solver budget, and reference motion remain unchanged.

The existing `mass_range` SHAC interface will carry the setting so the training
algorithm does not gain a second model-configuration path. G1 accepts only a
fixed pair `(scale, scale)` because per-episode model randomization is outside
this experiment. Nominal `(1.0, 1.0)` behavior remains byte-compatible at the
configuration boundary.

## Interfaces

`G1TrackingEnv.__init__` accepts `mass_range: tuple[float, float] =
(1.0, 1.0)`. It rejects nonfinite, nonpositive, unequal, or wrong-length
values. It exposes `body_mass_scale` and applies the scale to the MuJoCo model
before `mjx.put_model`.

`tools/run_g1_tracking_rmr50_shac.py` accepts
`--body-mass-scale FLOAT`, defaults to `1.0`, validates positivity, and maps it
to `mass_range=(value, value)` in the existing SHAC arguments.

`make_evaluation_env`, `evaluate_g1_tracking.py`, and
`compare_g1_tracking_residual.py` accept the same fixed scalar. Evaluation
summaries record it so nominal and shifted artifacts cannot be confused.

## Scientific Contract

Train a new zero-initialized residual directly through the fixed `1.15` MJX
model. Preserve the authoritative nominal contract:

- frozen float32 RMR source controller;
- residual action scale `0.1`;
- seed `0`;
- `256` environments;
- horizon `12`;
- `196,608` environment steps;
- actor learning rate `1e-4`;
- zero action noise;
- zero actor bootstrap;
- solver iterations `4` and line-search iterations `5`;
- fixed checkpoint selection at step `64,512`.

No mass sweep, checkpoint search, or post-hoc seed selection is allowed.

Compare the shifted frozen source and shifted adapted residual in one process
across reference phases `0`, `30`, `60`, and `90`. The primary success outcome
requires:

- no additional residual terminal failures;
- aggregate residual-minus-source reward at least `+0.001`;
- at least four of six aggregate tracking-error deltas below zero;
- reward improvement in at least three of four phases;
- all training gradients and selected-checkpoint leaves finite.

A preservation-only outcome has no additional terminals but misses the
improvement thresholds. A degradation outcome has an additional terminal or
aggregate reward below `-0.0001`. Missing or inconsistent evidence is an
invalid execution, not a scientific failure.

Render the exact step-64,512 shifted residual for a replay-free 60-step
phase-zero rollout. The trajectory, summary, video, and first/middle/final
montage must agree. A visible adaptation claim requires the robot to remain
upright and follow the paired reference without a material gait defect.

## Evidence And Failure Handling

The experiment is preregistered in `diffsim2real-lab` after the implementation
commit is pinned and before any scientific GPU execution. Execution uses
`researchctl runexp`; every result, artifact hash, verdict, hypothesis edge,
finding, and Obsidian projection is curated through the canonical registry.

If the fixed checkpoint preserves walking but does not improve shifted metrics,
the next action is objective/gradient localization on this exact shift, not a
mass or checkpoint sweep. If the source itself leaves the walking basin, the
result remains scientifically useful but the next registered shift must be
smaller before testing fine adaptation. Collocation remains deferred until this
RL adaptation gate is resolved.

## Testing

Focused tests prove:

- nominal construction leaves masses and inertias unchanged;
- scale `1.15` changes all and only non-world masses and inertias by `1.15`;
- invalid and nonfixed ranges are rejected;
- the runner maps `body_mass_scale` to the existing fixed `mass_range`;
- evaluation construction receives the exact scalar;
- summaries record the scalar.

Then run the existing G1 environment, runner, evaluator, residual comparison,
and checkpoint archival suites with x64 enabled. No full repository suite is
required because the change is confined to the G1 model-construction and tool
boundaries.
