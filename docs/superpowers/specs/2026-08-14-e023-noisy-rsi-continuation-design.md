# E023 Noisy-RSI Continuation Design

## Decision

Resume the retained final E023 `TrainState` for 64 additional H24 SHAC
updates and change only the reset-state distribution. Replace uniform exact
reference-state initialization with uniform reference-phase initialization
inside the existing standard RMR perturbation envelope at
`reference_reset_noise_scale=1.0`. Preserve the actor and critic parameters,
optimizer and normalizer state, five-bin CAGrad, zero terminal bootstrap,
plant, observations, rewards, termination, seed, solver, and endpoint
model-999 action-noise vector. Domain randomization, observation noise,
pushes, assistance, adaptive phase sampling, and carried-state resets remain
disabled.

The reset envelope independently samples joint offsets up to plus or minus
0.05 radians; root XY/Z offsets up to plus or minus 0.02/0.005 metres; root
orientation offsets up to plus or minus 0.1 radians; root linear-velocity
offsets up to plus or minus 0.25/0.25/0.1 metres per second; and root
angular-velocity offsets up to plus or minus 0.26/0.26/0.39 radians per
second. The actor is not told the sampled perturbation.

## Alternatives considered

Joint-only reset noise is cheaper conceptually but does not directly train
recovery from the accumulated root-position, orientation, and velocity drift
observed before E023's failed catch step. A carried-state reset bank is more
on-policy but introduces a policy-specific empirical distribution and actor
history artifact as a second causal mechanism. The standard full RMR envelope
is therefore the smallest existing treatment that targets the observed
balance-recovery deficit.

## Evidence and stop condition

E024 shows that unchanged optimization redistributes competence rather than
solving the walk: additional-update vectors 16/32/64/128 are
69/99/67/49/24, 105/99/74/49/24, 74/99/74/49/24, and 102/65/74/49/24 versus
retained E023 at 116/99/67/49/24. This localizes the next uncertainty to the
training state distribution.

Archive every eight additional updates and deterministically evaluate updates
8, 16, 32, and 64 at phases 0, 25, 50, 75, and 100. A checkpoint is eligible
only if it preserves E023's 116/99/67/49/24 vector componentwise. Exact
124/99/74/49/24 survival solves the short walk. If no checkpoint is eligible,
stop noisy-RSI tuning and test history-faithful carried-state resets rather
than sweeping noise magnitude.

## Safety and evidence

Pin the retained E023 checkpoint and sibling hparams hashes, clean code
commit, model, controller, reference, solver, seed, environment, and every
continuation kwarg. Explicitly authorize only the resume-time
`reference_reset_noise_scale` change and require the persisted hparams to
record it. Before GPU execution, test that scale zero is exact-state
compatible and scale one stays inside every registered bound. Fail closed on
resume-state drift, missing archives, nonfinite state or telemetry,
unoccupied or invalid CAGrad bins, altered action noise, nonzero bootstrap,
or unexpected randomization. Publish extended learning curves and only the
selected clean and training-distribution rollouts through the experiment
registry and Obsidian projection.
