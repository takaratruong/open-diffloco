# G1 Root-Recovery Reset Design

## Question

Can broader but bounded root-state reset coverage convert E012's narrow
assistance transfer into robust zero-wrench survival?

## Alternatives

1. Double `reference_reset_noise_scale`. This is smallest, but doubles root and
   joint noise together and cannot isolate root recovery.
2. Add a root-focused mixture. Preserve the current reset distribution in half
   the environments; in the other half, double only root pose and velocity
   perturbations while leaving joint perturbations unchanged. This is the
   selected design.
3. Reset from pre-failure carried states. This targets observed failures but
   repeats E010's bank/context complexity and does not guarantee recoverability.

## Causal Boundary

Resume the exact E012 final checkpoint at transition 1,720,320 and repeat
E017's 64-update, H12, effective-512, five-bin CAGrad continuation. Preserve
the frozen parent, residual actor and Adam state, critic, domain randomization,
reward, termination, action noise, phase sampling, solver, and exact-zero
assistance schedule.

Change only the reset distribution. The baseline cohort remains byte-for-byte
the existing `reference_reset_noise_scale=1.0` distribution. An independently
sampled 50-percent recovery cohort uses twice the existing root pose and root
velocity limits while retaining the existing joint limit:

- root translation: `±[0.04, 0.04, 0.01]` metres;
- root Euler rotation: `±[0.2, 0.2, 0.2]` radians;
- root linear velocity: `±[0.5, 0.5, 0.2]` metres/second;
- root angular velocity: `±[0.52, 0.52, 0.78]` radians/second;
- joint position: unchanged `±0.05` radians.

No recovery flag, perturbation value, assistance signal, or privileged state is
added to the actor observation. Evaluation remains exact-state, zero-wrench,
nominal replay-free physics.

## Implementation Boundary

Add two default-off environment/trainer settings:

- `reference_root_reset_noise_multiplier`, default `1.0`;
- `reference_root_reset_noise_probability`, default `0.0`.

The recovery cohort is sampled once per reset. The multiplier applies only to
the six root pose and six root velocity limits; the joint distribution and all
legacy default paths remain unchanged. Resume activation requires one explicit
`allow_resume_reference_root_reset_noise_change` authority flag and is recorded
in hparams.

## Evidence And Decision

Archive eight checkpoints and evaluate all from phases `0/100/200/300/400`
with exactly zero wrench. Select by minimum, median, mean, then earliest.

- Broad root-recovery gain: treatment beats E017's `50/67/73.0` control key,
  beats E012's `51/66/68.6` robust key, and does not regress any component of
  E012's selected `66/61/89/51/76` vector.
- Root-recovery tradeoff: treatment beats E017 but misses E012's key or a phase
  floor.
- Finite no gain: treatment does not beat E017.
- Invalid: any resume, cohort, bounds, zero-wrench, frozen-state, gradient,
  evaluation, provenance, or artifact gate fails.

Only broad gain continues this reset recipe. A tradeoff or no-gain result stops
root-noise tuning and returns to the objective/optimization branch.

## Tests

Unit tests verify legacy bit-equivalence at defaults, unchanged joint noise,
the exact root bounds for both cohorts, deterministic reset sampling, resume
authority, hparams persistence, runner exact-delta behavior, and exact-zero
post-training telemetry. Focused integration tests execute reset and one
trainer construction path without launching the registered GPU experiment.
