# G1 Actuator-Authority Shift Design

## Goal

Add one physically interpretable, fixed actuator-authority shift to the
validated G1 tracking task and screen its severity before spending compute on
differentiable adaptation. The screen must select a shifted model that
materially degrades the pretrained source actor while preserving enough
closed-loop walking to make recovery measurable.

## Alternatives

1. **Scale the torque ceiling (selected).** Multiply every RMR controller effort
   limit by one fixed positive scalar. This changes only available actuator
   authority, leaves PD gains and the tracking MDP unchanged, and is directly
   relevant to saturation under hardware mismatch.
2. **Scale PD gains.** This changes both response dynamics and requested torque.
   It is useful later, but it confounds controller calibration with actuator
   authority and is therefore not the first discriminator.
3. **Change mass or friction again.** The existing fixed 1.15 mass-and-inertia
   result barely changes source reward, so it cannot establish material
   recovery. Friction is contact-sensitive and less isolated than a torque
   ceiling.

## Environment Contract

`G1TrackingEnv` accepts `effort_limit_scale: float = 1.0`. The value must be
finite and strictly positive. The environment records it as
`env.effort_limit_scale` and uses
`controller.effort_limit * effort_limit_scale` as the symmetric torque clip in
every MJX substep. A value of `1.0` must preserve current behavior exactly.

The native RMR runner and replay-free evaluator expose the same scalar through
`--effort-limit-scale`. Training hparams and evaluation summaries must record
the effective scale. No PD gain, action scale, observation, reward, termination,
solver, reference, mass, or timing change is included.

## Same-Process Severity Screen

A dedicated comparison tool loads the pretrained RMR actor once and evaluates
the registered ordered scales `1.0`, `0.8`, `0.7`, `0.6`, and `0.5` in one
process. Each scale uses seed zero, phases 0/30/60/90, the strict 60-step suffix
protocol, x64, and solver budget 4/5. Scale `1.0` is the shared nominal
baseline.

For every shifted scale, the strict JSON artifact records:

- all four per-phase rollout summaries;
- aggregate terminal count and reward;
- the six aggregate tracking errors;
- shifted-minus-nominal deltas; and
- whether the scale passes the selection gate.

The selected scale is the first input-order shifted scale that:

1. adds no terminal relative to nominal;
2. decreases aggregate reward by at least `0.001`; and
3. worsens at least four of the six aggregate tracking errors.

If no scale passes, selection is null. The screen does not silently choose a
more severe or terminal model. JSON publication is atomic and rejects
non-finite values.

## Scientific Decision

- **Least-severe discriminative shift selected:** preregister full-policy
  differentiable fine-tuning from the pretrained RMR actor on that exact scale.
  Compare adapted and unadapted actors on the shifted model in one process.
- **No nonterminal shift qualifies:** stop this torque-ceiling branch and
  choose a separately registered dynamics mismatch; do not train against an
  uninformative shift.
- **Invalid execution:** repair only the transport, evaluator, or evidence
  boundary and rerun under a new experiment ID.

This screen is diagnostic evidence about task headroom. It is not itself
evidence that differentiable training adapts, transfers to hardware, or learns
walking from scratch.

## Testing

Focused tests cover positive-finite validation, exact effort-limit scaling,
neutral-scale preservation, runner/evaluator CLI transport, strict gate logic,
earliest selection, finite artifact rejection, and atomic output. Existing G1
environment, evaluator, runner, full-policy comparison, and controller tests
must remain green.
