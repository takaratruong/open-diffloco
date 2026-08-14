# G1 MJX PPO Positive Control Design

## Question

Can ordinary score-function PPO learn the exact G1 reference-tracking task that
the differentiable SHAC experiments use, without changing the simulator,
controller, observations, actions, resets, reward, or termination?

This is a positive control for the learning problem.  An Isaac Lab PPO result
does not answer it because simulator and controller details differ.

## Causal boundary

The environment is the existing
`g1_tracking_rmr_50hz_action_parity` MJX environment with solver profile
`g1-4x5`, the 125-frame walk reference, one-frame 154-dimensional actor
observation, 286-dimensional critic observation, 29-dimensional source-order
reference-delta action, full residual scale, and nominal physical parameters.
The environment retains its exact uniform reference-state reset and its current
reward and termination.  Observation noise, reset perturbations, domain
randomization, pushes, assistance, and differentiable simulator gradients are
disabled.

Only the policy optimizer changes: standard clipped PPO with GAE and a learned
diagonal Gaussian action distribution replaces SHAC/CAGrad.  Simulator states
and rollout data are stopped before the PPO update, so no physics gradient can
enter the policy loss.

## Architecture

Add a small PPO package beside SHAC rather than modifying SHAC.  It reuses the
existing RMR actor/critic parameter types, observation normalizers, environment
factory, checkpoint conventions, solver context, and deterministic evaluator.
The rollout collector vmaps the existing environment, stores pre-reset
bootstrap observations and true terminal flags already supplied by
`EnvState.info`, and computes GAE with terminal versus reference-end truncation
handled separately.

The trainer uses a learned log standard deviation, clipped likelihood ratio,
clipped value loss, entropy regularization, advantage normalization, minibatch
epochs, global gradient clipping, and Adam.  Training telemetry includes return,
episode length, policy/value losses, entropy, approximate KL, clip fraction,
gradient norms, action-mean RMS, action-standard-deviation RMS, and finite-state
validity.  Checkpoints bind code, model, controller, reference, solver, seed,
and all PPO hyperparameters.

## Initial recipe

- 4,096 MJX environments on one GPU.
- 24 environment steps per rollout.
- 98,304 transitions per PPO iteration.
- 4 minibatch epochs, 8 minibatches, GAE lambda 0.95, discount 0.99.
- Actor and critic topology: existing RMR MLP topology.
- Learned diagonal Gaussian standard deviation, initialized to 0.2.
- Adam learning rate 3e-4 with global norm clipping at 1.0.
- PPO clip epsilon 0.2, value coefficient 0.5, entropy coefficient 0.0.
- First gate at 32 iterations; extend only if finite learning and survival trend
  justify it.  Maximum initial budget is 512 iterations.

The 0.2 initial standard deviation matches the user-approved exploration scale.
Because it is learned by PPO, it may decrease as tracking competence emerges.

## Evidence and decision

Every selected checkpoint is evaluated replay-free and deterministically from
phases 0, 25, 50, 75, and 100.  Required evidence is a trajectory, metrics,
video, and frame montage using the same evaluator as the DiffSim experiments.

- **PPO competent:** all five suffixes reach the reference endpoint without a
  true terminal.  The environment and reward are learnable; compare PPO and
  SHAC gradient/update behavior from matched states.
- **PPO improves but is incomplete:** minimum and median survival improve
  monotonically enough to justify the preregistered extension, then re-evaluate.
- **PPO fails with finite optimization:** inspect reward/termination and policy
  diagnostics before blaming differentiable gradients.
- **Invalid execution:** any provenance mismatch, nonfinite rollout/update,
  wrong action contract, wrong reset distribution, or missing evidence.

## Boundaries

This experiment does not establish differentiable-policy training or
sim-to-real transfer.  It does not tune rewards, terminations, gains, resets,
or randomization.  The Isaac PPO checkpoint may be evaluated as a separate
transfer initializer later, but it is not used in this from-scratch control.
