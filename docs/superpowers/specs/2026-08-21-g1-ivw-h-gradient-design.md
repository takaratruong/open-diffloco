# G1 IVW-H Composite Gradient Design

## Decision

Run one frozen gradient discriminator before changing G1 training. Implement the
published IVW-H estimator at the action-distribution boundary: retain the exact
ordinary H24 pathwise gradient, independently compute a likelihood-ratio
gradient from the same noisy trajectories, estimate both variances per phase,
step, and action dimension, and combine them by inverse-variance weighting.

This is not full PIPPS Total Propagation. PIPPS propagates probability
distributions through a stochastic dynamics model, which the deterministic MJX
environment does not provide. IVW-H is the directly applicable, simpler
successor: it was designed for differentiable robotics simulators and performs
the composite operation at policy action nodes before one policy VJP.

Do not train until the frozen discriminator proves that the composite direction
is more repeatable across independent noise tapes and more consistent across
MJX solver profiles than the ordinary pathwise gradient at both fresh and E023
actor boundaries.

## Question

Does phase-wise, step-wise, action-wise inverse-variance fusion of pathwise and
score-function gradients produce a more stable G1 H24 policy direction without
deleting the contact credit that E005 proved necessary?

The answer changes the next branch:

- robust fusion authorizes one short, otherwise exact fresh E023-recipe IVW-H
  training run;
- neutral fusion rejects composite estimation as the next bottleneck and moves
  to a compliant-contact or objective-curriculum treatment;
- destructive fusion rejects this implementation without sweeping mixture
  weights, baselines, or noise magnitudes.

## Alternatives Considered

### Full PIPPS Total Propagation

Model a Gaussian state distribution at every transition and combine all
possible pathwise depths during a custom backward pass. This is theoretically
stronger, but requires a stochastic dynamics density, state-distribution
fitting, and nonstandard message passing that MJX/SHAC does not expose. It would
change several causal variables before we know whether a score component helps.
Rejected for this discriminator.

### Parameter-space REINFORCE plus pathwise gradient

Compute one score-function gradient and one pathwise parameter gradient for the
whole H24 trajectory, then blend their aggregate vectors. This is simple but
throws away the useful fact that contact instability is localized in time and
joint space. It also estimates variance over roughly 460,000 parameters rather
than the 29 action dimensions. Retained only as a diagnostic summary, not the
selected estimator.

### IVW-H at action nodes — selected

For each phase group, time step, and action coordinate, compute pathwise and
likelihood-ratio gradient samples across 24 actors, fuse them using their
empirical variances, then perform a single VJP through the actor outputs. This
matches the published differentiable-robotics algorithm, fits the existing
five-phase SHAC/CAGrad structure, and adds no simulator calls beyond the paired
frozen noise tapes required for validation.

## Exact Estimator

Use the existing fixed five-phase population: 24 replicas at each start phase
`0/25/50/75/100`. Evaluate two independently generated, preregistered float32
H24 Gaussian epsilon tapes. For actor boundary `b`, solver `s`, tape `k`, and
environment `i`, run the exact E023 observation, normalization, actor, action
noise, action boundary, reward, reset, and discount contract.

Let the pre-boundary stochastic action be

`a[t] = mu_theta(stop(obs[t])) + sigma_b * epsilon[k,i,t]`.

E023 has `clip_sampled_actor_actions=false`, so no clipping derivative is
introduced. The fresh boundary uses the registered scalar start standard
deviation `1.0`; E023 uses its exact 29-dimensional learned end standard
deviation. Sigma is fixed and is not trained.

For the pathwise sample, introduce an additive action-node perturbation
`delta[t]`, execute the ordinary differentiable rollout with `a[t] + delta[t]`,
and evaluate at `delta=0`:

`g1[i,t,a] = d(-return_i / H) / d delta[i,t,a]`.

The action-node construction must reproduce the existing direct
parameter-space pathwise gradient after a VJP through all cached actor means.
This equality is a validity gate, not a scientific outcome.

For the score sample, stop gradients through sampled actions, states, rewards,
and returns. Compute the discounted reward-to-go from each transition until
the first episode boundary or H24 end. For each fixed start-phase group and
time step, subtract the leave-one-out mean reward-to-go from the other 23
replicas. This action-independent baseline preserves the score estimator while
avoiding the known E023 critic-quality confound. For minimization of negative
return:

`g0[i,t,a] = advantage_loo[i,t] / H * (mu[i,t,a] - stop(a[i,t,a])) / sigma[a]^2`.

Do not whiten, clip, or normalize advantages in the discriminator. Require
strictly positive finite sigma and finite rewards, returns, and gradient
samples.

Within each start-phase group, compute sample variances across the 24 replicas:

`v0[p,t,a] = variance_i(g0[i,t,a])`

`v1[p,t,a] = variance_i(g1[i,t,a])`

and the published IVW-H weight:

`alpha[p,t,a] = v0 / (v0 + v1)`.

If both variances are exactly zero, set `alpha=0`; the fused sample is already
zero at that coordinate. Otherwise do not add a tunable epsilon or variance
floor. Fuse

`G[i,t,a] = alpha[p,t,a] * g1[i,t,a] + (1-alpha[p,t,a]) * g0[i,t,a]`.

Push each stopped action-gradient sample through the actor using cached,
stopped observations, aggregate finite per-environment parameter gradients
with the unchanged norm-1 and five-bin CAGrad reducer, and report ordinary,
score, and IVW-H directions separately.

## Frozen Discriminator

Use the exact E005 inputs and runtime assets:

- fresh seed-zero E023 actor initialization and E023 final actor/normalizer;
- raw 125-frame walking reference;
- `g1-4x5` and `diagnostic-10x20` solver profiles;
- five phase groups, H24, exact action standard deviations, no sampled-action
  clipping, and no critic bootstrap;
- seed zero for the first tape and one fixed distinct registered seed for the
  second tape.

This produces eight forward populations per actor pair: two actors by two
solvers by two tapes. Each population yields all three estimators from the same
trajectory. Persist initial states, epsilon, observations, sampled actions,
rewards, done/terminal flags, reward-to-go, score advantages, action-node
gradient samples, IVW weights, finite masks, aggregate/phase directions, and
all provenance hashes. Large parameter-gradient samples may be reduced to
norms and aggregate/phase vectors after validation; the action-space evidence
must remain complete.

## Gates And Outcome Map

Execution is valid only if:

- code, checkpoint, hparams, model, controller, reference, solver, actor, and
  both noise tapes match the registered provenance;
- initial states are bit-identical across solver arms;
- every forward and action-gradient artifact is finite;
- every phase has at least 16 finite contributors for every actor, solver,
  tape, and estimator;
- action-node pathwise VJP agrees with the direct ordinary parameter gradient
  at cosine at least `0.999` and norm ratio in `[0.999, 1.001]`;
- every aggregate and phase direction is finite and nonzero;
- manifest-last artifact hashes revalidate after publication.

For each actor, measure two reliability axes for ordinary pathwise and IVW-H:

1. solver cosine: `g1-4x5` versus `diagnostic-10x20` under each tape;
2. tape cosine: tape zero versus tape one under each solver.

Also measure the nominal `g1-4x5` IVW-H-versus-pathwise cosine and norm ratio.

- `ivw-h-robust`: at both actor boundaries, IVW-H improves the mean solver
  cosine and mean tape cosine by at least `0.05`, does not worsen any of the
  five phase cosines on either axis by more than `0.05`, retains nominal
  pathwise cosine at least `0.5`, and retains norm ratio in `[0.25, 4.0]`.
  Authorize one separately registered 16-update fresh IVW-H training gate.
- `ivw-h-neutral`: all validity and retained-signal gates pass, but either
  reliability improvement is below `0.05`. Reject IVW-H as the next causal
  treatment without changing thresholds.
- `ivw-h-destructive`: validity passes, but retained cosine/norm fails or any
  phase reliability regresses by more than `0.05`. Reject the treatment.
- `invalid-execution`: any provenance, reconstruction, finite-contributor,
  direct-VJP parity, nonzero-direction, or artifact gate fails. Repair only the
  execution boundary.

Outcome precedence is invalid, destructive, robust, then neutral. These are
diagnostic robustness thresholds, not learning-performance parameters.

## Implementation Boundaries

Add one small pure IVW-H module containing discounted reward-to-go,
leave-one-out baselines, variance weights, fusion, and validation. Add one
experiment-specific evaluator that reuses the E005 reconstruction and CAGrad
interfaces. Do not modify `train()` in this stage. Do not add DDCG, GAE, a new
critic, learned sigma, a scheduler, or a generic estimator framework.

If and only if the frozen result is `ivw-h-robust`, the subsequent training
implementation will expose one default-off IVW-H flag and reuse the same pure
module. It will be designed and registered separately.

## Tests

Test exact episodic discounted reward-to-go, leave-one-out independence,
Gaussian score sign and scale, zero-variance handling, known inverse-variance
weights, phase-local variance isolation, fusion, nonfinite rejection, and
parameter VJP parity on a synthetic recurrent rollout. Test fixed population
and distinct tape identity, exact E005 provenance, classification precedence,
finite phase counts, atomic artifact publication, recursive hashes, and tamper
rejection. Run a compiled one-phase/eight-replica smoke before the scientific
GPU diagnostic.

## Stop Rule

Stop after the eight registered forward populations and deterministic
classification. Do not train, sweep mixture weights, add a variance floor,
change the score baseline, alter noise, or add DDCG from this experiment. A
short training treatment is allowed only under `ivw-h-robust` and must be a new
experiment with unchanged replay-free evaluation, video, learning curves, and
action diagnostics.
