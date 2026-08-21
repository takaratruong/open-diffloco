# G1 Contact-Topology Gradient Truncation Design

## Decision

Run one frozen-gradient discriminator before another full G1 training run.
Preserve the raw 125-frame walking reference, E023 action/noise/observation/reward
contract, H24 forward rollout, effective phase coverage, and fixed MJX solver.
Change only the actor derivative at a physical body-pair contact change:
the transition reward and returned environment state keep their exact forward
values but are detached from the preceding trajectory. Later actions and later
rewards remain differentiable with respect to the shared actor parameters.

The event is the set of active unordered model-body contact pairs. Multiple
contact points and collision geometries between the same two bodies collapse
to one bit. A transition is an event when any pair bit changes. Terminal
auto-reset changes are not contact events. This rule has no calibrated force
threshold. A completed 120-rollout preflight found zero left/right foot-support
changes at every fixed phase group, so the narrower foot-only detector was
falsified before a scientific or training launch.

Do not launch full training unless the frozen diagnostic shows that this
surrogate is less sensitive to solver convergence while retaining a substantial
and directionally related actor signal.

## Question

Are contact-topology transitions a material source of unreliable pathwise
actor gradients in fresh raw-reference G1 walking, and does severing only those
transition derivatives produce a more solver-robust H24 direction without
discarding the useful smooth-segment gradient?

The answer changes the next branch:

- robust truncation authorizes one otherwise exact fresh E023-recipe training
  run;
- neutral truncation rejects this mechanism and returns to a hybrid
  pathwise/score-function or behavioral accept/reject treatment;
- destructive truncation rejects topology barriers without tuning the event
  rule.

## Alternatives Considered

### Force or stiffness threshold

Detach when root contact stiffness or foot normal force exceeds a threshold.
This is simple, but E003 showed that a threshold calibrated on competent E023
never activated on the weak fresh actor. A second threshold would be another
performance sweep rather than a causal test. Rejected.

### Solver-disagreement-triggered detachment

Run two solvers online and detach only where their transition Jacobians differ.
This is the strongest local reliability oracle, but it doubles training
dynamics, adds a second model state to every update, and cannot be deployed as
the final learning rule. Retain the second solver only as a frozen diagnostic.

### Grouped body-pair contact-topology truncation — selected

Contact-set changes are exactly where rigid contact dynamics change branch.
Grouping by unordered body pair avoids triggering on duplicate contact points
while covering ground-foot, ground-knee, ground-body, and self-contact changes.
The rule is phase-independent, force-scale-independent, and reusable on longer
motions. It leaves intervals with unchanged active body pairs fully pathwise
differentiable.

## Gradient Semantics

At every H24 step:

1. Compute the actor mean, reparameterized RMR action noise, action boundary,
   and `env.step` exactly as the retained E023 recipe.
2. Compute the pre-step and post-step grouped body-pair signatures from the
   physical MJX data. Stop gradients through both signatures.
3. Set `contact_event` when a body-pair bit changes and the transition is not an
   environment reset.
4. When enabled and `contact_event` is true, apply a tree-wide gradient barrier
   to the candidate transition state before its reward is read and before it is
   carried to the next step. The forward state, reward, done, terminal,
   observations, phase, and action must remain bit-identical to the ordinary
   rollout.
5. Continue the scan through all 24 transitions. Later policy evaluations
   still depend directly on the shared actor parameters, so every smooth
   post-contact segment contributes gradients. Discounting and the scalar H24
   objective remain unchanged; only its pathwise derivative is a surrogate.

The disabled path must preserve the current actor loss and checkpoint schema.
The enabled treatment persists event counts, per-step masks, gradient norms,
and all existing CAGrad telemetry. No force threshold, learned horizon, reward,
observation, reset, action noise, or critic-bootstrap change is included.

## Frozen Discriminator

Use two immutable actor boundaries:

- the exact fresh seed-zero actor initialization used by E023; and
- the exact retained E023 final actor and normalizer.

For each actor, construct 120 exact reference starts: 24 replicas at phases
`0/25/50/75/100`, with a fixed float32 H24 RMR noise tape and restored actor
history/context. For the same initial states and noise, compute
training-equivalent per-environment actor gradients for:

1. ordinary H24 under `g1-4x5`;
2. topology-truncated H24 under `g1-4x5`;
3. ordinary H24 under `diagnostic-10x20`; and
4. topology-truncated H24 under `diagnostic-10x20`.

Compile the ordinary and truncated objectives separately, exactly as they
would execute in training. Persist their complete finite forward divergence
rather than requiring bit identity. A common-trajectory manual reverse pass
was rejected because it produced zero finite fresh-actor gradients in every
phase bin and therefore did not reproduce the working SHAC boundary.

Clip each environment gradient to norm 1.0 before the unchanged five-bin
CAGrad aggregation. Persist event masks/counts, raw and clipped norm
distributions, finite fractions, five phase directions, aggregate directions,
all pairwise cosines, and input/asset/code hashes. Both solver arms must execute
the same forward contract and fixed noise; their difference is diagnostic, not
a training input.

## Gates And Outcome Map

Execution is valid only if all four captures at both actor boundaries are
finite and nonzero after unchanged per-environment rejection, all five phase
bins retain a finite contributor, both separately compiled forward paths are
finite and their drift is persisted, contact
events occur in at least three of the five fixed phase bins with at least 24
total events per actor/solver boundary, and all artifact hashes validate.
The coverage gate prevents a sparse or inactive detector from authorizing
training based on an unrepresentative subset of the motion.

Define solver angular error as `1 - cosine(g_4x5, g_10x20)` for aggregate and
per-phase clipped/CAGrad directions.

- `contact-truncation-robust`: at both fresh and E023 boundaries, truncation
  reduces aggregate solver angular error by at least 20 percent, does not worsen
  any occupied phase's cosine by more than 0.02, retains nominal
  ordinary-versus-truncated cosine at least 0.8, and retains a nominal gradient
  norm ratio in `[0.25, 4.0]`. Authorize one exact fresh H24 training run with
  topology truncation.
- `contact-truncation-neutral`: execution is valid and the truncated gradient
  retains cosine at least 0.8 and norm ratio in `[0.25, 4.0]`, but the robust
  gate fails. Reject this treatment without tuning and design a hybrid
  pathwise/score-function or replay-free accept/reject update.
- `contact-truncation-destructive`: execution is valid but either actor loses
  the retained-signal cosine/norm gate, or truncation materially worsens solver
  agreement. Reject topology truncation.
- `invalid-execution`: provenance, exact input, contact identity, forward
  identity, finiteness, CAGrad, solver, or artifact validation fails. Repair
  execution only.

Outcome precedence is invalid, destructive, robust, then neutral. The 20
percent angular-error reduction is a relative robustness test, not a tunable
performance parameter.

## Implementation Boundaries

Add one small contact-topology module that discovers and validates the two G1
foot collision-geometry groups and computes support signatures/events. Add one
generic tree barrier helper with an exact-forward custom derivative. Wire a
single default-off SHAC flag and resume contract into the existing actor-loss
path. Put frozen capture, provenance, validation, plots, and manifest-last
publication in one experiment-specific tool. Do not add a scheduler, new
training framework, generic experiment abstraction, or reward feature.

## Tests

Test grouped multi-geometry support detection, inactive contacts, touchdown,
liftoff, non-foot contacts, terminal-reset exclusion, exact forward equality,
zero cross-event state/reward VJP, retained direct post-event actor gradient,
disabled legacy equality, resume fail-closed behavior, event telemetry, fixed
population/noise identity, two-solver comparison math, ordered classification,
recursive state/hash validation, atomic artifacts, and tamper rejection. Run a
compiled one-environment H24 enabled/disabled smoke before the frozen GPU
diagnostic.

## Stop Rule

This stage ends after exactly eight frozen gradient captures: two actor
boundaries by two derivative modes by two solver profiles. Do not train, tune a
force threshold, change event grouping, alter the 20-percent gate, or add a
reward from this diagnostic. A full training run is allowed only under
`contact-truncation-robust` and must be separately registered and evaluated on
the unchanged five replay-free phases with video and learning/action/contact
diagnostics.
