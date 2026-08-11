# G1 History-Faithful Carried-Survival Design

## Purpose

Test whether the remaining G1 SHAC failure is caused by a temporal-credit and
state-distribution mismatch. The selected E008 policy survives 44 to 95 steps
from the registered starts, while its differentiable actor horizon is only 12
steps. Uniform reference-state initialization therefore rarely places the H12
objective on the drifted states that precede closed-loop failure.

## Alternatives Considered

### 1. History-faithful carried resets — selected

Collect states reached by the selected E008 policy before each registered
failure and restore their complete actor context during training. This directly
tests the state-distribution hypothesis while leaving the actor objective,
optimizer, architecture, and simulator fixed.

### 2. Existing qpos/qvel-only carried resets — rejected

The existing bank restores `qpos`, `qvel`, and phase, but reset reconstructs the
ten-frame actor history by repeating one frame and sets the previous action to
zero. E008's actor observes both quantities. This is not the policy-carried
state that produced the failure and would confound a negative result.

### 3. Terminal-value objective — deferred

The actor already uses critic bootstrap. Changing its terminal-value treatment
would test critic calibration and temporal credit simultaneously. It remains
the next bounded intervention if faithful carried resets fail.

## Frozen Scientific Boundary

The treatment starts from E008's selected Adam checkpoint at transition
1,327,104. It preserves the frozen parent actor, trained nonlinear delta-preview
residual, residual Adam optimizer, actor normalizer, critic, H12 rollout,
effective batch 512, five-bin CAGrad, fixed `g1-4x5` solver, reference motion,
reward, termination thresholds, action noise schedule, seed, and nominal physics.
The sole causal change is the reset distribution.

Training continues for 64 updates, or 393,216 transitions, and writes the same
eight-checkpoint grid used by E008 and E009.

## Immutable Carried-State Bank

Run replay-free nominal rollouts of the exact selected E008 checkpoint from
phases `0/100/200/300/400`. For each rollout, retain the 24 nonterminal pre-step
states whose old-policy terminal transition is 6 through 29 steps away. This
produces 120 rows: enough to cover immediate failure gradients and earlier
precursors without using the final five likely unrecoverable states.

Each row stores:

- `qpos` and `qvel`;
- reference `phase`;
- `last_act` in the environment's prepared action convention;
- the complete `actor_obs_history` with shape `(10, actor_frame_obs_dim)`;
- source start phase and transitions-to-terminal provenance;
- the action subsequently selected by E008;
- termination errors and thresholds for admission checks.

The collector observes every complete suffix before admitting a bank. Two
GPU-7 preflights of the exact E008 checkpoint measure `75/63/94/74/45` and
`63/63/95/69/44`, demonstrating process-level MJX contact nondeterminism even
on one physical GPU. Exact survival is therefore provenance, not an admission
invariant. Every source must terminate and survive at least 29 transitions so
the fixed band exists. The collector rejects any row that is non-finite or
already outside a hard termination threshold, any phase discontinuity, any
unnormalized root quaternion, or any last history frame that differs from a
fresh observation of the stored physical state and previous action. It writes
the NPZ atomically; its SHA-256 freezes the exact sampled trajectory states.

## Reset Semantics

On each reset, the registered 0.5 Bernoulli selects either an exact carried-bank
row or E008's unchanged noisy-reference fallback. Carried rows are never
perturbed; non-carried rows retain `reference_reset_noise_scale=1.0`. This
preserves E008's original reset distribution in the control half instead of
silently replacing it with exact reference states.

At each environment reset, sample the existing uniform reference phase and one
bank row. Select the carried row with fixed probability 0.5. A reference reset
keeps the existing zero previous action and repeated-frame history. A carried
reset restores the matched physical state, phase, previous action, and complete
history from one bank index. Bootstrap actor and critic observations must agree
with that restored context.

Legacy banks without policy context retain their existing behavior outside this
runner. The E010 runner requires the context-rich schema and fails closed if it
is absent or mismatched. Resuming with a changed bank is forbidden by default;
only the dedicated runner opts into the exact registered bank SHA and reset
probability.

## Evaluation And Decision

Every checkpoint is evaluated replay-free on the unchanged registered starts.
E008's archived control vector is `70/63/95/70/44`, with minimum/median/mean
`44/70/68.4`. Because individual contact traces are not repeatable, the selected
treatment checkpoint is also evaluated against E008 in a paired parent-versus-
treatment process on the same GPU. The treatment advances only if it:

1. clears the archived E008 phase floors and lexicographic key;
2. improves the paired parent comparison rather than only an unpaired replay;
3. has finite training state and nonzero authorized adapter updates;
4. preserves the frozen parent actor and actor normalizer exactly; and
5. restores the registered carried context exactly in a reset audit.

Dense checkpoint selection remains minimum, median, mean, then earliest step.
No training loss is used for behavioral selection. A held-out offset phase grid
may be reported diagnostically, but it cannot replace the registered gate.

## Failure Outcomes

- If training is finite but does not clear the no-regression gate, conclude that
  this fixed 50/50 carried-reset intervention is insufficient and move to a
  separately registered terminal-value treatment.
- If context restoration, provenance, resume identity, or frozen-state auditing
  fails, classify the run invalid and repair only that boundary before rerun.
- Do not tune reset probability, bank offsets, Muon, activation, width, horizon,
  reward weights, or phase sampling from this result.

## Testing

Pure tests cover bank row selection, schema validation, exact context pairing,
resume override admission, and runner arguments. Environment tests verify that
a probability-one carried reset restores one row's physical state, phase,
previous action, history, actor observation, and critic bootstrap consistently;
probability-zero and legacy behavior remain unchanged. A one-update JIT smoke
must prove finite gradients and state before experiment registration.
