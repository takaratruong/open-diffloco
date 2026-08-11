# G1 Frozen Residual-Preview Adapter Design

## Scientific question

Can a small nonlinear action-residual adapter improve robust replay-free phase
survival when the exact E008 parent policy, parent optimizer moments, actor
normalizer, simulator, differentiable objective, CAGrad, batch, horizon, and
training exposure remain fixed?

E-20260811-001 and E-20260811-003 show that append-only preview learning is
finite, repeatable, and behaviorally useful, but the only trainable mapping is
from preview inputs into a frozen first hidden layer. E-20260811-007 shows that
replacing absolute previews with future-minus-current deltas changes which
phase survives without raising the worst-case or median survival. Parent
forgetting and simple input redundancy are therefore no longer sufficient
explanations. The remaining capacity question is whether future context needs
its own nonlinear path to the action.

## Alternatives considered

1. Unfreeze the complete parent actor. This provides maximal capacity but
   reintroduces the interference already observed in E-20260811-000, so a
   negative result would be uninterpretable.
2. Add another trainable hidden layer inside the parent. This also changes the
   parent's internal representation and cannot prove exact parent
   preservation.
3. Add a parallel zero-effect residual adapter while freezing the parent.
   This cleanly tests capacity, retains the known parent behavior at update
   zero, and is the selected design.

## Policy architecture

The treatment resumes exact E008 and performs the already validated append-only
future-reference state migration in `delta` mode. The migrated parent actor
still receives the full normalized 10-frame treatment observation, but every
parent parameter and optimizer moment is frozen. Its appended preview rows
remain zero forever.

A separate one-hidden-layer ELU adapter receives only the newest normalized
328-value treatment frame: the 154-value current reference-conditioned state
plus 174 future-delta values at offsets +4/+8/+12. Its hidden width is 256 and
its 29-value output head is initialized to exact zeros. The adapter contributes
`tanh(logits)` as an additive action residual. The environment's existing
action preparation retains the physical `[-1, 1]` action boundary. The adapter
contains exactly 91,677 trainable scalars, close to the 89,088 trainable scalars
in the linear-preview control.

The parent parameter tree is stop-gradiented, which preserves policy feedback
derivatives through observations while removing parent-parameter gradients
from CAGrad. An exact mask remains mandatory because inherited Adam momentum
would otherwise move frozen values even under zero new gradient. Only adapter
parameters and adapter Adam moments may change; the parent parameters, parent
Adam moments, actor normalizer, PRNG/environment continuation state, critic
state, and global optimizer count retain their established semantics.

## Training and telemetry

The treatment preserves E001/E003/E007's seed 0, E008 parent, 64 H12 updates,
256 physical environments, accumulation factor two, effective population 512,
five phase-bin CAGrad, zero termination margin, action-noise schedule, domain
randomization, fixed 4x5 MJX solver, and 393,216-transition exposure. It archives
every eight updates.

Migration evidence must prove parent action equivalence to `1e-7`, exact parent
parameter and moment retention, zero adapter moments, finite adapter state, and
an exact-zero initial residual. Each checkpoint must record finite nonzero
adapter gradients and updates, five occupied CAGrad bins, parent/action
deviation, and zero parent/moment/normalizer drift. The configuration defaults
off and treated checkpoints cannot silently resume with the flag, hidden width,
or representation changed.

## Evaluation and branch decision

All eight checkpoints are evaluated replay-free from phases 0/100/200/300/400
and selected by maximum minimum survival, then median, mean, and earliest step.
Advancement requires per-phase floors 23/50/72/16/40 and a key strictly greater
than E001's 27/54/53.0. Training reward cannot select a checkpoint.

If the adapter advances, retain frozen-parent nonlinear residual capacity for
the next motion. If training is valid but no checkpoint advances, reject
capacity-only adaptation under this contract and move to an objective that
contains carried survival information, such as a learned terminal value. Do
not unfreeze the parent as another rescue. Invalid migration, frozen-state,
telemetry, provenance, evaluation, crash, or timeout evidence is repaired
without scientific interpretation.

## Verification boundary

Tests must prove exact-zero initial composition; adapter-only gradients,
updates, and moments; exact parent and normalizer preservation; current-frame
input slicing; explicit resume validation; unchanged disabled behavior; runner
delta isolation; evaluator parity with training composition; and persistent
metadata for adapter kind, width, parameter count, and delta preview mode.
