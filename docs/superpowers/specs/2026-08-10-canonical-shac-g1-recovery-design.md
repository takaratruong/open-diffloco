# Canonical Open-DiffLoco SHAC Recovery for G1

## Objective

Repair the missing baseline in the G1 differentiable-physics program. First
establish what the unmodified Open-DiffLoco SHAC recipe does on a humanoid at
its full training budget. Then run the corrected G1 motion-tracking task with
the same learning recipe while comparing the upstream and G1-selected solver
configurations.

Finite-difference agreement is diagnostic telemetry, not an update-admission
gate. Behavioral learning and replay-free closed-loop evaluation decide
whether simulator gradients are useful.

## Existing evidence and gap

- E016--E021 preserved the upstream SHAC and network hashes and changed only
  the embodiment adapter. The longest run stopped at 494,592 transitions and
  161 actor updates without stepping.
- E023 showed that pristine Go2 also does not learn in the short gate and
  requires 8,000,000 transitions and 2,604 actor updates for active locomotion.
  The clean humanoid port was therefore stopped after only 6.2 percent of the
  relevant budget.
- E039 and E048 attempted upstream-style learning on G1 reference tracking but
  were invalidated by rotation, joint-order, and action-support defects.
- E069 is the only valid upstream-style successor on the corrected task, but
  it used 64 environments for 64 updates. No corrected authoritative-scale or
  full-budget canonical G1 run exists.
- Forward solver screens establish that stock 1/5 fails the source G1 tracker,
  fixed-scan 4/5 completes all 60 steps and slightly improves the measured
  errors over 10/20, and 10/20 is much more expensive. No paired learning
  comparison exists.

## Experimental lanes

### Lane A: frozen upstream humanoid completion

Run the E016 frozen humanoid port from initialization for 8,000,000
transitions. Preserve the upstream network, optimizer, actor bootstrap,
gradient aggregation, 256 environments, horizon 12, domain randomization,
curriculum, action-noise schedule, actor history, and stock MJX 1/5 solver.
This lane answers whether the original command-conditioned locomotion recipe
can learn any humanoid gait. It does not answer G1 reference tracking.

### Lane B: corrected G1 canonical solver pair

Run two otherwise identical corrected-G1 campaigns concurrently:

1. **Upstream solver arm:** stock MJX with one outer iteration and five line
   searches.
2. **G1 solver arm:** the same MJX solver equations with four outer iterations,
   five line searches, and the process-local reverse-safe fixed scan.

Both arms run to the 8,000,000-transition target and publish immutable
checkpoints at 128, 512, 1,024, and 2,604 actor updates. Early checkpoints are
evaluated but do not terminate a finite full-budget run. This avoids repeating
the underpowered stopping decision made after E021 and E069.

A fixed-scan 10/20 arm is authorized only if 4/5 is nonfinite or materially
worse than 1/5. It is not part of the default pair because the existing forward
screen shows no fidelity benefit and a tenfold larger outer-times-line-search
budget.

## Canonical learning contract

Except for explicit morphology and task adapters, Lane B matches the pristine
Open-DiffLoco SHAC contract:

- 256 environments and horizon 12 (3,072 transitions per actor update)
- 8,000,000 requested transitions, producing 2,604 complete actor updates
- 16 critic updates per actor update
- actor MLP 512--256--128 with LayerNorm, ELU, zero-initialized tanh head
- critic MLP 512--256--128 with LayerNorm and ELU
- actor learning rate 5e-3 and critic learning rate 5e-4
- action noise linearly annealed from 0.5 to 0.32
- terminal target-critic bootstrap enabled at scale 1.0
- upstream per-environment gradients, elementwise `nanmean`, remaining
  nonfinite entries replaced by zero, and global norm clipping at 1.0
- ten-frame actor history
- actor-only observation noise on equivalent current-state fields: angular
  velocity at 0.2, anchor-orientation representation at 0.05, and joint
  position/velocity at 0.01; reference targets, previous actions, and critic
  observations remain clean
- curriculum grace for the first 10 percent of transitions and ramp over the
  next 80 percent
- no finite-difference rejection of a finite update

Per-environment norm clipping remains implemented but disabled for the
canonical arms. Telemetry records finite fractions and per-environment norms.
It may be enabled only in a separately registered successor if an observed
outlier dominates an update or creates nonfinite state.

## Necessary G1 task adapters

The following differences are intentional and must remain identical across
the solver pair:

- 29 G1 actions and the corrected source joint order
- the corrected RMR reference-conditioned actor and critic observations
- the RMR full-body tracking reward, action-rate penalty, joint-limit penalty,
  and strict termination contract
- 50 Hz control with four 5 ms physics substeps and the pinned reference
- a reference-relative reset rather than a fixed standing reset
- G1 per-joint PD action scales and effort limits
- float64 MJX physics with float32 networks

The tanh head and action clipping are restored for this canonical comparison.
An unbounded linear-head successor is a separate action-support ablation, not
part of the solver comparison.

## Domain-randomization mapping

The current G1 environment silently ignores most Go2 randomization arguments,
so Lane B requires an explicit per-environment model path. The mapping is:

- friction: multiply all baseline friction components by a per-episode scale
  sampled uniformly from 0.5 to 2.0
- mass and inertia: multiply every non-world body's baseline values by one
  per-episode scale sampled uniformly from 0.85 to 1.15
- proportional gains: preserve the G1 joint-specific gain ratios and multiply
  the complete source gain vector by a scalar in 25/35 to 45/35
- derivative gains: preserve joint-specific ratios and multiply the source
  vector by a scalar in 0.3/0.5 to 0.7/0.5
- COM offset: add a per-episode offset within plus or minus
  (0.05, 0.05, 0.04) m to the pelvis inertial frame
- velocity pushes: set root x/y velocity from a uniform (-1, 1) m/s sample at
  the upstream four-second interval
- terrain: disabled, matching the pristine `shac_go2.yaml` run
- reset perturbations: sample a uniform reference phase and perturb root pose,
  root velocity, and joints with the existing scale-one reference-reset
  distribution; this is the reference-tracking analogue of perturbing Go2's
  standing reset

All sampled physical parameters are stored in environment state, are carried
through steps, are resampled only on reset, and affect both forward dynamics
and differentiated rollouts. Tests must prove that difficulty zero is nominal,
difficulty one reaches every registered range, and two environments can carry
different models inside the same vectorized update.

## Evaluation and decision rules

Every fixed checkpoint receives deterministic replay-free evaluation from the
same frozen phase grid. Report survival, terminal cause, tracking reward, all
six body-space errors, action magnitude, finite-gradient fraction, raw
gradient-norm distribution, and video.

Solver selection uses this order:

1. finite completion and finite checkpoints
2. replay-free survival across phases
3. body-position and orientation tracking
4. lower compute and compile cost

Training reward alone cannot select a checkpoint or solver. A solver wins only
on material closed-loop evidence. If 1/5 and 4/5 are behaviorally equivalent,
select 1/5 for upstream fidelity; if 4/5 preserves the already demonstrated
forward advantage or learns materially better, select 4/5.

## Verification

Before launching Lane B:

- unit tests cover each randomization range and nominal difficulty behavior
- a same-seed 1/5 stock-versus-fixed-loop identity test is bit-exact or within
  a preregistered floating-point envelope
- a two-environment differentiated smoke proves distinct randomized models,
  finite rewards, finite actor/critic gradients, and a finite optimizer update
- effective hyperparameters and solver provenance are serialized and checked
  before training
- the solver arms start from byte-identical actor, critic, optimizer,
  normalizer, phase, and random-key state

The experiment registry records invalid execution separately from negative
learning evidence. A failed compile or broken artifact contract does not count
against either solver.
