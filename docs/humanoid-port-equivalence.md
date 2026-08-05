# Humanoid Port Equivalence Contract

## Goal

Determine whether the pinned Open-DiffLoco implementation can train a
free-base humanoid to walk without first mixing in the existing
DiffSim2Real-Lab trainer. The control is upstream commit
`f0141d38c4a107d701f36ebc516173d568f89dd4`.

## Frozen algorithm boundary

The first humanoid SHAC experiment preserves these upstream values and code
paths:

- `src/algorithms/shac/algorithm.py` SHA-256
  `69a3a9a1c6fa38a666abb245141b92225ef3701dbb8ca645c7a21e4e52228f30`
- `src/core/networks.py` SHA-256
  `b67b3956535a5c2ae899c71e7e9eb065ad12421c129bf4056d1e4e3730439b81`
- SHAC, not JAVE, for the first causal gate
- 8,000,000 requested environment steps for a full run
- 12-step differentiable unrolls
- 256 parallel environments
- 16 critic updates per actor update
- actor learning rate `5e-3`
- critic learning rate `5e-4`
- discount `0.99`
- GAE lambda `0.95`
- target critic update rate `0.01`
- actor and critic global gradient clipping at `1.0`
- actor MLP widths `(512, 256, 128)`, LayerNorm, ELU, and a zero-initialized
  tanh action head
- critic MLP widths `(512, 256, 128)`, LayerNorm, and ELU
- float64 actions and physics with float32 networks and normalized observations
- action scale `0.5`
- action-noise schedule `0.5 -> 0.32`
- five MJX physics substeps per control step at a `0.004` second physics step
- MJX solver iteration count `1`, line-search iteration count `5`, Euler
  integration, and disabled Euler damping
- friction range `(0.5, 2.0)`, mass scale `(0.85, 1.15)`, gain ranges
  `kp=(25, 45)` and `kd=(0.3, 0.7)`, COM offset
  `(+/-0.05, +/-0.05, +/-0.04)`, and velocity pushes `(-1, 1)` every four
  seconds
- the upstream velocity-command ranges, reward coefficients, reset grace
  period, curriculum behavior, checkpointing, rematerialized environment step,
  per-environment actor gradients, NaN-mean reduction, and non-finite gradient
  replacement

The JAVE arm remains frozen at its upstream SHA until SHAC demonstrates a
valid humanoid training signal.

## Permitted embodiment adaptations

Only facts that cannot remain quadruped-specific may change:

- MJCF model and home keyframe
- action and joint dimensions inferred from the model
- initial joint perturbation dimension
- pelvis/torso and foot body, site, and geometry identifiers
- two physical feet represented through the four-slot upstream terrain/contact
  interface
- nominal standing height and termination height scaled from the model's home
  pose
- observation and JAVE reward-reconstruction slices derived from the model
  action dimension
- visualization geometry and camera

No optimizer, network, SHAC objective, rollout, precision, randomization, or
reward coefficient may change in the first port.

## Staged gates

1. **Pinned Go2 runtime gate:** the unmodified upstream environment must reset,
   step, and produce a finite one-step reward gradient on the local L40S.
2. **Model gate:** the humanoid MJCF must load with `nq=35`, `nv=34`,
   `nu=28`, a valid home keyframe, four named virtual foot-contact slots, one
   solver iteration, and five line-search iterations.
3. **Environment gate:** reset and zero-action step must be finite; shapes must
   match the inferred dimensions; the home state must begin upright and above
   the termination height.
4. **Gradient gate:** eager and JIT one-step action gradients must be finite,
   nonzero, repeatable, and directionally consistent with a central finite
   difference probe.
5. **Short training gate:** an unchanged SHAC update must compile and complete
   with finite actor/critic losses and gradients. Compare its wall time to the
   pinned Go2 smoke boundary.
6. **Learning gate:** a preregistered short run must improve velocity tracking,
   survival, and visual motion over initialization before authorizing the full
   8M-step run.
7. **Replication gate:** only a successful full-motion result advances to
   multiple seeds and then to the unchanged upstream JAVE algorithm.

## Failure localization

- Go2 fails gate 1: runtime/dependency incompatibility.
- Go2 passes but humanoid fails gates 2-3: embodiment adapter defect.
- Forward execution passes but gate 4 fails: humanoid MJX/contact derivative
  defect.
- Gradient passes but gate 5 fails: algorithm/environment interface defect.
- Training is finite but gate 6 fails: reward, controller, or embodiment
  conditioning is insufficient; do not tune multiple parameters at once.

