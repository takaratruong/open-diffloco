# G1 RMR Action-Noise Design

## Question

Does replacing SHAC's uniform scalar action noise with the converged
per-joint action standard deviations learned by the successful RMR PPO policy
improve zero-assistance balance retention? The treatment is the exact 29-value
`std` tensor from RMR checkpoint `model_4999.pt` (SHA-256
`5174a0f1dc8c83ef9ea45769c3b0f19383e5aeeafea2171433f8e7bb88b21746`),
matched by joint name to the existing RMR controller authority.

## Noise contract

SHAC continues to use pathwise reparameterized noise inside the differentiable
rollout:

```text
epsilon ~ Normal(0, I)
action_noisy = clip(action_mean + action_std * epsilon, -1, 1)
```

`action_std` may be either a finite nonnegative scalar or a finite
nonnegative vector of shape `(29,)`. A vector is stored losslessly as a JSON
list in hparams and is validated against `env.actor_joint_names`; it is never
silently broadcast, reordered, clipped, or averaged. The RMR treatment vector
is fixed throughout the continuation rather than annealed. Existing scalar
runs retain byte-compatible behavior.

The treatment vector in RMR source actor order is:

```text
0.240586, 0.241456, 0.170483, 0.199771, 0.210100, 0.265766,
0.178219, 0.179202, 0.261124, 0.218310, 0.227686, 0.277083,
0.279196, 0.233729, 0.249337, 0.255798, 0.247496, 0.209423,
0.209507, 0.185532, 0.189358, 0.237779, 0.234698, 0.223405,
0.221006, 0.358318, 0.359416, 0.373313, 0.348448
```

Production constants use the full float32 values extracted from the pinned
checkpoint, not these rounded display values.

## Matched rollout pair

Before training, evaluate the exact selected E-20260812-008 checkpoint from
phase zero twice from the same reset state:

- deterministic control: `action_std = 0`;
- noisy treatment: the fixed RMR vector with seed zero.

Both arms use the same model, controller, reference, solver, checkpoint,
termination, and rollout limit. Publish provenance-bound JSON/NPZ, MP4, and
contact sheets for both. The noisy arm is descriptive robustness evidence;
it cannot select a checkpoint.

## Continuation and decision

Resume the exact selected E-20260812-008 checkpoint at transition `1,867,776`
for 32 effective-512 H12 five-bin CAGrad updates. Change only action noise from
the inherited scalar `0.32` to the fixed RMR vector under explicit resume
authority. Preserve zero terminal bootstrap, exact-zero assistance, actor,
optimizer, reset distribution, domain randomization, rewards, solver, and all
other settings.

Evaluate every eight-update checkpoint deterministically from phases
`0/100/200/300/400`. Select by maximum minimum survival, then median, mean,
then earliest transition. The treatment advances only if its key strictly
beats the parent's `56/62/66.6` and no phase falls more than five transitions
below `68/62/89/56/58`. Invalid provenance, joint order, vector shape,
nonfinite telemetry, nonzero assistance, wrong update count, or missing media
is invalid execution. A finite miss closes fixed RMR noise as the next recipe
ingredient; it does not authorize arbitrary noise tuning.

