# Isolated RMR Action-Noise Gate

## Goal

Replace the visually underpowered fixed-0.2 half-scale diagnostic with one clean action-exploration gate that matches the RMR action boundary while excluding observation, reset, physics, and external-force noise.

## Contract

- Fresh zero-output actor; no resume or inherited policy.
- `q_target = q_reference + action * exported_joint_scale` with residual scale `1.0`.
- Linear actor output and no normalized-action clamp. Physical effort and joint limits remain active.
- Gaussian action noise starts at scalar standard deviation `1.0` and anneals toward the pinned learned 29-joint RMR standard-deviation vector over 800,000 transitions.
- Clean actor observations, exact uniformly sampled reference-state resets, nominal mass/friction/gains/COM, and no pushes, terrain force, carried reset, root perturbation, or torso assistance.
- Preserve the established effective-512, H12, five-bin CAGrad, future-reference, solver, reward, and termination recipe.

## Gate and Evidence

Run exactly one 6,144-transition update first. It passes only if the runtime hparams prove the contract, all five CAGrad bins are occupied, actor gradients and updates are finite and positive, and serialized state is finite. Then create a seeded phase-zero training-distribution replay from the resulting checkpoint using the exact action-noise setting, save its action/noise tape, MP4, and contact sheet, and expose a robot-only preview in Obsidian. This replay is diagnostic evidence of the distribution used by training; it is not a locomotion claim.

Only after the gate and visual inspection pass may the longer 786,432-transition run start. Early checkpoints must retain the same training-episode evidence.

