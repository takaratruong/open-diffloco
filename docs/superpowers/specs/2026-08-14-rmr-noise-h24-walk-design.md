# RMR-Calibrated H24 Walking Noise Design

## Question

Does E022 fail to learn the complete walking catch sequence because its fixed normalized action-noise standard deviation of `0.2` is substantially narrower than the competent RMR walking PPO training distribution?

## Evidence motivating the treatment

The action-to-PD-target and nominal action-to-torque mappings already match RMR. Both sample in normalized action coordinates and then compute `q_target = q_reference + action_scale * action`. E022 fixes every joint at standard deviation `0.2`. The pinned RMR walking checkpoints instead have action-standard-deviation RMS values `0.9972532` at model 0, `0.64704794` at model 500, and `0.4521128` at model 999. The final 29-joint vector comes from model-999 SHA-256 `5db9d8371754a635d162c416e192b49ec2064d3133d20eea0df63463d1c8ae03`.

## Treatment

Derive the complete E022 H24 fresh full-action recipe and change only:

- `action_noise_std_start`: scalar `0.2` to scalar `1.0`.
- `action_noise_std_end`: scalar `0.2` to the exact model-999 29-joint standard-deviation vector in RMR actor order.
- `action_noise_schedule_steps`: preserve the existing full-run endpoint `1,572,864`, yielding a deterministic linear approximation to RMR's learned decay.

Preserve H24, effective population 512, 128 optimizer updates, fixed seed zero, fresh zero output head, full unbounded reference-relative action authority, clean observations, exact reference-state initialization, nominal physics, no pushes, no assistance, no domain randomization, zero terminal bootstrap, five-bin CAGrad, per-environment clipping, and every other E022 setting.

## Alternatives rejected

- A larger fixed scalar does not reproduce RMR's decreasing exploration.
- Learning a SHAC variance parameter would add a new optimizer objective and entropy/variance-collapse question, confounding the immediate noise-width discriminator.
- A three-knot hand schedule would overfit three saved PPO checkpoints without evidence that intermediate shape, rather than endpoints, matters.

## Evaluation

Require exact provenance and finite training artifacts at updates 16, 32, 48, 64, 80, 96, 112, and 128. Evaluate replay-free clean phase grids at updates 16, 32, 64, and 128 using phases 0/25/50/75/100. Compare against E022 final `[74,84,49,49,24]` and E021 componentwise floors `[62,63,55,49,24]`. Render the selected clean rollout and its exact checkpoint-bound training-noise episode with clean observations and exact phase-zero reset.

The treatment advances only if a finite checkpoint is componentwise safe against E021 and improves phase-zero survival beyond 74 without a compensating phase regression. Exact completion remains `[124,99,74,49,24]`. Otherwise broader RMR-calibrated action noise is insufficient under fresh H24 SHAC.
