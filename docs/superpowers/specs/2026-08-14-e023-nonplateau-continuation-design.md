# E023 Non-Plateau Continuation Design

## Decision

Continue the exact final E023 `TrainState` for 128 additional H24 SHAC updates.
The run changes only the resume source, absolute training endpoint, and archive
cadence. It preserves the optimizer state, actor and critic parameters,
normalizer, uniform exact reference-state initialization, five-bin CAGrad,
zero terminal bootstrap, plant, observations, and RMR action-noise schedule.
Because the original schedule has reached its endpoint, the continuation uses
the exact learned model-999 per-joint noise vector throughout.

## Evidence and stop condition

E023 phase-zero survival rises 42, 45, 71, and 116 at updates 16, 32, 64, and
128, while logged reward continues rising through the final update. This is a
completed budget without demonstrated convergence. Archive every 16 additional
updates and deterministically evaluate additional updates 16, 32, 64, and 128
at phases 0, 25, 50, 75, and 100. Select only a checkpoint that preserves
E023's 116/99/67/49/24 vector componentwise. A 124/99/74/49/24 vector solves
the short walk; otherwise the final curve determines whether training plateaued.

## Safety and evidence

Pin the E023 checkpoint and sibling hparams hashes, clean code commit, model,
controller, reference, solver, seed, environment, and all continuation kwargs.
Fail closed on state drift, missing archives, nonfinite leaves or telemetry,
unoccupied/invalid CAGrad bins, altered action noise, or nonzero bootstrap.
Publish the extended learning curves and selected clean/noisy rollouts through
the existing experiment registry and Obsidian projection.
