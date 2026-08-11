# G1 CAGrad Horizon-24 Exact Continuation

## Objective

Test one scientific change after E008: increase the differentiable actor
horizon from 12 to 24 simulator steps while preserving the complete five-bin
CAGrad treatment and the exact final E008 `TrainState`.

## Immutable contract

The runner requires the E008 final checkpoint at step `1,179,648` and inherits
the reviewed E008 runner contract: 256 physical environments, two sequential
gradient shards, five phase bins, CAGrad `alpha=0.5`, 32 Frank-Wolfe
iterations, solver `g1-4x5`, seed zero, and checkpoint interval `196,608`.
It changes only `unroll_length` from 12 to 24 as a scientific treatment.

The absolute endpoint becomes `1,572,864`, adding exactly `393,216`
transitions. At 512 effective environments and horizon 24 this is 32 actor
updates. Checkpoints therefore land after updates 16 and 32, at steps
`1,376,256` and `1,572,864`. The endpoint change encodes the fixed continuation
budget; it is not a second treatment.

The CLI exposes only solver profile, reference path, seed, output root, and a
required resume path. Batch shape, horizon, budget, optimizer, CAGrad settings,
and checkpoint cadence cannot be overridden.

## Verification boundary

Contract tests compare the runner with the E008 builder after applying only
the new absolute endpoint, prove that horizon is the sole remaining delta,
check the exact update/checkpoint arithmetic, and reject scientific CLI
overrides. This task does not register or launch the experiment.
