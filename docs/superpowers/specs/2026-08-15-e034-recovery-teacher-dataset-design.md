# E034 Recovery Teacher Dataset Design

## Purpose

Convert E034's thirteen successful open-loop recovery tapes into immutable,
state-conditioned supervision without yet changing or training a policy.

## Data contract

Replay the exact E034 checkpoint, reference, 24 carried states, 24-by-32-by-29
correction tape, seed, environment, and fixed solver. At every pre-action state
persist the flattened 10-by-328 actor history, phase, parent action, correction,
raw parent-plus-correction action, final `[-1,1]` effective action, alive mask,
terminal bit, reward, and four normalized termination errors. Preserve all 24
trajectories and publish a success mask; downstream distillation may use only
the thirteen exact H32 successes unless separately registered otherwise.

## Validation

Fail closed unless input hashes match E034, every array is finite and has the
exact registered shape, replay survival exactly matches E034, exactly thirteen
starts complete H32, and `effective_action == clip(raw_action, -1, 1)` bitwise.
Report clip occupancy for all, recovered, and failed transitions. Write the NPZ
atomically before a hash-bound manifest-last JSON.

## Next decision

If replay is exact, train one small residual expert on the successful
state/history-to-correction pairs and evaluate it closed loop on the same 24
starts before any integration into SHAC. If replay differs, repair collection;
do not train on ambiguous targets.
