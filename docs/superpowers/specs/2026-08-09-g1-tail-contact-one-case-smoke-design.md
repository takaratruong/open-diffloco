# G1 Tail-Contact One-Case Smoke Design

## Decision

Run one non-scientific operational smoke on E011 shard 0, phase bin 0. Do not launch the registered twenty-case E012 audit automatically. The smoke exists only to decide whether the already established forward-action-JVP mechanism is executable and locally relevant enough to justify an actual SHAC update experiment.

## Frozen input

- E064 checkpoint, actor, normalizer, carried states, task, reward, horizon 48, population 64, sigma 0.1, gamma 0.99, and MJX solver 4/5 remain unchanged.
- Authenticate the completed E011 run and its outcome, validity, failure-weight, estimator, and manifest receipts.
- Recompute shard 0 through the authoritative E011 estimator and require byte-exact estimator, loss, phase, and selected-trajectory agreement.
- Select the single rank-zero highest-loss environment in phase bin 0, breaking loss ties by lower environment index.

## Runtime

Prepare the explicit first-action objective only after authoritative compiled/vmapped replay matches. Compile one reverse `value_and_grad` kernel and one directional `jvp` kernel. Record compilation separately from blocked cached execution. Assemble each complete 29-coordinate forward gradient sequentially and repeat exactly three times; never `vmap` the tangent directions. Run centered finite-difference probes at epsilon 0.001 and preserve the nominal done and finite-support traces.

The smoke writes a distinct receipt outside `runs/E-20260809-012`; it is operational telemetry, not E012 evidence. It records source/code/runtime hashes, the selected case, all derivative gates, preparation/compile/cached/probe timings, host peak RSS, device memory observations when supported, and a conservative twenty-case projection.

## Decision boundary

- `authorize-forward-shac-method`: authoritative replay passes; the complete forward gradient is finite, three-sweep repeat error is at most 1e-6, directional finite-difference error passes the existing 5 percent/1e-6 rule, done/support traces are preserved, and the conservative projected runtime is at most 3600 seconds with usable device headroom. Reverse disagreement may be present or absent; it is telemetry, not a reason for another diagnostic.
- `abandon-forward-shac-mechanism`: any forward-validity, authoritative-replay, memory, or runtime gate fails. Do not repair or expand this smoke into more cases.

After authorization, preregister an actual forward-action-JVP plus policy-only-reverse-VJP SHAC update. After abandonment, return to objective/variance or collocation work. In neither branch run the full twenty-case audit by default.

## Testing

Unit tests cover exact source binding, one-case selection, explicit JIT compile/cached call accounting, sequential 29-direction sweeps, memory/timing receipt construction, decision precedence, no E012 evidence writes, and fail-closed malformed inputs. The live smoke is additionally verified in a fresh process on one GPU.
