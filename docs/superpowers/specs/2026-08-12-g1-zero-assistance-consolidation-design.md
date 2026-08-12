# G1 Zero-Assistance Consolidation Design

## Question

Does extending the exact-zero-assistance tail from 16 to 80 total updates
broaden the narrow assistance transfer observed by E016?

## Causal Boundary

Resume the exact E012 final checkpoint at transition 1,720,320 and run 64
additional H12 updates to transition 2,113,536. Preserve E012's effective-512
five-bin CAGrad, frozen parent, residual Adam state, critic, reset noise,
domain randomization, solver, reward, termination, observation, and seed. Keep
the registered assistance schedule unchanged: its end remains transition
1,622,016, so every new environment receives exact zero wrench for every new
update. Archive checkpoints every 49,152 transitions.

The sole causal change is 64 more exact-zero-assistance updates. Existing root
reset perturbations remain at scale 1.0 and are not changed in this experiment.

## Decision

Evaluate all eight new checkpoints at exact phases 0/100/200/300/400 with zero
wrench. Select by minimum, median, mean, then earliest transition.

- Broad consolidation: selected policy improves E012's `51/66/68.6` key and
  does not regress any E012 phase `66/61/89/51/76`.
- Phase-local tradeoff: aggregate key improves but one or more phases regress.
- Finite no gain: no checkpoint improves the E012 key.
- Invalid execution: any resume, exact-zero, frozen-state, gradient, solver,
  checkpoint, evaluation, artifact, or provenance gate fails.

Only broad consolidation justifies continuing the same recipe. Otherwise the
next experiment changes the already-present root perturbation distribution as a
separate ingredient.
