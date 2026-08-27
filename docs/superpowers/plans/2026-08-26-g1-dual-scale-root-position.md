# G1 Dual-Scale Root-Position Implementation Plan

1. Add an opt-in `dual_scale` anchor-position kernel to the reward and G1
   environment while preserving the exact legacy default. Add focused value,
   gradient, and environment-boundary tests first.
2. Thread the kernel through SHAC resume authority, persisted hparams, the
   replay-free evaluator, and the phase-grid contract. Add resume and evaluator
   regression tests before implementation.
3. Add one registered paired-continuation runner from retained E002 update 8,
   validate the exact delta and checkpoint grid, run focused tests and static
   checks, then launch both arms on separate GPUs.
4. Evaluate source/control/treatment, retain only a componentwise-safe treatment
   improvement, publish curves/video to the research dashboard, and record the
   result in the experiment graph.
