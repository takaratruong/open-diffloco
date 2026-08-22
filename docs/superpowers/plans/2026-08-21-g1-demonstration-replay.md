# G1 DiffMimic Demonstration Replay Implementation Plan

1. Add pure fail-closed replay configuration/resume helpers and environment
   state restoration with tests proving trigger, no-trigger, terminal bypass,
   exact reference state, preserved runtime context, coherent history, and
   stopped replay gradients.
2. Wire the default-off treatment through SHAC actor rollouts, hparams, metrics,
   and checkpoint telemetry. Prove legacy disabled equivalence and exact resume
   authority test-first.
3. Add an E023-derived update-16/32 runner, single-delta/provenance validation,
   replay telemetry gates, matched replay-free evaluation, outcome classifier,
   learning plot, rollout media, and manifest-last publication.
4. Run focused and neighboring tests, static checks, and a one-update CPU/GPU
   smoke. Review the bounded diff before registering or launching E008.
5. Register and execute one seed-zero treatment. Curate the exact outcome and
   stop or continue according to the preregistered result.
