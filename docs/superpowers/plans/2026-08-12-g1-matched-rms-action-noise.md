# G1 Matched-RMS Action-Noise Control Plan

1. Add RED tests for the exact RMS constant, the minimal kwargs delta, immutable
   CLI, preflight provenance, and training validation with scalar endpoints.
2. Extract a parameterized action-noise continuation builder and validator from
   the E010 runner while preserving its public behavior and tests exactly.
3. Add a thin immutable matched-RMS runner that reuses the proven E010 preflight,
   CAGrad, checkpoint, and artifact validators with a distinct protocol and
   output filenames.
4. Run focused E010 and matched-control tests, Ruff, py_compile, and diff check;
   obtain an independent bounded review before launch.
5. Register one 32-update GPU run, evaluate all four checkpoints at phases
   `0/100/200/300/400`, select deterministically, and compare directly with
   E010 and the same-code E008 parent.
