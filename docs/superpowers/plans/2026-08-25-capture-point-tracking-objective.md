# Capture-point tracking objective implementation plan

1. Add shared CPU/MJX capture-point measurement to the centroidal module with
   parity, shape, finite-value, and gradient tests.
2. Add a pure masked pseudo-Huber capture objective and frozen-trace
   discriminator tests.
3. Integrate the default-off objective into SHAC with explicit resume
   authority, H+1 state capture, telemetry, hparams, and disabled equivalence.
4. Add an immutable frozen-E026 continuation runner and exact-delta tests.
5. Run the no-training discriminator and a bounded derivative/gradient-scale
   audit. Register one static weight; no sweep.
6. If and only if those gates pass, run the bounded continuation, evaluate all
   checkpoints on the replay-free five-phase grid, publish video/curves, and
   register the result in the research graph.

