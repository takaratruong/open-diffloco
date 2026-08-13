# Fresh fixed-noise retraining implementation plan

1. Add focused RED tests for a fresh-run builder and its fail-closed contract.
2. Implement a small immutable runner by reusing the existing canonical G1
   SHAC and provenance utilities.
3. Add focused RED tests for checkpoint episode-log discovery and action
   saturation diagnostics, then implement the reusable logger.
4. Run focused tests, static checks, and a one-update launch gate.
5. Commit, register the experiment, and launch the 128-update GPU run.
6. Evaluate each archived checkpoint with matched noisy and clean rollouts,
   build the progression media/manifest, publish the dashboard preview, and
   interpret the learning trajectory.
