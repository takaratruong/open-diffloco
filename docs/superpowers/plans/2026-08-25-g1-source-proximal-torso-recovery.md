# G1 source-proximal torso recovery implementation plan

1. Add fail-closed source-policy anchor loading and configuration tests.
2. Apply the frozen source residual on the candidate's current observations and
   use it as the weight-one action anchor target.
3. Persist and validate source checkpoint provenance without changing the
   default parent-anchor path.
4. Add an exact E026 continuation runner with one 32-update treatment and
   eight-update checkpoints.
5. Run focused tests, static checks, clean preflight, training, replay-free
   phase-grid evaluation, pitch diagnostics, and video publication.
