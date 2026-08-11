# Frozen-Preview Termination-Margin Implementation Plan

1. Add failing contract tests for an immutable continuation from E011's selected
   midpoint.
2. Add a runner whose only scientific change is margin weight 0.5, plus the
   explicit resume-treatment flag; fix the endpoint and checkpoint cadence for
   32 updates and 196,608 transitions, matching E011's final global step.
3. Run focused runner, resume, reward, preview, and CAGrad tests with x64, then
   Ruff, `py_compile`, and diff checks.
4. Preregister only after the current parent screen is evaluated. Execute one
   seed-zero continuation, audit the frozen boundary, evaluate all checkpoints
   replay-free, and curate the immutable selection.
