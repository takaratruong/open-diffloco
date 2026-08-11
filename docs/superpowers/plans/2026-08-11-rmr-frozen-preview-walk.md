# RMR Frozen-Preview Walk Implementation Plan

1. Add failing unit tests for RMR input migration, exact action equivalence,
   preview-only masks, and trainer configuration validation.
2. Implement RMR migration/masking in `preview_adapter.py` and the bounded
   trainer integration, then run the focused preview/CAGrad tests.
3. Add failing contract tests for an immutable walk runner and generic RMR
   replay-free phase-grid evaluator; implement only those interfaces.
4. Run focused tests, Ruff, `py_compile`, and a small one-environment migration
   smoke. Commit the code checkpoint.
5. Preregister and execute the untouched-parent phase grid. Continue only if it
   clears its competence gate.
6. Preregister one bounded frozen-preview training run, evaluate every archived
   checkpoint, audit frozen state directly, curate the result, and choose the
   next scientific mechanism.

