# E034 State-Conditioned Recovery Expert Implementation Plan

1. Add focused failing tests for dataset filtering, normalization/frame
   extraction, expert fitting, outcome classification, and artifact validation.
2. Implement one bounded training/evaluation tool that validates all immutable
   inputs, fits the existing residual adapter, and performs the exact 24-state
   closed-loop evaluation.
3. Run focused tests and static checks, then obtain an independent code review.
4. Register one GPU experiment, execute it once, validate and curate its
   artifacts, and update the generated research vault.
5. If the reproduction gate passes, render the resulting closed-loop rollouts;
   otherwise use the measured failure mode to choose the next causal change.
