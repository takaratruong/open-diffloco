# Task 1 Report: E038 Pure Transfer Contracts

## Scope

Implemented only Task 1 in commit `297dbc1d09198001f8587d880a204b599d6d3107`:

- `tools/evaluate_g1_e038_recovery_transfer.py`
- `tests/test_g1_e038_recovery_transfer.py`

No MJX rollout, artifact publication, or Task 2 behavior was added.

## Test-first evidence

1. Added focused tests for the exact ordered 120-row source layout, malformed
   source groups, row alignment, terminal-to-survival extraction, every
   registered outcome, destructive median regression, and seed zero.
2. Ran the specified RED command before the evaluator existed. It produced
   eight expected `ModuleNotFoundError` failures for
   `tools.evaluate_g1_e038_recovery_transfer`.
3. Added a further row-alignment test after the initial green cycle. It failed
   as expected because the first implementation accepted a 119-row `qpos`
   array; the minimal row-alignment validation made that test green.

## Delivered contracts

- `validate_bank_layout` normalizes supplied arrays, requires
  `source_start_phase`, enforces five ordered 24-row source bands for phases
  `0/100/200/300/400`, and rejects non-row-aligned state arrays.
- `survival_from_terminals` derives the first terminal index per H32 row or
  returns 32 for terminal-free rows.
- `classify_transfer` applies invalid, destructive, mixed, generalizing, and
  local-only outcome precedence from the preregistered thresholds.
- `_zero_seed` and `build_parser` reject nonzero seeds.

## Fresh verification

All commands exited zero after the final change:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_e038_recovery_transfer.py
9 passed in 0.10s

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tests/test_g1_e038_recovery_transfer.py tools/evaluate_g1_e038_recovery_transfer.py
All checks passed!

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile tests/test_g1_e038_recovery_transfer.py tools/evaluate_g1_e038_recovery_transfer.py

git diff --check
```

## Self-review

The commit contains exactly the requested evaluator and test module. The
required report is intentionally left uncommitted so the scoped commit
contains only those two files. No concerns found.

## Follow-up outcome-map repair

Independent review identified an unclassified conservative boundary: a finite
run with survival regressions but no improvements could otherwise reach the
classifier's mixed fallback even when the untouched median stayed unchanged.

- Added a boundary test confirming matched improvements and regressions remain
  `recovery-expert-mixed-transfer`.
- Added a regression-only boundary test. It produced the expected RED result:
  the prior helper returned `recovery-expert-mixed-transfer` rather than the
  required `recovery-expert-destructive`.
- Updated the preregistered design definition so any regression without an
  improvement is explicitly destructive.
- Updated only the transfer helper to apply that fail-closed branch.

Follow-up commit: `587fe7060f9de4feec9472b911e833848692ae43`.

Fresh verification after the repair:

```text
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q tests/test_g1_e038_recovery_transfer.py
11 passed in 0.11s

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m ruff check tests/test_g1_e038_recovery_transfer.py tools/evaluate_g1_e038_recovery_transfer.py
All checks passed!

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m py_compile tests/test_g1_e038_recovery_transfer.py tools/evaluate_g1_e038_recovery_transfer.py

git diff --check
```
