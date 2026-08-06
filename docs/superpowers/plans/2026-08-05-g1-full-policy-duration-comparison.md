# G1 Full-Policy Duration Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic same-process evaluator that compares the source RMR actor with ordered full-policy training checkpoints and selects the earliest strict improvement.

**Architecture:** Add one dedicated tool that reuses the validated rollout and aggregation functions. Keep selection as a pure function and leave the existing single-checkpoint evaluator unchanged.

**Tech Stack:** Python, argparse, JAX/MJX, JSON, unittest.

## Global Constraints

- Evaluate the source and all candidates in one process and one environment.
- Preserve phases `0/30/60/90`, evaluation seed `0`, 60-step suffixes, and solver budget `4/5`.
- Selection requires no added terminal, positive aggregate reward delta, and at least four of six negative tracking-error deltas.
- Select the earliest passing checkpoint in CLI order; never select from training reward.
- Publish no partial JSON on failure.

---

### Task 1: Ordered Multi-Checkpoint Contract

**Files:**
- Create: `tools/compare_g1_tracking_full_policy_checkpoints.py`
- Create: `tests/test_g1_tracking_full_policy_checkpoint_comparison.py`

**Interfaces:**
- Consumes: `aggregate`, `rollout`, and `summary_delta` from `tools.compare_g1_tracking_residual`; `apply_trainable_rmr_policy`; the existing G1 evaluator environment and RMR loader.
- Produces: `build_parser()`, `candidate_passes(candidate: dict) -> bool`, `select_earliest_candidate(candidates: list[dict]) -> dict | None`, and a CLI JSON artifact.

- [ ] **Step 1: Write failing parser and selection tests**

```python
def test_cli_preserves_checkpoint_order():
    args = build_parser().parse_args([...])
    self.assertEqual([path.name for path in args.checkpoints], ["a.pkl", "b.pkl"])

def test_selects_first_candidate_meeting_strict_gate():
    selected = select_earliest_candidate([terminal_candidate, passing_a, passing_b])
    self.assertEqual(selected["step"], 12288)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_tracking_full_policy_checkpoint_comparison -v
```

Expected: import failure because the new tool does not exist.

- [ ] **Step 3: Implement the parser and pure selector**

Implement repeated ordered checkpoint parsing, the six-error predicate, strict
terminal/reward gates, and earliest selection without importing the simulator
inside the pure selector.

- [ ] **Step 4: Implement same-process rollout and atomic JSON publication**

Load each checkpoint's `actor_params`, evaluate source once per phase, evaluate
every actor sequentially, aggregate each candidate against the shared source,
apply the pure selector, reject non-finite documents, and atomically replace the
output.

- [ ] **Step 5: Run focused and neighboring tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest \
  tests.test_g1_tracking_full_policy_checkpoint_comparison \
  tests.test_g1_tracking_full_policy_comparison \
  tests.test_g1_tracking_evaluator -v
```

Expected: all tests pass.

- [ ] **Step 6: Verify and commit**

Run:

```bash
/home/ubuntu/miniconda3/envs/rl/bin/python -m compileall -q \
  tools/compare_g1_tracking_full_policy_checkpoints.py \
  tests/test_g1_tracking_full_policy_checkpoint_comparison.py
git diff --check
```

Then commit the design, plan, implementation, and tests and push the research
branch.
