# G1 LAFAN Action-Only Direct-Shooting Gate Implementation Plan

> **For agentic workers:** Execute inline with strict red-green-refactor TDD. Do not commit, push, register, or launch the GPU experiment.

**Goal:** Build a bounded 12-step action-only direct-shooting preflight and three-iteration optimizer over phase 105 through 117. Obtain the phase-105 initial state by carrying the exact E012 actor and full `mjx.Data` forward from one phase-zero reset, then continue without state reconstruction, replay, or reset.

**Architecture:** A focused `action_shooting` module owns the continuous physical rollout, exact SHAC reward objective, forward-JVP audit, support-switch classification, and fixed projected Armijo optimizer. A thin CLI owns immutable hashes, E012 actor initialization, fixed constants, JSON/NPZ publication, and fail-closed classification. Existing G1 environment physics, actor loading, rewards, termination errors, and fixed solver patch remain the production authority.

**Tech Stack:** Python 3.11, JAX x64, MuJoCo MJX 3.9, NumPy, stdlib `unittest`.

**Window evidence:** A read-only exact E012 carried rollout first terminates at phase 140 and changes its actual support set at phases 2, 95, 106, 135, 138, 139, and 140. The nearest finite, nonterminal 12-transition window to the rejected phase-111 start is 105 through 117. Its support chronology is right-only at phase 105 and bilateral from phase 106 through phase 117.

## Global Constraints

- Base commit is `1498b3791ce7d1e59e7237ed90ee17774cbcdfd6` in the isolated branch `research/g1-lafan-action-shooting-20260808`.
- Reference SHA-256 is `bf8c8b407062d1b309440f4c1787c345b04d79501ea75f615e5b41c0c5ebb6db`.
- E012 step-688128 checkpoint SHA-256 is `12198b38443c2705da5e26a58ddd320f4d5837880b32f7404db428b7220164d4`.
- E012 hparams SHA-256 is `a03053410c21c54d4175c7634eaf77f7886b4ec9a23e515952d0ad5d7380c3cf`.
- The validated task remains 50 Hz, reference stride one, solver iterations 4, line-search iterations 5, and float64 physics.
- Reset exactly once at phase zero, carry the E012 actor and full `mjx.Data` through 105 prefix transitions, and retain that complete carried phase-105 data as the immutable shooting initial state. Carry it for 12 further transitions through phase 117. Never use `reset_at_phase(105)`, reconstruct qpos/qvel, replay reference state, or invoke task reset after phase zero.
- Decision variables are the exact E012 actor outputs with shape `(12, 29)`. The validated source-order task deliberately permits values outside `[-1, 1]`; projection means the fixed L-infinity trust box `actor_action +/- 0.02`, not clipping the immutable actor initializer.
- Objective is negative mean unchanged SHAC reward plus `1e-3 * mean((action - actor_action)^2)`.
- Preflight requires direct/JVP primal identity at most `1e-8`, 348 finite scalar JVPs, two gradients agreeing within `1e-6`, and sampled central-FD relative error at most `0.05`. Nominal, both centered-FD probes, and every Armijo candidate must have the exact 13-state support trace: phase 105 right-only `[false, true]`, followed by bilateral `[true, true]` at every phase 106 through 117. Any additional support switch fails closed.
- Optimization is exactly three iterations, trust radius `0.02`, and ordered multipliers `1, 0.5, 0.25, 0.125`.
- Material success additionally requires direct carried mean reward improvement at least `0.001`, no terminal, and no state reset.
- No GPU experiment, registry edit, commit, or push is authorized.

---

### Task 1: Pure contracts and fixed projected Armijo behavior

**Files:**
- Create: `src/envs/g1_tracking/action_shooting.py`
- Create: `tests/test_g1_action_shooting.py`

**Interfaces:**
- Produces `ShootingConfig`, `validate_action_sequence`, `support_switch_count`, `project_trust_box`, and `run_projected_armijo`.
- `run_projected_armijo(actions, objective_and_gate, gradient_fn, config)` returns an immutable trace and selected actions after exactly three iterations.

- [ ] Write failing tests for the fixed dimensions/constants, exact left/right support-switch count, trust-box projection around an out-of-range actor tape, ordered line-search acceptance, and exactly three iterations.
- [ ] Run `python -m unittest tests.test_g1_action_shooting -v` and confirm imports fail for the missing module.
- [ ] Implement only the pure validation, switch, projection, and optimizer contracts required by those tests.
- [ ] Re-run the focused test and retain green output.

### Task 2: Continuous full-state rollout and unchanged task objective

**Files:**
- Modify: `src/envs/g1_tracking/action_shooting.py`
- Modify: `tests/test_g1_action_shooting.py`

**Interfaces:**
- Produces `rollout_actions_without_reset(env, initial_data, start_phase, initial_previous_action, actions)` and `shooting_objective(rollout, actions, nominal_actions, action_deviation_weight)`.
- Rollout returns carried qpos/qvel, prepared actions, raw torques, per-step rewards, termination errors, terminal flags, and phases 106 through 117 while keeping `mjx.Data` as the scan carry.

- [ ] Write a failing fake-environment test proving the second transition receives the first transition's full data object rather than reconstructed qpos/qvel.
- [ ] Write a failing exact-reward test proving the objective is negative mean rollout reward plus the declared regularizer.
- [ ] Implement the fixed-shape `lax.scan` over `env.advance_physics`, reuse `_body_state`, `_tracking_reward_from_body_state`, `rmr_regularization_reward`, and `_termination`, and expose only the registered arrays.
- [ ] Run the focused tests and preserve green.

### Task 3: Forward action-gradient identity and FD gate

**Files:**
- Modify: `src/envs/g1_tracking/action_shooting.py`
- Modify: `tests/test_g1_action_shooting.py`

**Interfaces:**
- Produces `canonical_forward_gradient(objective_fn, actions, identity_tolerance)` and `directional_fd_audit(objective_fn, actions, gradient, epsilon, seed)`.
- The gradient helper performs all `actions.size` canonical `jax.jvp` calls, compares every returned primal with one direct value, and returns the 348-vector gradient plus maximum primal defect.

- [ ] Write failing quadratic-objective tests for exact 348-entry ordering, primal identity rejection, nonfinite derivative rejection, repeatability, and support-safe centered-FD relative error.
- [ ] Implement sequential canonical JVP assembly and one deterministic normalized FD direction without reverse mode.
- [ ] Run focused tests and preserve green.

### Task 4: E012 actor tape and physical contact chronology

**Files:**
- Modify: `src/envs/g1_tracking/action_shooting.py`
- Create: `tests/test_g1_action_shooting_runner.py`
- Create: `tools/run_g1_action_shooting_gate.py`

**Interfaces:**
- Runner loads the existing `_load_policy` actor, creates one exact phase-zero reset, carries actor history, the prepared previous action, and full `mjx.Data` for 105 transitions, and captures the next 12 closed-loop actor actions over phases 105 through 117. The differentiable evaluation reuses the complete phase-105 carry and never invokes `env.step`, task reset after phase zero, or qpos/qvel reconstruction.
- Produces `support_trace_from_states(model, qpos, qvel, penetration_allowance)` using MuJoCo CPU kinematics/contact evaluation and left/right foot geometry names.

- [ ] Write failing runner tests for exact parser defaults, immutable hashes, action shape, phase chronology, checkpoint identity failure, and left/right support-switch reporting.
- [ ] Implement the parser, fail-closed identity checks, actor-tape capture, and physical support trace without adding an optimizer framework.
- [ ] Run runner and core unit tests and preserve green.

### Task 5: Bounded evidence publication and classification

**Files:**
- Modify: `tools/run_g1_action_shooting_gate.py`
- Modify: `tests/test_g1_action_shooting_runner.py`

**Interfaces:**
- Produces `preflight.json`, `initial_rollout.npz`, `gradient_gate.json`, `optimization_trace.json`, `candidate_rollout.npz`, and `summary.json` under `--output-dir`.
- Classification is one of `contact-shooting-authorized`, `finite-contact-no-material-step`, `action-gradient-identity-blocked`, `contact-window-invalid`, or `invalid-execution`.
- The output directory must not exist and must not be a symlink. A successful run contains exactly the six regular files above; `summary.json` closes the other five artifacts with their SHA-256 digests.

- [ ] Write failing artifact/classification tests using deterministic toy results; require strict JSON without NaN and atomic writes.
- [ ] Implement bounded classification and artifact writers. Do not implement rendering or launch physics from unit tests.
- [ ] Run runner/core tests and preserve green.

### Task 6: Fresh verification

**Files:**
- Verify all files above; do not commit.

- [ ] Run `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true /home/ubuntu/miniconda3/envs/rl/bin/python -m unittest tests.test_g1_action_shooting tests.test_g1_action_shooting_runner tests.test_g1_failure_collocation tests.test_g1_collocation_transfer tests.test_g1_tracking_environment.G1TrackingEnvironmentTest.test_advance_physics_matches_nonterminal_checkpointed_step_data -v`.
- [ ] Run `/home/ubuntu/miniconda3/envs/rl/bin/python -m py_compile src/envs/g1_tracking/action_shooting.py tools/run_g1_action_shooting_gate.py tests/test_g1_action_shooting.py tests/test_g1_action_shooting_runner.py`.
- [ ] Run `git diff --check`, `git status --short`, and inspect the complete diff against the approved contract.
- [ ] Report exact verification evidence and any remaining GPU-only boundary. Do not run the registered scientific command.
