# Canonical G1 Detached Resume

## Objective

Continue the finite `g1-4x5` canonical G1 SHAC arm from the immutable
step-2,359,296 checkpoint after its interactive Codex session was restarted.

## Scientific boundary

The resume changes no scientific setting. It retains the 8M-transition target,
seed 0, 256 environments, horizon 12, actor and critic architecture, optimizer
states, normalizers, environment population, PRNG state, reference-relative
zero head, randomization, curriculum, action-noise schedule, and fixed 4x5
solver. The existing `TrainState` checkpoint and exact-resume path already
carry those dynamic states.

## Operational change

Add an optional `--resume-from PATH` argument to the immutable canonical runner.
The runner resolves the path and passes it as `resume_from` to `train`; it does
not expose any scientific overrides. Launch the separately registered successor
through the existing guarded experiment runner inside a detached `tmux` session
so Codex-session lifetime cannot terminate it.

## Verification and evidence

A test must fail before implementation because the canonical kwargs builder does
not accept a resume checkpoint, then pass after it transports the exact resolved
path without changing any canonical setting. Existing exact-resume tests remain
green. The successor pins the source checkpoint SHA-256, code commit, model,
reference, solver, and output location. Success requires the detached process to
remain alive, restore step 2,359,296, and eventually publish the complete 8M
final artifacts and replay-free phase grid.
