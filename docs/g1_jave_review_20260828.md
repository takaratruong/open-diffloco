# G1 differentiable tracking and JAVE review — 2026-08-28

Status: engineering review and smoke evidence. The JAVE runs below were not
registered scientific experiments and do not change the retained policy.

## Bottom line

The evidence no longer supports “G1 cannot track this motion” as the leading
hypothesis. An action-matched PPO actor completed the registered 125-state walk
in Isaac Sim and then completed every exact MJX suffix without a policy update
(E-20260813-020 and E-20260813-023). A learned torso wrench also completed the
longer 271-transition evaluation (E-20260825-004). A dynamically adequate
controller and a stabilizing correction therefore exist.

The repeated differentiable-training failure is narrower: useful corrections
are learned for particular starts and contact transitions, but a shared actor
update trades competence between them. Freezing the retained parent and adding
a zero-initialized residual did not remove the trade, so this is not merely
catastrophic parameter overwriting. Demonstration replay, noisy reference-state
initialization, carried-state recovery, teacher objectives, root objectives,
and additional residual capacity all produced the same broad pattern.

JAVE is now implemented on the exact G1 SHAC continuation path and its learned
dynamics and gradient-Bellman losses are finite and active. The first smoke does
not establish a policy improvement. It also does not reject JAVE: the only
evaluated active checkpoint was an unmatched one-update engineering smoke, and
the later matched-arm attempts diverged during the pre-treatment warm-up because
separate GPU contact executions were not bitwise paired.

## What the primary methods actually solve

### Open-DiffLoco and JAVE

The [Open-DiffLoco paper](https://arxiv.org/html/2608.02069) trains a Unitree Go2
for command-velocity locomotion in MJX with SHAC. Its main deployed task has no
motion reference. The policy runs at 50 Hz while physics runs at 250 Hz. JAVE
adds a learned critic-observation dynamics model and supervises the critic's
input Jacobian with a differentiated one-step Bellman target.

This matters because SHAC's terminal critic contributes to the actor gradient
through the critic Jacobian at the truncation boundary. The paper describes
JAVE's benefit as improved **early** gradient stability and support for a longer
analytical horizon; it explicitly says the gain is not a decisive final-policy
improvement. JAVE does not add actor state, discover contact modes, or remove
conflicting actor updates.

### DiffMimic

[DiffMimic](https://arxiv.org/html/2304.03274) is a different optimization
topology. It trains a 13-link, 34-DoF animation humanoid in Brax at 480 physics
steps per second by directly minimizing full-trajectory position, rotation, and
velocity error. It does not use SHAC's learned terminal critic.

Its key long-horizon device is Demonstration Replay: after rollout error becomes
large, the simulated state is replaced by the reference state at the same time,
with the next segment optimized from that anchor. The appendix also reports a
large benefit from random reference-state initialization. The evaluated source
clips are short, mostly cyclic motions; the paper notes that arbitrary and more
complex interactions remain open.

### Gradient limits

[Suh et al.](https://proceedings.mlr.press/v162/suh22b.html) show why successful
differentiable examples do not imply that every contact-rich task has a useful
first-order gradient: stiffness and discontinuities can bias or destabilize the
analytical estimator. That limitation is directly relevant to the G1 stance
exchange where our policies fail.

## What has been established locally

| Question | Evidence | Status |
| --- | --- | --- |
| Does native DiffMimic work in its intended backend? | E-20260710-002 completed all 599 frames on all 32 deterministic Brax rollouts from the saved-best seed-0 checkpoint. | Yes, seed-0 setup gate. Confirmatory seeds remain pending. |
| Can the shared Open-DiffLoco SHAC machinery learn a humanoid controller in MJX? | E-20260810-000 completed 8M finite transitions and improved command tracking. | Yes as a runtime/learning positive control, but the actor mostly stood and did not prove commanded humanoid walking. |
| Can PPO track a G1 reference and execute it in exact MJX? | E-20260813-020 completed the 125-frame reference in Isaac Sim; untouched E-20260813-023 completed MJX suffixes 124/99/74/49/24. | Yes for the registered 125-state walk. |
| Can differentiable training learn a useful G1 correction? | Retained E-20260826-002 reaches 136/144/84/90/79 on the longer phase grid; E-20260825-004's learned torso wrench completes 271 transitions. | Yes, but not as a complete unassisted joint policy. |
| Does training longer finish the walk? | Later E002 checkpoints and numerous earlier continuations improve some starts and damage others. | No. |
| Is reward saturation the sole problem? | Quadratic root credit reaches 134/97/132/112/129, strongly improving late starts while damaging phase 25. | No. |
| Is frozen-parent preservation sufficient? | A bit-exact frozen parent plus zero-head residual still redistributes phase competence (E-20260826-004). | No. |
| Does DiffMimic-style replay solve G1? | Sparse threshold replay first improves safely, then full-budget checkpoints reach mixed phase-local gains and no componentwise-safe actor (E-20260821-008/009). | Helpful curriculum, not a solution. |
| Does noisy reference-state initialization solve it? | E-20260814-025 remains finite but reaches only 63/99/62/49/24 at update 128. | No under the tested recipe. |
| Can a local recovery expert learn the missing action? | E-20260815-038 reproduces local H32 recoveries; global application and transfer variants regress other states. | Locally yes; globally no. |
| Does JAVE execute on the current G1 route? | The active smoke has finite learned-dynamics loss 184.915, gradient-Bellman loss 0.1853, target norm 0.5298, 12,288 replay rows, and fully finite actor gradients. | Yes, engineering feasibility only. |

## What remains unproven or not done

1. No registered, causally paired G1 JAVE experiment exists. The current outputs
   are explicitly engineering smokes.
2. The strongest registered PPO proof is the 125-state walk. Local code and
   artifacts connect PPO to the LAFAN source motion, but a fresh hash-bound PPO
   phase grid under the retained 271-transition evaluator is not registered.
3. JAVE has not been tested from fresh initialization, where the paper claims
   its main early-training benefit. The current smoke resumes a late retained
   policy.
4. A contact-mode-gated or factorized residual actor has not been tested at the
   retained E002 boundary. Neither has an explicit reference-contact-mode input
   been isolated there.
5. The original full-trajectory DiffMimic optimization topology has not produced
   a successful current-G1 result. What was tested successfully on G1 was its
   replay idea inside SHAC, not a critic-free full-trajectory reproduction.
6. The canonical stance-foot world-velocity treatment named in the current lab
   decision has not yet been run.

## JAVE implementation

The implementation is opt-in and preserves the existing actor and critic
checkpoint schema when disabled.

- `src/algorithms/jave/gradient_bellman.py` implements auxiliary-observation
  normalization, learned residual dynamics, the differentiated Bellman target,
  and gradient-Bellman loss.
- `src/envs/g1_tracking/environment.py` exposes a reward-sufficient JAVE
  observation without changing the existing 286-wide critic observation. The
  JAVE observation adds one action-independent tracking-reward feature; action
  regularization is reconstructed analytically.
- `src/algorithms/shac/algorithm.py` integrates replay, learned-dynamics updates,
  bounded JAVE batches, telemetry, resume-relative warm-up, and the weighted
  critic-gradient objective.
- `tools/run_g1_jave_continuation.py` pins the retained E002 checkpoint, model,
  reference, solver, and effective-512/H24 execution contract.

Commits:

- `ac5eec2fca8dd820f2f647faa330e5747cd9cd39` — G1 JAVE path.
- `c13f25bfe4b9ec1d508780e931e53a7bd20282ab` — matched auxiliary collection
  graph for control and treatment.
- `9bdfd5a6b5317f021bd74edfda8921094f0b1670` — JAVE weight made a dynamic state
  value so both arms compile the same executable graph.

The focused verification passed 42 tests after the final parity change; the
larger initial verification passed 51 tests. Ruff, Python compilation, direct
reward reconstruction, and direct JAVE math checks also passed.

## JAVE smoke result and why it is not causal

The first active one-update treatment checkpoint evaluates to
`113/119/68/90/75`; retained E002 is `136/144/84/90/79`. A separately launched
bootstrap-only control is `115/118/64/88/75`. The treatment therefore does not
componentwise preserve either E002 or that control. Because those arms compiled
and executed different graphs, this is only an engineering observation.

Two repairs made the collection graph equal and then made the JAVE weight a
runtime scalar so both arms used the same HLO. Even after the latter repair, the
inactive warm-up checkpoint differed across the two separate GPUs:

- PRNG key and actor normalizer: exact match;
- actor parameters: max absolute difference `0.00121049`;
- simulator qpos: max absolute difference `2.14907`;
- simulator qvel: max absolute difference `38.7005`.

At the second checkpoint the control has JAVE inactive with diagnostic
gradient-Bellman loss `0.19593`; the treatment has JAVE active with loss
`0.17108`. Those later differences cannot be attributed to JAVE because the
states had already diverged before activation.

This does **not** show that JAVE itself is nondeterministic or harmful. It shows
that separate-GPU, cross-process execution is not an adequate bitwise pairing
method for this chaotic contact rollout.

## The likely missing ingredient

The highest-probability explanation is conditional specialization plus safe
consolidation across contact modes, not lack of a stronger global gradient.
Several independent treatments find a correction that helps a subset of
states. The same parameterized actor then damages another subset, including
when the parent is frozen and only a new residual is trained. This is consistent
with similar actor inputs requiring materially different corrections around
opposite stance exchanges or accumulated recovery states.

JAVE could still help if an inaccurate terminal-critic Jacobian is causing those
bad updates, but there are two important limitations:

1. Retained E002 uses zero actor bootstrap, so its critic Jacobian does not
   influence the actor. The JAVE smoke necessarily enables bootstrap in both
   arms; it tests reopening critic-boundary credit plus JAVE, not a pure E002
   continuation.
2. JAVE changes critic geometry, not actor observability or expert routing. It
   cannot by itself distinguish aliased contact modes.

## Recommended next decision

Do not launch a long JAVE run from the current smoke. First close the causal
measurement problem with one preregistered bounded discriminator:

1. Compile one shared executable in one process on one GPU.
2. Execute the common warm-up once, then clone that exact post-warm-up
   `TrainState` as the branch point.
3. Run control (`weight=0`) and treatment (`weight>0`) from those identical
   bytes, with the dynamic scalar as the only treatment difference.
4. Evaluate every bounded checkpoint on the complete phase grid. Stop unless a
   treatment checkpoint componentwise preserves the control and retained E002.

That experiment answers whether late JAVE can help this retained controller. A
separate fresh-initialization experiment would be required to test JAVE's
published early-training claim.

If the bounded JAVE discriminator again produces only phase redistribution,
retire it for this branch and test the more directly supported structural
hypothesis: a zero-initialized, reference-contact-mode-gated residual over
frozen E002, with strict componentwise preservation. That is the smallest
experiment that asks whether the actor must represent different correction
functions on different stance/contact regimes.
