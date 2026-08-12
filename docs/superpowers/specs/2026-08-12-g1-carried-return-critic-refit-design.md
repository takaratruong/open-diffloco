# G1 Carried-Return Critic Refit Design

## Purpose

Repair the concrete long-credit defect measured by E020 without changing the
actor. The exact E012 target critic has rank correlation `0.469` and NRMSE
`1.686` against realized carried terminal returns, and all five H12-boundary
errors exceed `0.93`. Another actor update cannot be interpreted until its
terminal target is improved.

## Treatment

Freeze the exact E012 TrainState actor, actor optimizer, actor normalizer,
environment, reference, reward, and solver. Collect replay-free nominal carried
trajectories under that actor from three disjoint phase sets:

- fit: `10,30,...,390`;
- validation: `20,120,220,320,380`;
- final test: `0,100,200,300,400`.

For every pre-step state, store the training-identical critic observation and
the realized gamma-0.99 return through the natural terminal transition. Fit all
critic parameters by MSE for one fixed 2,000-step continuation using its existing
Adam state and learning rate `5e-4`; inspect validation every 20 steps and select
the checkpoint with minimum validation NRMSE, breaking ties by higher rank
correlation and then earlier step. The test split cannot select a checkpoint.

Set both `critic_params` and `target_critic_params` to the selected critic and
retain its continued optimizer state. Every non-critic TrainState leaf must be
byte-exact to E012. Publish the refit checkpoint only if held-out validation and
test improve over the original target critic.

## Gates

The refit advances only if validation and final-test rank correlation are at
least `0.8`, NRMSE is at most `0.25`, all five final-test H12 relative errors are
at most `0.25`, and the original actor/actor optimizer/normalizer/environment
state remain exact. Also report the original current critic and lagged target
critic on identical captured observations; if merely synchronizing the current
critic already clears the gates, do not optimize and select that smaller
treatment.

Every rollout must naturally terminate with finite state, observations, action,
reward, and return and exact-zero external wrench. Code, checkpoint, reference,
model, controller, solver, seed, phase splits, step count, optimizer, and output
hashes are immutable.

## Interpretation

Passing proves only scalar held-out calibration for this actor's carried state
distribution. A separately registered actor continuation must still test whether
the repaired critic supplies a useful policy gradient. Failure closes simple
supervised critic refitting and points to value-gradient or objective design,
not more reset noise or identical training.
