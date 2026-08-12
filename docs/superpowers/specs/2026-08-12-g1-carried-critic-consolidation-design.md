# G1 Carried-Critic Consolidation Design

E021 established a strong but incomplete critic repair: validation/test rank
rose above 0.91 and four of five canonical H12 errors passed, while the fit split
was nearly exact. This successor freezes E021's selected recipe rather than
tuning it: restart from exact E012, train the critic for exactly 1,640 continued
Adam steps on all 30 E021 trajectories, then evaluate once on previously unseen
phases `5/105/205/305/405`.

The E021 dataset, parent checkpoint/hparams, code, model, controller, reference,
solver, seed, optimizer, step count, and confirmation grid are hash-pinned. The
actor and every non-critic TrainState field stay exact. The confirmation grid
cannot affect fitting or selection.

Publish a consolidated checkpoint only if confirmation rank correlation is at
least `0.8`, NRMSE at most `0.25`, every H12 relative error at most `0.25`, and
rank correlation, NRMSE, and every H12 error strictly improve over the original
target critic on identical confirmation trajectories. Pearson, RMSE, and bias
remain descriptive. Otherwise close simple dataset consolidation and move to
value-gradient/objective design.
