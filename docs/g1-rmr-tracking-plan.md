# G1 RMR tracking implementation plan

The first deliverable is a visible, replay-free G1 motion-tracking policy. A
collocation comparison begins only after this baseline passes.

1. Pin the RMR G1 model, 121-frame grounded reference, controller arrays, 14-body
   ordering, reward weights, and control/reference time base.
2. Test and implement the six rigid-body reward terms independently of the
   environment.
3. Precompute the reference body's world poses and velocities through the same
   MuJoCo model used for training.
4. Implement random reference-state initialization, phase advancement, the RMR
   deploy observation, and a privileged tracking observation for the critic.
5. Require exact-reference reward, isolated-perturbation, phase, time-base, and
   finite nonzero MJX action-gradient gates.
6. Adapt SHAC only at the environment/metrics boundary; retain its differentiable
   first-order actor update and learned terminal-value bootstrap.
7. Run a short one-seed learning discriminator, compare the identical initial and
   final policies on fixed held-out phases, and render both robot and reference.
8. If learning is real but incomplete, localize one failure at a time while
   holding the validated task geometry fixed. If the task or gradient gates fail,
   repair them before training.
