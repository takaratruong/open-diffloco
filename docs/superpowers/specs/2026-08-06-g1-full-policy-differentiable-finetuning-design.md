# G1 Full-Policy Differentiable Fine-Tuning Design

## Goal

Replace the bounded residual diagnostic with a standalone G1 tracking actor
whose complete neural-network weights and biases are optimized through MJX
rollouts. Preserve the working RMR actor only as initialization and as the
fixed comparison baseline.

## Scientific boundary

- Initialize the trainable actor exactly from the registered RMR checkpoint.
- Freeze only the source observation-normalization statistics.
- Optimize every actor weight and bias with the existing SHAC simulator
  gradients.
- Use the validated 50 Hz G1 task, strict terminations, solver budget 4/5,
  256 environments, horizon 12, zero action noise, zero critic bootstrap, and
  the already validated reward/observation definitions.
- Produce a self-contained checkpoint that evaluates without calling the
  source RMR actor.
- Compare the selected full actor against the original source in one process
  across phases 0/30/60/90 and render the selected actor visibly.

The existing bounded residual result remains diagnostic evidence that the
humanoid contact-gradient path can preserve the walking basin. It is not the
target method or final controller.

## Implementation

The existing `RmrPolicy` pytree already contains the exact source mean,
standard deviation, weights, and biases. A trainable application function will
stop gradients only through mean and standard deviation while allowing
gradients through every weight and bias. SHAC will accept this pytree as the
initial actor parameters and use it directly in actor rollouts. The checkpoint
therefore contains a complete standalone actor and requires no architecture
conversion.

Evaluation receives an explicit full-RMR-actor flag, loads the actor pytree
from the SHAC checkpoint, and applies it directly to raw task observations.
Residual composition is not involved.

## Gates

1. Exact initialization: the trainable actor matches source actions before any
   update.
2. Trainability: gradients are finite and nonzero for actor weights/biases and
   exactly zero for frozen normalization statistics.
3. Execution: a guarded one-update run compiles and produces finite gradients
   and checkpoint leaves.
4. Preservation: the selected standalone actor completes the strict rollout
   without additional terminals and remains visibly aligned to the reference.
5. Improvement: paired same-process evaluation reports reward and all six
   tracking-error deltas against the fixed source. Improvement is a measured
   result, not assumed from preservation.

