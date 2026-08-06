# G1 Full-Policy Differentiable Fine-Tuning Plan

1. Add an exact trainable RMR policy application whose normalization is frozen
   and whose weights and biases remain differentiable.
2. Add focused tests for initialization parity and gradient ownership.
3. Transport the initialized full actor through the registered G1 SHAC runner,
   reject incompatible residual or task combinations, and record actor kind.
4. Add standalone full-actor checkpoint evaluation and focused CLI tests.
5. Run the focused test suite and a guarded one-update finite-gradient smoke.
6. Preregister and run the fixed full-policy fine-tuning experiment.
7. Evaluate the selected checkpoint against source across four phases and
   render visible evidence.
8. Curate code, artifacts, experiment cards, current state, and Obsidian views.

