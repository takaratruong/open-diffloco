# MJX Pytree Receipt Hashing

- Scope: hash-bound identity receipts for saved JAX/MJX training state.
- Read when: comparing a loaded `EnvState`, creating a synthetic checkpoint, or
  publishing its adjacent `hparams.json` under an exact SHA-256 contract.
- Last verified: 2026-09-02.
- Evidence: `E-20260901-002` and `E-20260901-003` failure receipts; focused
  regressions in `tests/test_g1_fixed_batch_distribution_audit.py`; successful
  paired execution `E-20260901-004` on source `994c677`.

## Current facts

- `tools.evaluate_g1_e038_recovery_transfer.parameter_tree_sha256` includes
  `repr(treedef)`. Do not use it for an MJX `EnvState`: MJX `Contact`
  custom-node auxiliary `_NumPyArrayHashWrapper` objects include fresh Python
  memory addresses in that representation, so repeated hashes of unchanged
  leaves can differ.
- For a common-state receipt, hash the root type plus ordered leaf paths, leaf
  count, dtypes, shapes, and bytes. Compare both states inside the same process
  and separately validate their expected concrete state type. Keep ordinary
  file SHA-256 as the artifact-identity authority.
- Parameter-only trees without address-bearing custom-node auxiliary data can
  continue using the established parameter-tree hash, but a new tree type must
  be checked before assuming its treedef representation is stable.
- If a checkpoint sidecar is required to retain the source SHA-256, copy its
  bytes atomically. Parsing and reserializing equivalent JSON changes the file
  hash and violates the receipt contract.

## Gotchas

- Tree equality and leaf-byte equality can both hold while `repr(treedef)`
  differs.
- A stable within-process leaf receipt is not a replacement for the exact
  checkpoint-file SHA-256 and should not be relabeled as one.
- Validate exact-byte copying with a deliberately noncanonical JSON fixture so
  a formatting rewrite cannot accidentally pass.

## Stale when

This note is stale if the shared parameter-tree hash stops using treedef repr,
MJX changes `Contact` pytree auxiliary metadata to a stable representation, or
the checkpoint/sidecar receipt contract is replaced. Re-run the address-bearing
custom-node regression and one real saved-state repeat before updating it.
