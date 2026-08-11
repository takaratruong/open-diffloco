# Frozen Delta-Preview Design

## Scientific question

Does representing the +4/+8/+12-frame joint preview as future-minus-current
reference deltas improve replay-free G1 dance survival over the raw absolute
preview, when the parent checkpoint, optimizer, frozen boundary, batch,
horizon, objective, randomization, and training exposure are unchanged?

E-20260811-006 rejects a fixed termination-margin objective as sufficient.
The remaining representation concern is concrete: the current actor receives
174 absolute future joint position/velocity values even though the current
58-value reference command is already present. Only the newest preview rows
are trainable, so asking a frozen trunk to discover all three differences may
be unnecessarily ill-conditioned.

## Design

Add an explicit `actor_reference_preview_mode` contract with two values:
`absolute` preserves every existing path and remains the default; `delta`
returns each future 58-value reference command minus the current 58-value
reference command. The observation dimensionality, offsets, ordering,
normalization procedure, append-only migration, and trainable mask remain
unchanged.

Thread the mode through the G1 environment, SHAC training metadata, safe resume
validation, immutable experiment runner, and no-render evaluator. A legacy or
raw-preview checkpoint is interpreted as `absolute`. A change of mode on a
checkpoint that already has preview inputs is rejected; `delta` is authorized
only while upgrading a no-preview parent.

The treatment resumes the exact E008 no-preview parent at step 1,179,648 and
repeats the E001/E003 contract: 64 effective-512 H12 updates, five-bin CAGrad,
zero termination-margin weight, frozen normalizer, and only the 89,088 newest
preview-to-first-layer weights trainable. It archives every eight updates.
Zero-initialized append-only rows must preserve the parent action to 1e-7 before
the first update.

## Evaluation and decision

Evaluate all eight checkpoints replay-free from phases 0/100/200/300/400 and
select by maximum minimum, median, mean, then earliest step. Advancement
requires the selected vector to preserve floors 23/50/72/16/40 and
lexicographically exceed E001's raw-preview key 27/54/53.0. Training reward
cannot select a checkpoint.

If delta preview advances, retain it for the next motion. If it completes
cleanly without advancing, reject input conditioning as the primary bottleneck
and test a zero-initialized nonlinear residual adapter with greater capacity.
Invalid migration, resume, frozen-state, telemetry, or evaluation evidence is
an execution defect and receives no scientific interpretation.

## Verification boundary

Tests must prove default absolute behavior is unchanged, delta values and end
clamping are exact, invalid modes fail, resume metadata cannot silently change
an existing preview layout, the runner changes only preview mode and checkpoint
cadence from E001, evaluator artifacts record the mode, and migration retains
action equivalence.
