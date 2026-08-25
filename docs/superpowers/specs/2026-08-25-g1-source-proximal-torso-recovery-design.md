# G1 source-proximal torso recovery

## Question

Can the direct torso-attitude objective reduce E026's backward lean without
forgetting the already competent short-walk behavior?

## Design

Resume the exact E026 update-16 checkpoint and keep the E001 torso objective.
Replace only the existing residual-to-parent action anchor with an action-space
proximal anchor to the exact resumed E026 source policy.  The source checkpoint
path and SHA-256 are explicit immutable inputs.  Candidate and source actions
are evaluated on the same current observation; gradients stop at the source.
The existing weight-one quadratic anchor is retained, so this test changes the
anchor target rather than introducing another tuned coefficient.

The implementation must default to the legacy parent anchor, reject a source
anchor without the residual adapter, reject a wrong source hash or parent
mismatch, persist the source provenance, and remain deterministic across
resume.

## Decision gate

Evaluate checkpoints every eight updates on exact phases 0/25/50/75/100.
Select only a checkpoint that preserves E026 survival 124/99/74/49/24 and
reduces phase-zero last-ten absolute torso pitch by at least 25 percent without
increasing its maximum.  If the torso correction remains but competence is not
preserved, a scalar proximal penalty is insufficient and the next method must
use a hard constrained/projected update.
