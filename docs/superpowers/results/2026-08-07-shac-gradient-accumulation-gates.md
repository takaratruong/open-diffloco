# SHAC Gradient Accumulation Gates

## Outcome

The sequential accumulation implementation is admitted for physical-256 G1
learning at factors two and four. Both one-update production gates completed
on one L40S with fully finite actor and critic gradients and checkpoints.

| Factor | Physical envs | Effective envs | Step | Compile | Update | Result |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 256 | 512 | 6,144 | 868.7 s | 21.0 s | pass |
| 4 | 256 | 1,024 | 12,288 | 876.1 s | 41.6 s | pass |

The factor-two checkpoint contains 512 environment states and all 214 numeric
leaves among its 215 total leaves are finite. Its actor and critic gradient
finite fractions are both 1.0. Factor four reports the same finite fractions.

## Small changed-width diagnostic

The planned monolithic-four versus physical-two-by-two comparison is not a
valid implementation-parity oracle for this contact-rich MJX task. With the
exact 512-by-512 random-head actor, horizon 12, seed zero, and identical
effective population, both executions remained finite and advanced 48 steps,
but changing physical `vmap` width changed the trajectories and gradients:

| Tree | cosine | relative L2 |
|---|---:|---:|
| actor parameters | 0.999779460274 | 0.02100 |
| critic parameters | 0.999983018465 | 0.005828 |
| actor optimizer | 0.428114271647 | 0.9907 |
| critic optimizer | 0.722183739542 | 0.6920 |
| actor normalizer | 0.998127614013 | 0.06404 |
| environment state | 0.863246751519 | 0.5123 |

The result must not be converted into a pass by tuning tolerances. Production
accumulation instead keeps every differentiation shard at E098's proven
physical width of 256. Its implementation authority is the exact clipped
shard-mean tests, factor-one compatibility, and the two finite production
gates above.

## Decision

Run the preregistered 128-update factors two and four concurrently. Select the
smallest factor whose fixed-update checkpoint exceeds E098's strict 55-of-60
phase-zero survival. If neither exceeds 55, reject effective-batch scaling as
the next mechanism rather than testing factor eight.
