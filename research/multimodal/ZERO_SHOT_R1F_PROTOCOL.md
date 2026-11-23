# R1f — Independent-target zero-shot gate and learned controls

**Status:** controlled synthetic evaluation implemented; native Brain integration NOT demonstrated.
**Date:** 2026-11-23.
**Entrypoint:** `python -m research.multimodal.zero_shot_gate`.
**Primary question:** Can a method infer the role-conditioned transformed assembly
for a content/role pair it has NEVER trained on?

## Test construction

Generate `contents=12` independent lexical k-of-n sparse codes with n=512,
k=16. For each of two roles, construct an independently seeded full bijection
over neuron IDs, fixed for the entire experiment. The target code is
`sort(permutation_role[lexical_code])`. Define the checkerboard split so
every item is observed in one role and withheld from the other; both roles
have 6 training examples, 6 withheld examples. This prevents held-out lexical
identity or missing-role novelty from explaining failure.

A model receives cue, role and (for exact-key controls) item index during
prediction. The **correct output is not the input identity**, but the
independently generated target sparse assembly. All true held-out target
codes are withheld from the model's `train` method; oracle access to the
role permutation is explicitly privileged, labeled and disallowed for a
learned model.

Strict score: exact k-of-n target recovery; raw overlap; rank-1 against
other role-specific targets, requiring a positive overlap margin.
Ties are not successes. All split definitions and input/target code
SHA-256 hashes are reported.

## Controls

1. **KnownPermutationOracle**: given the exact teacher permutation
   (privileged), should be perfect; this is a correctness upper bound,
   not a learned representation.
2. **ExactPairMemory**: stores only observed (item,role) targets; should
   perfectly retrieve training pairs and return no held-out target.
3. **CopyBaseline**: returns the lexical code unchanged; guards against
   scoring a trivial identity echo as composition.
4. **HebbianRoleMap**: independently learned source→target coactivation
   weights, one table per role, trained only on the checkerboard examples.
   This is a learned, unprivileged local associative baseline.

## Local exploratory observation

On an isolated Python 3.13 / NumPy CPU implementation at seeds
`0,1,17,42` using n=512, k=16 and contents=12:

| Method | Trained exact | Held-out exact |
|---|---:|---:|
| Privileged permutation oracle | 100% | 100% |
| Exact-key episodic pairs | 100% | 0% |
| Sparse Hebbian role map | 100% | 0% |
| Copy input | not guaranteed | 0% |

Mean held-out overlap for sparse Hebbian baseline varied by seed:
`[0.078125, 0.0677083, 0.1197917, 0.0989583]`.
**18 local unit tests passed**, including four-seed learned-baseline checks.
These numbers characterize an engineered synthetic task, not the real
`Brain` model and NOT human language acquisition.

## What the result means

Local Hebbian coactivation *can memorize* examples, but that does not
establish extrapolation of a random role-specific bijection to unseen
source codes. In fact, with few observations a random unknown
permutation is mostly information-theoretically unidentifiable.
This is a **deliberately hard stress test** and may be impossible to
solve from the observations given, even for more expressive learned models.

Consequently this test is a NEGATIVE CONTROL for claims of automatic
compositionality, not a scientifically fair primary generalization
benchmark. A fair next task must use a structured, learnable role
operator (e.g. small shared latent transform with many exemplar
constraints), and report an identifiability upper bound and a
trained-parameter-matched baseline. Do not award architecture credit
for a task whose target mapping has not been inferable from training.

## Next gate

- Add a learnable shared transform family with explicit train/test
  identifiability and an independent held-out target.
- Separate same-role item generalization from unseen role composition.
- Implement a native `Brain` adapter with stable neuron-ID readout;
  do NOT pass the oracle permutation to that adapter.
- Execute full repo tests and R1b/R1c/R1d native experiments before
  claiming a learned zero-shot effect.
- Benchmark time, memory and write/read consistency against the
  fully indexed exact-key baseline and a capacity-matched linear model.
