# R1k — Zero-shot leakage audit and seed-paired synthetic baselines

**Status:** implementation committed; synthetic controls locally reproduced; native Brain remains unexecuted.
**Branch:** `research/multimodal-continual-assemblies`
**Date:** 2026-11-23

## Critical flaw found and corrected

The previous `zero_shot_gate.evaluate` created `Dataset(Config(seed=S))`
and passed that exact `Config`, including `seed=S`, to every model
factory. The hidden role permutation is deterministically generated from
`S`. Therefore a model could do this without learning any examples:

```python
class SeedReplay:
    def __init__(self, cfg): self.dataset = Dataset(cfg)
    def train(self, *args): pass
    def predict(self, item, role, cue): return self.dataset.target(item, role)
```

**Verified exploit on the prior interface:** for dataset seed 47, this
replay achieved **100% held-out exact recovery** in BOTH the
`cyclic_shift` and `random_permutation` families. The earlier
"leakage-resistant" description was not defensible.

**Fix:** `Config.seed` now belongs ONLY to private dataset generation.
`Config.model_seed` is a distinct learner-initialization seed (default
1729); `learner_view(cfg)` returns a separate `ModelConfig` whose
`.seed` is the learner seed, not the data seed. Both
`evaluate` and `teacher_control_matrix.run_conditions` pass this
restricted view to unprivileged factories. Public and private task
geometry share one validation owner. Equal private/public seeds are
rejected. Privileged full-Dataset access is restricted to the explicitly
named `KnownPermutationOracle`.

The original seed-replay attack, modified to use the public learner
configuration, returned **0% held-out exact** on the same test seed 47
in both task families. This fixes *accidental API-level disclosure*,
not a security boundary against malicious Python code with
introspection, inspection of the caller, or brute forcing a tiny seed
space. Serious blinded evaluations must run learners in isolated
processes with inaccessible private generation state.

## Local seeded baseline sweep (synthetic only)

`baseline_sweep.py` independently evaluates four TRAIN-ONLY baselines:
a cyclic-shift-hypothesis learner, sparse additive Hebbian pairs,
exact-key pair memory, and input-copy. The task has known
`cyclic_shift` structure; no learner receives the private data seed,
hidden shift parameter, held-out pair, or target during training.

Seed set: `0..11` (12 independently generated datasets).
Geometry: `(n,k)` of `(128,8), (256,8), (256,16), (512,16)`.
Lexical contents: 12. Two roles, balanced checkerboard split.
Each of 4 methods is run for every dataset/geometry pair:
**192 model/seed/geometry evaluations** total. These runs are not
192 independent scientific replications; they share the experimental
design and pseudorandom generation family.

| n | k | Shared-shift exact held-out | Hebbian pair exact held-out | Exact pair memory | Input copy |
|---:|---:|---:|---:|---:|---:|
| 128 | 8  | 100% | 0% | 0% | 0% |
| 256 | 8  | 100% | 0% | 0% | 0% |
| 256 | 16 | 100% | 0% | 0% | 0% |
| 512 | 16 | 100% | 0% | 0% | 0% |

Every row above averages 12 independent dataset seeds. The
shared-shift learner is supplied with the *correct shift hypothesis
class* and is a deliberately strong symbolic control; the plain
Hebbian model learns individual pairs but not that rule.

Mean Hebbian pair held-out OVERLAP (not exact recovery): 0.1606,
0.1198, 0.1363, 0.0877 in table order. These nonzero overlaps are
not evidence of exact compositional transfer.

A critical interpretive detail: the input-copy baseline sometimes
gets positive **rank-1 scores even though exact target recovery is
0%**, so a rank-1-only evaluation is insufficient. Never collapse
exact k-of-n correctness, overlap, and strict positive-margin ranking
into a single headline metric.

## Reproduction and hashes

The code runs without the native library, GPU or GitHub Actions:

```sh
python -m pytest -q research/multimodal/test_zero_shot_gate.py \
  research/multimodal/test_teacher_control_matrix.py \
  research/multimodal/test_baseline_sweep.py

python -m research.multimodal.baseline_sweep \
  --output /tmp/r1k-baseline-12seed.json
```

Local independent source mirror: **48 tests passed** across the
three suites (40 existing/seed-isolation tests, 8 sweep tests).
The complete 192-row JSON was produced locally:

- SHA-256 of canonical rows: `7ba366dbbd5323fd5bcff3341fa461aa0fc4b1c05318c6a6c58d13bb6f6475c0`
- SHA-256 of pretty-printed JSON: `7a8e6a0103de869f8bb5f987c5152957b9358858c67aeeb0d02d56ebee599f21`

**Limitations:** These are synthetic controls, not trained native
assemblies. The local mirror exercised this research subtree, not
the full GitHub checkout. The published run hash should be reproduced
on a full branch checkout before it is cited as final benchmark evidence.
There is no matched memory or computation budget, no rigorous held-out
benchmark curation, and no blinded adversarial execution environment.

## Next gates

1. Run the branch's **complete** `research/multimodal` pytest subtree
   in a full checkout and resolve failures.
2. Run R1b–R1j native Brain protocols with the updated independent
   `model_seed` parameter, recording actual weight changes and read
   immutability.
3. Freeze seed and workload selection before scaling to more seeds and
   model architectures, and implement process-isolated evaluation
   before publishing leakage-resistant claims.
4. Compare native learned binding against the strong shared-shift rule
   baseline; do not claim zero-shot transfer from memorized pairs.
