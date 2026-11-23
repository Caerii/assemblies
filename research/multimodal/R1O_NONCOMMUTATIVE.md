# R1o: Blind atomic learning and noncommutative composition

**Status:** synthetic protocol locally executed, tested, and replay-verified; native `Brain` adapter source/mocks implemented but numerically **unverified**.
**Date:** 2026-11-23. **Canonical branch:** `research/multimodal-continual-assemblies`.

## Scientific question

Can a learner infer TWO reusable operations from separately demonstrated atomic examples, then apply them to novel source assemblies through previously unseen **ordered compositions**? Correct atomic binding alone is not sufficient: the model must distinguish `A→B` from `B→A`, and generalize to depth 3–4 without being shown any composite targets during training.

The private teacher is a bijective affine map over a finite neuron-ID ring:

`T_role(x) = (a_role * x + b_role) mod n`, with `gcd(a_role, n) = 1`.

Each operator is sampled so its multiplier is nontrivial (`a != 1`), and the two operators are explicitly checked to **not commute**. The transform is applied to each element of a k-of-n assembly, preserving size and stable IDs. We fix one private dataset-generation seed per trial and supply a **different public model-initialization seed** to each learner. Teacher parameters and held-out target assemblies are never passed to an unprivileged learner.

This model family is a *deliberately engineered structural prior*, not an emergent neural rule. A learner that already searches affine maps may generalize perfectly even if a generic Hebbian network cannot.

## Protocol

- **Training:** eight atomic demonstrations for A and eight for B, with independent k-of-n source assemblies. No composed trajectories are trained. Each model receives only `(role, cue, target)` for those 16 observations.
- **Held-out atomic:** eight completely disjoint source assemblies evaluated under each individual operation.
- **Held-out composed:** six previously unseen operation sequences (AB, BA, ABA, BAB, AABB, BABA), eight disjoint source cues each. Depth up to four.
- **Strict metrics:** exact winner-set match; fraction overlap; strict positive-margin rank-1 against other test targets; paired AB/BA correct in **both** orders on the same cue.
- **Integrity:** prediction is checked for any change in model state; output must contain precisely k distinct valid IDs. Input/probe hashes and private-operator commitments are recorded.

Control baselines:

1. `AffineRuleLearner`: learned maximum-overlap affine-parameter hypothesis from training examples only. This learner is **given the correct operator family**.
2. `OrderBlindAffine`: learns the same individual rules but sorts operation order at inference, exposing order-discrimination artifacts.
3. `ShiftOnly`: wrong structural family, allowing only additive shifts (`a=1`).
4. `HebbianPairs`: local coactivation per role, sequentially applying each learned top-k projection.
5. `AtomicKeyMemory`: exact-key memory of only the demonstrated single-step pairs.
6. `CopyInput`: no-op control.

## Measured results (20 paired seeds per geometry)

| Method | n=64,k=8 | n=96,k=8 | n=128,k=12 | Both AB/BA correct |
|---|---:|---:|---:|---:|
| Learned affine rule | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| Order-blind affine | 33.3% | 33.3% | 33.3% | 0.0% |
| Shift-only hypothesis | 0.8% | 0.0% | 0.0% | 0.0% |
| Hebbian pair projections | 0.0% | 0.0% | 0.0% | 0.0% |
| Atomic exact lookup | 0.0% | 0.0% | 0.0% | 0.0% |
| Copy input | 2.5% | 0.9% | 0.8% | 0.0% |

Percentages are **exact compositional target recovery** averaged over 20 dataset seeds in each geometry, except the final column, which is the mean ability to get both AB and BA correct on the same withheld cue (all three geometries yielded the displayed 0%/100% contrast).

**Do not say that the negative baselines always achieve zero.** Input copying occasionally reconstructs a composite target by chance; the wrong shift-only model also gets 0.8% on the smallest geometry. For Hebbian pair projections, exact training recovery was 99.4% at n=64 (100% at n=96 and 128), demonstrating that storing atomic associations is still insufficient for these unseen compositions.

The order-blind baseline achieves 33.3% because two of six selected compositions are unchanged by sorting A/B tokens. It cannot correctly recover *both* AB and BA for a given cue. This proves the test penalizes order blindness rather than rewarding commutative composition.

**Important:** `strict_rank1` is often nonzero even when **exact target accuracy is zero**. We keep those measures separate to prevent a partially overlapping answer from counting as successful exact composition.

The three geometries share the same 20 seed identifiers; the experiment contains 60 seed–geometry trials × 6 models = **360 model-cell evaluations**. Their outcomes are paired by seed and source/test stream. This is an exploratory sensitivity study, not independent population replication or publication-grade confidence intervals.

## Local validation / provenance

Executed from `/mnt/data/r1o_work` on Python 3.13, NumPy, pytest CPU:

```sh
PYTHONPATH=. python -m pytest -q research/multimodal/test_*.py
# 37 passed
PYTHONPATH=. python -m research.multimodal.noncommutative_sweep \
  --output /tmp/r1o_20seeds.json
PYTHONPATH=. python -m research.multimodal.verify_noncommutative --replay
# Full 360-cell deterministic replay passed
```

The raw `r1o_20seeds.json` was compressed using LZMA and base64-encoded into `evidence/r1o-20seed.json.xz.base64` for text-only Git storage. Exact reproducibility hashes:

- Full pretty-printed JSON SHA-256: `868f6a45295ebf703fc8c690063242059a717d3de605a564cf955807bccf725b`
- Canonically serialized model-cell list SHA-256: `3bd3d1cb5f3dba7f62ed39b08ed2885a57c678c7833028cd2848c57da84547db`
- Compressed `.xz` SHA-256: `550ca5c8c1dca64d3cf896f8c3e9cd429cf45943c04f60bba34dad27e490690c`

**This does not imply native runtime validation.** The full `neural_assemblies` package is not installed in the sandbox and this environment cannot resolve GitHub for a normal clone. The GitHub connector can commit source files but does not furnish a mounted working tree.

## Optional native adapter — R1o-N

`native_affine_chain.py` implements the same `train(role,cue,target)` / `predict(roles,cue)` surface using the already established R1h `NativeTeacherDriven` and its explicit Brain synaptic backend. It teaches *only atomic demonstrations*, then applies each role-specific Brain projection sequentially with `read_only` and `frozen` guards. Unit tests exercise scheduling with a fake Brain adapter. **Native numerical outcomes are not known** until the following actually runs in the full checkout:

```sh
python -m pytest -q research/multimodal/test_noncommutative*.py \
  research/multimodal/test_native_affine_chain.py
python -m research.multimodal.native_affine_chain \
  --n 64 --k 8 --atomic-examples 8 --test-cues 8 --data-seed 23
```

A native implementation that memorizes the atomics but fails composed tests is scientifically useful negative evidence. Compare against the strong affine hypothesis learner; do **not** conflate performance of the symbolic control with native assembly generalization.

## Next falsification gates

1. Execute the full branch-local unit suite in a native checkout. Resolve any native API failures and run the R1h/R1i synaptic-mutation audits before interpreting the native compositional output.
2. Run the native R1o-N adapter across the same **predeclared** n/k/seeds and record exact, margin, read-only state, runtime, and memory. Native does **not** receive hidden operator parameters.
3. Require both AB and BA correct, not only aggregate top-1. Increase composition depth to stress degradation when per-step projection errors compound.
4. Add inductive bias ablations and strong learned operator alternatives. Cost-match inference and memory before claiming superiority over retrieval or compiled symbolic rules.
5. Only after native semantic validation, port an equivalent primitive to Rust and measure faithful end-to-end speedups instead of speculative FPGA projections/sec.
