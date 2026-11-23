# R1m — Train-only surprise quarantine and exponential forgetting

**Status:** synthetic, experimental, locally validated. Not native Brain/neuromorphic/FPGA evidence.
**Branch:** `research/multimodal-continual-assemblies`
**Date:** 2026-11-23

## Experiment

Identical private R1l twenty-round curriculum: 20 independent data seeds, alternating role updates, a hidden rule shift after round 10, and intentionally incorrect teacher labels at rounds 7 and 16. Every learner receives only its current `(role, cue, observed_label)` at the training boundary. All probe cues remain withheld; all readouts are fingerprinted for state immutability.

The **R1m replication invariant passed for all 20 seeds**: training-stream SHA-256, withheld-probe SHA-256, and hidden-rule SHA-256 each match the original R1l campaign. This is the same workload, not an easier resampling.

Three new train-only, hypothesis-aware controls:

- `quarantine2`: do not change an established role shift on one surprising example; switch only after two matching contradictory training examples.
- `decay_0.5`: exponentially discounted evidence with per-role factor 0.5.
- `decay_0.8`: retain more historical evidence with per-role factor 0.8.

## Measured seed-averaged exact held-out recovery

| Model | Rounds 1–10 | Rounds 11–20 | At round 16 | At round 20 |
|---|---:|---:|---:|---:|
| Quarantine, 2 confirmations | 95.0% | 75.0% | 100% | 100% |
| Decayed evidence, gamma=0.5 | 86.0% | 81.0% | 50% | 100% |
| Decayed evidence, gamma=0.8 | 95.0% | 54.5% | 52.5% | 100% |
| Prior R1l last-example-only | — | 85.0% | 50% | — |
| Prior R1l sliding 3 examples | — | 79.5% | — | — |
| Prior R1l all-history | — | 15.5% | — | — |
| Prior R1l pairwise Hebbian | — | 0% | — | — |

A faster-adapting last-example learner remains superior in **average** post-change accuracy to the three new policies. The quarantine policy instead protects against a single poisoned label once stable, at the expense of slower adjustment after genuine drift. This exposes a real speed-vs-robustness tradeoff in this particular synthetic setting.

**Do not confuse the dataset’s ground-truth rule change with something available to the learner.** Only the evaluator knows the switch. The model does not receive it. These algorithms do, however, explicitly assume the hypothesis class is cyclic shifts; they do not discover general compositional structure.

## Validation and provenance

`pytest -q research/multimodal/test_continual_twenty.py research/multimodal/test_robust_shift.py`: **32 tests passed** locally.

The full native `neural_assemblies` checkout, compiled Rust extension, and GPU/FPGA hardware were not accessible in this sandbox. No native performance, compositionality, or retention improvement is claimed.

The full raw JSON contains per-seed per-round records. The adjacent `.xz.base64` file is the exact raw JSON compressed with LZMA and base64-encoded for a text-only Git commit. Verify after decoding:

```sh
base64 -d research/multimodal/evidence/r1m-20rounds.json.xz.base64 | xz -dc > r1m-20rounds.json
sha256sum r1m-20rounds.json
```

Expected SHA-256: `8018bf2c4bab9671fd05ecb337b52f677009e32fd49facf1acf85f542406ef6a`.

Re-run from the branch checkout:

```sh
python -m pytest -q research/multimodal/test_continual_twenty.py research/multimodal/test_robust_shift.py
python -m research.multimodal.robust_shift --output /tmp/r1m-20rounds.json
```

## Next decisive gate

Reproduce on a full checkout; add train-only change-point inference with unknown contamination rates and multiple switch schedules, and compare under a predeclared evaluation grid. Move the same prequential protocol to the real native Brain only after its R1b–R1j correctness and read-only isolation checks pass. Do not tune hyperparameters directly on the fixed hidden test probes.
