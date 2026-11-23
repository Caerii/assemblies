# R1n — Unknown change points, paired corruption, and streaming adaptation

**Status:** Synthetic benchmark implemented, locally executed and replay-verified. **Not** a native `Brain`, Rust, NEMO, vision-language, or neuromorphic-hardware result. 2026-11-23.

**Canonical branch:** `research/multimodal-continual-assemblies`.

## Research question

What is the stability–plasticity tradeoff when two role-specific cyclic mappings may change without advance warning, and the teaching stream may contain either isolated or consecutive misleading labels? Does a learner resistant to one corrupted observation retain that advantage when the misleading observations arrive repeatedly?

This study extends R1m without selecting learning rates or confirmation thresholds on withheld probes. Every model knows the **cyclic-shift hypothesis family**, so this measures adaptation policies under that inductive bias—not general zero-shot compositional learning.

## Frozen experiment matrix

- Geometry: `n=128, k=8`, `SUBJ` and `OBJ`, five withheld probes for each role.
- Training: **48 rounds** per trial; the trained role alternates each round.
- Seeds: `0..19`; each seed is paired across all four scenarios.
- Scenarios: stationary/clean; three hidden rule changes at rounds **13, 27, 39**; the same changes plus isolated corruptions at **7, 18, 32, 44**; or the same changes plus consecutive same-role corruptions at **18, 20, 22, 24**.
- The evaluator privately determines the current role-specific shift and the corruption schedule. Neither the model's `train` nor its `predict` method receives the phase, change flag, corruption flag, shifts, or probe labels.
- Dedicated deterministic RNG streams govern rule shifts, withheld probes, and training cues. Across all four scenarios for the same seed, **training cue SHA-256, probe SHA-256 and hidden-rule commitment SHA-256 are identical**. Only observed teaching labels change.
- Eight fixed control policies: all-history, most-recent-only, last-three, two-confirmation quarantine, exponential evidence decay (`γ=0.5`, `γ=0.8`), an exploratory dual-timescale confirmation learner, and nonstructural pairwise Hebbian association.

**Total:** 4 scenarios × 20 seeds × 48 rounds = **3,840 sequential training updates per model**, or **30,720 model-training calls** over eight models. Evaluations use held-out probes after every update. These units are not independent statistical replications.

## Exact held-out recovery, rounds 13–48 (mean over 20 paired seeds)

| Method | Clean drift | Isolated label corruption | Consecutive same-role corruption |
|---|---:|---:|---:|
| Last example | 95.83% | 87.50% | 84.72% |
| Last three | 79.17% | 79.17% | 70.42% |
| Quarantine (2 confirmations) | 79.17% | 79.17% | 69.44% |
| Exponential decay 0.5 | 89.44% | 82.78% | 79.03% |
| Exponential decay 0.8 | 62.50% | 54.86% | 52.64% |
| Dual-timescale (decay + confirmation) | 72.78% | 68.47% | 62.92% |
| All-history | 8.89% | 8.96% | 6.60% |
| Pairwise Hebbian | 0% | 0% | 0% |

**Observation:** two-confirmation quarantine does not lose any exact accuracy under the isolated-corruption schedule in this test, whereas its post-change accuracy drops by **9.72 percentage points** under the consecutive same-role schedule. The one-example model adapts fastest to changes but loses **8.33 percentage points** from isolated corruption. This is a specific, constructed failure mode of confirmation-based filtering, not a universal result about noise robustness.

The same-seed, same-probe comparison is essential. 

## Paired corruption penalties (post-change exact accuracy)

Values are percentage-point differences from the corresponding **clean-drift** stream. Intervals resample **20 paired seeds** (3,000 bootstrap replications, deterministic resampling seed 1791). They are **descriptive**, not population-level significance claims: the corruption schedule, hypothesis class and parameter values were chosen by the experimenter, and several policies have identical per-seed effects by construction.

| Policy | Isolated corruption | Consecutive same-role corruption |
|---|---:|---:|
| Last example | −8.33 pp | −11.11 pp |
| Quarantine (2) | 0.00 pp | −9.72 pp |
| Decay 0.5 | −6.67 pp | −10.42 pp |
| Dual-timescale | −4.31 pp | −9.86 pp |

**No policy won on every criterion.** The last-example learner is often fastest under abrupt drift; quarantine provides a narrower protection against isolated wrong examples; sustained incorrect examples can make the quarantine promote a wrong rule. The exploratory dual-timescale policy was not better than established baselines in overall post-change accuracy.

## Reproduction, validation and artifact provenance

```
python -m pytest -q \
  research/multimodal/test_unknown_drift.py \
  research/multimodal/test_analyze_unknown_drift.py \
  research/multimodal/test_verify_unknown_drift.py \
  research/multimodal/test_continual_twenty.py \
  research/multimodal/test_robust_shift.py

python -m research.multimodal.unknown_drift \
  --output /tmp/r1n_20seed.json
python -m research.multimodal.analyze_unknown_drift \
  /tmp/r1n_20seed.json --output /tmp/r1n_analysis.json
python -m research.multimodal.verify_unknown_drift --replay
```

- Local source suite: **66 tests passed** (including regression checks from R1l/R1m).
- Full 80-trial JSON SHA-256: `ebfc57c6c6c0580d3933707b6445110ebbeacf86c1cc1b364491e685eeafbd06`.
- Separate analysis JSON SHA-256: `ac28276274c67f15d8a7196d740ef551f07e6ba12254b1ff0253c16f74ea9dd4`.
- The exact raw result is stored in Git as `evidence/r1n-20seed.json.xz.base64` (LZMA then Base64). `verify_unknown_drift.py` decodes it, checks the SHA, validates paired streams, and can regenerate and compare **all 80 trials**. The local replay gate passed.
- Environment used for these local tests: Python 3.13, NumPy 2.3.5, pytest 9.0.2. Native `neural_assemblies` package and Rust toolchain were absent; a full Git checkout could not be cloned because the local runtime could not resolve GitHub. This does not change the GitHub connector's write access.

## Interpretation limits and next necessary experiments

1. **Independent corruptions and timing**: schedules are fixed and visible to an external evaluator. For a more defensible robustness conclusion, preregister stochastic contamination schedules with held-out seeds and blinded learner execution.
2. **Hypothesis class**: shift-aware learners can only infer integer cyclic shifts. No evidence for general language grammar, visual grounding, causal world models, or emergent compositionality follows.
3. **Capacity and compute**: the baseline models have different storage and cost. Future sweeps must log model bytes, training latency and inference latency, and equalize cost budgets.
4. **Native integration**: after R1b–R1j native correctness gates pass, repeat the same prequential schedule using an actual native assembly learner; preserve paired controls, positive read margin and non-mutating readout. Failure is informative and must not be hidden.
5. **Avoid held-out tuning**: do not change `γ`, confirmation count or scenario timing based on the reported test probes, then claim the new configuration was preregistered.
