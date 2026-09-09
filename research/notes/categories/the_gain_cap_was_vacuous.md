# The sweep axis never reached the mechanism — GAIN_MAX is vacuous and the "gain" was mostly familiarity suppression

**Task #132 (E3). Experiment: `research/experiments/gain_max_sweep.py` (pre-registered S1–S5). 6 configs × 10 seeds, 60 cells, ~25 min on the full speed stack (forked pre-stages, pooled, cached).**

## The anomaly, and its diagnosis

All four `BOTH:g` arms (g ∈ {2, 4, 6, 8}) returned **bit-identical** results — number 0.7100 ± 0.0554, tense 0.6937 ± 0.0338, exactly. An unmoved metric across a swept axis means the axis never touched the mechanism (the saturated-metric lesson: identical output justifies "the decision did not change," so census the mechanism). Instrumented on a cached checkpoint with the cap removed entirely (`gmax=999`):

```
episodes=213  max_gain=1.215  p95=1.177  mean=0.998  frac>2 = 0.000
mean exposure count = 1.41 over 151 (feature, form) keys
```

The registered formula `min(GAIN_MAX, sqrt(mean_count/count))` **self-caps at ~1.2** on this corpus: with mean exposure 1.41, a first-seen form gets sqrt(1.41) ≈ 1.19, and repeated forms get gains *below 1*. Every GAIN_MAX ≥ 1.2 is the same experiment. This is the same shape as the COLT22 beta_0 finding — a bound that is formally present and vacuous at every practical operating point.

**Which reframes E2.** The GAIN arm's +0.04 and BOTH's margin over SCALED were produced by modulations in [~0.8, 1.2] — the mechanism as implemented acts mostly as mild **familiarity suppression** (frequent forms' episodes write slightly weaker), not surprise amplification. The label "surprise-modulated gain" over-described what ran; the composition result stands but its gain component was smaller and different in kind than designed.

## Registered verdicts

- **S1 REFUTED at every g**: best 0.710 (n=10). Note also E2's 0.740 at n=5 regressed to 0.710 at n=10 — the five extra seeds pulled it down, which is what n=5 optimism looks like.
- **S2 REFUTED**: BOTH−SCALED = +0.060 ± 0.065 at n=10 — CI spans zero, one negative seed (−0.05). Per the registered decision rule: **the BOTH-over-SCALED margin is not established; scaling alone is the honest recommendation** until a mechanism change earns more.
- **S3 PASSES**: tense at BOTH is +0.016 vs OFF — the no-tense-cost property of the composition is solid at n=10.
- **S5**: the curve is flat because the axis was dead — reported as such, not as "robust to GAIN_MAX."

## What the next experiment actually is (E4, to be registered separately)

The lever is the **formula**, not the cap. Candidates, each still label-free and self-normalizing: `mean_count/count` (no sqrt — a form seen once among 5×-seen neighbors gets 5×), or `(max_count/count)^alpha`. The sqrt was chosen for gentleness and turned out to choose near-inertness. Separately, the decision rule's other branch stays live: does the composed ceiling move with n? Both are new pre-registrations, not sweep continuations.

## Standing conclusions after E1–E3

1. Frequency swamping is real and mechanistically understood (#129).
2. Scoped homeostatic scaling is the one mechanism with an established margin: +0.15 over OFF at n=10 (0.65 vs 0.53), guards exact. It is the honest current recommendation, still default-OFF pending the tense-cost story at scale.
3. The composition (scaling + per-episode modulation) never hurts, eliminates scaling's tense cost (S3), and reads +0.06 over scaling alone — suggestive on every-seed direction in E2, not established at n=10 in E3.
4. The forcing-rate retirement A/B remains **gated**: nothing has cleared 0.75 yet.
