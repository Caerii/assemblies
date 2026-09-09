# Homeostatic scaling rebalances what frequency swamped — but cannot invent what was never written

**Task #130 (E1). Experiment: `research/experiments/scaled_feature_recall.py` (pre-registered Q1–Q4). Follows #129's finding that Hebbian mass follows token frequency (PL recall at chance with every wire live). 5 paired seeds × 2 arms, n=3000, natural corpus rates — no forcing knobs.**

## The mechanism was already in the repo

`NumpySparseEngine._normalize_area_columns` — Turrigiano-style multiplicative column scaling to a fixed setpoint after each plasticity step — has existed since the norm_init work, OFF by default for a **measured** reason: a per-fiber setpoint cancels the net potentiation that makes an assembly self-sustaining (stability 0.01, completion 0.000, documented in the method). This unit did not add a mechanism; it added a **scope**: `synaptic_scaling` now accepts a collection of target-area names (engine → Brain → EmergentParser, forwarded verbatim), so stimulus-anchored feature areas can be scaled while every attractor-bearing area stays untouched. The scoped form also cannot license global self-recurrence (`allow_self` in `project_rounds` now requires `is True`), closing a truthiness hazard the widened type introduced.

The theoretical bet: the documented failure mode needs an attractor to cancel. TENSE/NUMBER recall needs the afferent fiber to be *discriminative*, not self-sustaining — and column scaling is exactly anti-swamping (a post-neuron's response is divided by its accumulated mass, so 50-noun singular columns dilute 50× while rare plural columns keep their per-synapse advantage).

## Results

| | number balanced | PL acc | SG acc | tense balanced |
|---|---|---|---|---|
| OFF | 0.500 ± 0.044 | 0.0–0.1 | 0.9–1.0 | 0.679 ± 0.096 |
| SCALED {TENSE, NUMBER} | **0.660 ± 0.111** | 0.3–0.6 | 0.7–1.0 | 0.615 ± 0.075 |
| paired Δ (number) | **+0.160 ± 0.111** — every seed positive (0.05–0.25) | | | Δ −0.064 |

- **Q1 REFUTED at its bar** (0.660 < 0.75; PL ≥ 0.6 on only 2/5 seeds) — but the underlying effect is real: the paired CI excludes zero.
- **Q2's registered bar refuted** (Δ > 0.2 on 2/5 seeds, not 4/5) — the effect is roughly half the predicted size.
- **Q3 REFUTED**: scaling costs tense −0.064 (allowance was −0.05). The cost mechanism is the same as the benefit: PRESENT's accumulated-mass advantage was part of what made tense readable.
- **Q4 CONFIRMED exactly**: roles 9/9, C1 identical winners, C2 ≤ 0.067 on every run in both arms — the scope does not leak into the parse. (Pinned mechanically in `test_scoped_synaptic_scaling.py`, which also records why the pin is aggregate signatures, not bit-equality: the shared RNG cursor makes bit-identity a realization claim, not a mechanism claim.)

## Reading

Scaling removes the **accumulated-mass** component of frequency imbalance and only that. A plural form seen twice still wrote only two Hebbian updates; normalization makes those two updates *visible* against the singular mass (PL 0.0 → ~0.45 mean) but cannot make them *strong*. The residual gap is **exposure**, and the biologically-matched lever for exposure is gain, not normalization: surprise-modulated plasticity (neuromodulated beta on high-prediction-error events) writes bigger updates for rare events at the moment they occur. That is E2, now front of queue — with the interesting question being whether scaling + surprise-gain **compose** (they attack independent components: mass and write-strength).

**Adoption decision: default stays OFF.** The tense cost and the half-rescue mean scoped scaling is a validated ingredient, not a production setting. No accuracy pinned; the plumbing is.

## Provenance notes

- A corpus-level null was rejected at design time (it empties the test set); the ablate-style null from #129 established the readout refuses (all ties) when nothing was learned.
- Both drafts of the mechanism test failed informatively before the aggregate form: setpoint arithmetic fights expansion-after-scaling ordering; paired bit-equality fails through the shared RNG cursor (winner-count changes shift every later draw). Both are recorded in the test's docstring.
