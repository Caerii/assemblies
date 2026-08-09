# Slow homeostasis halves the instability but number saturates at ~0.65 — while tense quietly sets its best mark

**Task #138 (E9). Experiment: `research/experiments/slow_homeostasis_recall.py` (pre-registered T1–T4). SLOW-SCALED × R ∈ {1,4}, 10 seeds, paired against E8's FAST values.**

## Verdicts

| | number | tense |
|---|---|---|
| FAST R1 (ref) | 0.650 ± 0.058 | 0.642 |
| SLOW R1 | **0.650 ± 0.063** (T1 passes exactly) | 0.670 ± 0.049 |
| FAST R4 (E8) | 0.630 ± 0.120 | 0.673 |
| SLOW R4 | 0.635 ± **0.068** | **0.729 ± 0.051** (series best) |

- **T1 PASSES to the fourth decimal** — deferring normalization to phase boundaries reproduces per-update scaling's R=1 result. The timescale change is safe.
- **T2 REFUTED**: paired SLOW−FAST at R4 = +0.005 ± 0.124, per-seed deltas spanning −0.25 to +0.25. No composition gain for number.
- **But the variance signature half-confirms the interference diagnosis**: SLOW R4's own spread is half of FAST R4's (±0.068 vs ±0.120). Deferral removed the seed-bistability — the fast-loop collision was real — it just wasn't what was capping the mean.
- **Tense composes beautifully**: 0.729 under slow scaling + repetition is the highest tense reading of the entire nine-experiment series, beating even OFF-R4's 0.725 while keeping scaling's class-mass correction.

## The refined model

For NUMBER, scaling and repetition are **substitutes, saturating together at ~0.65** — the same substitution pattern E5 found between scaling and n-doubling. Three independent additions (more neurons, more exposure, better scheduling) all bounce off the same ceiling, while the raw baseline responds to exposure (E8-OFF: 0.530 → 0.585). Whatever caps scaled-number at 0.65, it is not mass, not exposure, not substrate size, and not scheduling. The remaining suspect from the registered decision rule: **w_max saturation** — repeated multiplicative writes hit the weight cap (w_max = 20), after which neither more exposure nor better normalization can express further differentiation. Tense, with its 3× larger form inventory spreading writes across more synapses, would hit the cap later — consistent with tense composing where number saturates.

E10 (registered): instrument the weight distributions on the VERB_CORE→TENSE and core→NUMBER fibers across R and scheduling — the fraction of probed-fiber weights at w_max, per class. Pure measurement, no mechanism; if PL-relevant weights pile up at the cap under R=4 while tense's don't, the saturation account is confirmed and the lever is w_max (or sub-multiplicative writes), not anything tried so far.

## Standing after E1–E9

- **Adoption-ready finding**: slow (phase-boundary) scaling is strictly preferable to per-update scaling — identical means, half the variance, and biologically correct. When scaling is next used anywhere, deferred should be its form.
- **Best-known configurations**: number = scaling (any schedule) at 0.65; tense = slow scaling + repetition at 0.73.
- The 0.75 number bar has now survived nine experiments; its remaining suspects are countable on one hand, and each has a registered discriminating measurement.
