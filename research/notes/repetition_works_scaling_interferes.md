# Repetition confirms the exposure account — and exposes scaling's timescale error

**Task #137 (E8). Experiment: `research/experiments/repetition_recall.py` (pre-registered P1–P5). R ∈ {2,4} × {OFF, SCALED}, budget 50, 10 seeds, vs measured R=1 references.**

## Verdicts

| config | R=1 (ref) | R=2 | R=4 |
|---|---|---|---|
| OFF number | 0.530 ± 0.038 | 0.550 ± 0.041 | **0.585 ± 0.048** |
| OFF tense | 0.678 | 0.684 ± 0.039 | **0.725 ± 0.038** |
| SCALED number | 0.650 ± 0.058 | 0.610 ± **0.098** | 0.630 ± **0.120** |
| SCALED tense | 0.642 | 0.677 ± 0.037 | 0.673 ± 0.032 |

- **The exposure hypothesis is CONFIRMED where it can be tested cleanly**: OFF rises monotonically with R on BOTH features — the only manipulation in eight experiments that lifted the raw baseline. Per-form exposure is what reliability is made of, exactly as E7's pinning analysis predicted.
- **P1/P2 REFUTED for SCALED**: repetition and scaling interfere — flat means and a variance explosion (±0.058 → ±0.120), i.e., seed-dependent instability, not mere no-effect.
- **P3 passes everywhere** (tense improves under repetition).

## The interference has a name: timescale

Biological synaptic scaling is SLOW — Turrigiano's homeostasis operates over hours-to-days, explicitly segregated from fast Hebbian plasticity, and that segregation is load-bearing in the theory (fast learning inside a slowly renormalized envelope). Our `_normalize_area_columns` runs after **every projection** — a fast loop fighting a fast loop. At R=1 the interference was mild enough to leave a +0.12 margin; repetition multiplies write events and with them the per-update renormalizations, and the two mechanisms' interaction becomes seed-bistable.

E8 therefore produced something better than its bars: **evidence for the biological timescale separation**, from a system that was never told about it. The fix is not a parameter — it is scheduling: apply the homeostatic step once per epoch (or per phase), not per update. Fast Hebbian inside slow homeostasis, as the biology specifies.

## E9 (registered): slow homeostasis × repetition

Engine change: `synaptic_scaling_interval` — scale touched columns every K updates or at phase boundaries instead of every update (current behavior = interval 1, byte-identical default). Arms: {fast-scaling, slow-scaling} × R ∈ {1, 4}, 10 seeds. Registered expectation: slow scaling preserves R=1's margin AND composes with repetition's exposure gains — the first configuration with a principled shot at 0.75 since the series began, and the most biologically faithful configuration yet tested.

## Standing after E1–E8

Exposure raises reliability (E8-OFF, confirmed). Scaling corrects class mass (E1, established). Their composition requires the timescale separation biology already mandates (E9's question). Every mechanism default remains OFF; every corpus default remains at the measured baseline.
