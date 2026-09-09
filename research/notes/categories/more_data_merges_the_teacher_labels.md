# More data merges the teacher labels — the 2×2 finds the real scaling blocker one level up

**Task #143 (E14). Experiment: `research/experiments/sampling_budget_interaction.py` (bars I–IV registered before any cell trained). {UNIFORM, ZIPF} × FRAMES {50, 200} × seeds 42–51, SLOW-scaled R1, n=3000.**

## The 2×2 (attested number balanced)

| | frames 50 | frames 200 | slope |
|---|---|---|---|
| UNIFORM | 0.650 ± 0.063 | **0.490 ± 0.052** | **−0.160** |
| ZIPF | 0.583 ± 0.047 | 0.565 ± 0.050 | −0.018 |

## Honest bar accounting

- **Bar I passes as written (+0.142 ± 0.117, CI excluding zero) and the
  claim it operationalized FAILS.** The interaction was registered as
  "zipf's budget slope exceeds uniform's" because the pinning-break
  prediction was a *positive* zipf slope. Measured: zipf is flat and
  uniform *collapses*; the inequality holds through the wrong term. A
  bar that pattern-matches while its motivating claim dies gets reported
  as both — this is the E-series' recurring lesson (assert the claim,
  not its proxy) landing on my own registration.
- **Bar II fails** (0.565 ≪ 0.75).
- **The exposure arithmetic was CONFIRMED**: zipf-200 delivered 49 PL
  episodes, head forms at 4.0 exposures/seed ("children", "foods",
  "women"), ten more at 2.0. Coverage was not bounded at ~9 as guessed
  (tail attests once → n=30), but the redistribution itself worked. So
  the registered branch is: *exposure rises and accuracy does not
  follow* — the suspect is whatever breaks the mass→accuracy link at
  scale.
- **Bar III catches that suspect red-handed.** Shared SG∩PL image
  columns: 2.6 → **17.8**/30 (uniform), 2.7 → **10.3** (zipf) as frames
  go 50 → 200. E12 showed identical-replay repetition merges the label
  images; E14 shows **merging follows total label-projection COUNT,
  through fully varied sentences**. More corpus per se degrades the
  labels' separability. It explains every cell: uniform-200 collapses
  hardest (most projections, 17.8), zipf-200's earned exposure buys
  almost nothing ("children" at 4 exposures reads 0.80 where E11/E12's
  "girls" at 4 read 1.00 on clean 2.6-col images), and tense sags at
  200 in both arms (0.67→0.62, 0.68→0.60) because its labels merge the
  same way.
- **Bar IV holds** (ρ(mass, correct) = 0.648 ± 0.057 — fourth corpus).
  Noted without a registered claim: the mass→accuracy correlation has
  slid as merging rose across experiments (≈0.85 clean → 0.76 → 0.65
  here), which is what a blurring readout should do to it.

## What the arc now says, finally

The corpus/learning-rule axis is EXHAUSTED, with a law at each level:

1. **Reliability = per-item afferent mass** (ρ 0.65–0.87, four corpora).
2. **Mass = per-form exposure**, reliable regime ~2–4 episodes.
3. **Exposure = budget × allocation**, and Zipf × budget genuinely
   raises it (E14's head forms) where either alone cannot (E7, E13).
4. **But the current feature architecture — TWO label values sharing ONE
   k-WTA area — destroys label separability as total label projections
   grow.** k-WTA amplifies shared drive (the COLT22 amplification
   memory); both label stimuli increasingly recruit the area's dominant
   mass-attractor, and the images converge. This is not a statistics
   problem and no corpus shape fixes it: it is the architecture squeezing
   two attractors into one area.

The fix is the papers' own: **per-value group areas with mutual
inhibition** — precisely the machinery this codebase implemented
(inhibition.py) and then found DORMANT (zero co-targeting call sites).
The dormant paper mechanism finally has a measured job: keep the value
images apart while the corpus scales.

**E15 (to register): per-value feature sub-areas + mutual inhibition.**
NUMBER splits into NUMBER_SG/NUMBER_PL (TENSE likewise) as group areas;
label training co-targets the group so mutual inhibition actually fires
(the paper-spec memory's condition); recall reads out per-area instead
of comparing images inside one area — image merging becomes structurally
impossible. Bars: (a) the merging counter stays at floor at 200 frames
BY CONSTRUCTION (verified, not assumed); (b) zipf-200 balanced ≥ 0.75
(steps 1–3 say exposure is already sufficient there); (c) the mass
mechanism transfers (ρ > 0 per-area); (d) uniform-200 recovers to
≥ uniform-50 (the collapse was merging; removing merging must remove
the collapse — the strongest falsifiable prediction of this note's
model).
