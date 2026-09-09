# The decision was the bottleneck — the split makes data monotone, and the within-area readout proves it

**Task #144 (E15). Mechanism `db898fb` (per-value areas + the paper's mutual inhibition, opt-in, default byte-path untouched); measurement `ca03d54` + amendment `540fec1` (both readouts recorded per item). {UNIFORM, ZIPF} × {50, 200} × seeds 42–51, SLOW-scaled on the value areas, MI readout registered primary.**

## Registered bars

- **A PASSES**: label images in disjoint areas in every cell (verified),
  and the functional guard inverts E14's pathology — MI margins RISE
  with budget (0.085→0.105 uniform, 0.075→0.099 zipf) where the shared
  area's separability collapsed.
- **D1 PASSES — the merging model is CONFIRMED BY INTERVENTION.** The
  uniform budget slope is −0.030 ± 0.102 (E14: −0.160; the CI excludes
  it). **D2**: uniform-200 recovers 0.490 → 0.590 (readout-caveat
  attached, but the direction is the model's prediction).
- **B MISSES on the registered (MI) readout**: zipf-200 = 0.637 < 0.75.
- **C PASSES**: mass→correctness transfers across two weight matrices
  (ρ = 0.503 ± 0.070 — fifth corpus/architecture).

## The readout A/B (pre-declared diagnostic, now the headline)

| cell | MI (registered) | within-area OVERLAP |
|---|---|---|
| UNIFORM-50 | 0.620 ± 0.099 | 0.540 ± 0.075 |
| UNIFORM-200 | 0.590 ± 0.046 | 0.605 ± 0.039 |
| ZIPF-50 | 0.594 ± 0.080 | 0.528 ± 0.080 |
| ZIPF-200 | 0.637 ± 0.048 | **0.700 ± 0.037** |

Three facts:

1. **Under the overlap readout both budget slopes are POSITIVE**
   (uniform +0.065, zipf +0.172) — the first configuration in fifteen
   experiments where more data monotonically helps — and zipf's slope
   exceeds uniform's, which is E14's pinning-break claim finally holding
   in the form its bar was written for.
2. **The readouts CROSS**: MI wins at 50 frames, overlap wins at 200.
   Single-step drive comparison saturates early (margins pinned at
   7–10% regardless of upstream separation), while evidence-overlap
   keeps converting exposure into accuracy. This is the THIRD
   independent measurement of the #24 finding — cross-area drive
   comparison is the model class's weak primitive — now in a setting
   with a clean side-by-side.
3. **zipf-200-overlap = 0.700 ± 0.037** is the best 200-frame number of
   the program (seed range to 0.80), 0.05 under the E-series bar, with
   the per-item table showing the residue is exposure: most items sit
   at 1–2 episodes, below the ~3–4 reliable regime.

## What this closes and what it opens

CLOSED: the architecture question. Two values in one k-WTA area is a
measured scaling pathology (E14); one area per value removes it by
construction and by measurement (D1/A). The paper's per-value-area
choice for ROLE now has a quantified justification.

OPEN, and named: the paper's competition semantics as we implemented
them — ONE step of drive comparison — discard most of the margin the
representation provides. The paper's own dynamics suggest the fix:
the winner LATCHES via recurrent self-amplification ("the current
constituent will continue to receive the most input from its own
recurrent firing"), growing the margin multiplicatively instead of
reading it once. That is a rounds parameter, not a new mechanism.

**E16 (to register): latched competition × the last exposure gap.**
(i) MI competition run for T ∈ {1, 3, 10} recurrent steps at zipf-200:
bar — margin grows with T and latched-MI accuracy reaches the overlap
readout's (the neural mechanism matching the Python argmax it is meant
to implement); (ii) zipf-400 under the best readout: bar — attested
balanced ≥ 0.75 (the exposure arithmetic: ~100 PL episodes over ~30
bounded forms ≈ 3+/form, the reliable regime for most of the exam).
Pass both and the number arc ends with every level fixed where it
lives: corpus statistics, area architecture, competition dynamics.
