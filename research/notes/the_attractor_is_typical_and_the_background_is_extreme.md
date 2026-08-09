# The attractor is typical and the background is extreme — both registered accounts refuted, and the inversion is a readout comparing the wrong quantities

**Task #151, attribution unit. Registration `09a7e9d` (bars G1–G3 before
the run); run `imbalance_attribution.py` on the Brown slice, seeds 42–46
× n ∈ {3000, 10000}, split default, scaling OFF (the D/E substrate).
Equivalence gate: decomposed single-word training is answer-identical to
the monolithic call, 409/409.**

## Both registered hypotheses are refuted

- **H2 (winner churn) REFUTED by G2**: per-form training-image
  consecutive-episode stability is ~0.99 for BOTH classes at BOTH n
  (gap −0.014 ± 0.007 at n=3000, i.e. SG marginally MORE stable).
  Images do not churn; norm_init + stimulus anchoring hold.
- **H1 (background growth, as registered) REFUTED by G1's letter**: all
  drives SHRINK with n (own ratio 0.062 ± 0.012, other 0.134 ± 0.016 —
  the bar asked whether other grows; nothing grows). But G1's SPIRIT
  survives as a relative statement, and it is the finding:

## What the instruments actually showed

1. **Own-drive falls below other-drive as n grows.** med(own)/med(other)
   = 1.12 at n=3000 → 0.53 at n=10000. Between these scales the trained
   signal CROSSES BELOW the untrained background of the competing area.
2. **The frequent class crosses harder** (G3): exposure-matched own-drive
   SG/PL = 0.795 at n=3000 → 0.342 at n=10000.
3. **Each value area holds ONE assembly — the class attractor.** Every
   form's training image overlaps the class-modal columns at 0.998. Word
   identity lives in the WEIGHTS into that shared assembly, not in which
   neurons fire. (This was equally true where the architecture WORKED:
   the synthetic 0.727 is a classification readout over exactly such
   attractors. Not a defect — a description.)

## The mechanism, stated

The MI competition compares two quantities of DIFFERENT kinds:

- **Own area**: drive into attractor columns that were selected by the
  LABEL stimulus, so they are statistically TYPICAL w.r.t. the word's
  core rows (~k·p hits), multiplied by the Hebbian boost
  (1+β)^exposure ≈ 1.16–2 at β=0.05, exposure ~3.
- **Other area**: k-WTA freely selects that area's EXTREME background
  columns w.r.t. the word's rows — order statistics that strengthen
  relative to the typical column as the candidate pool grows.

Boosted-typical loses to selected-extreme once n is large enough — the
[[norm-init-stability-threshold]] ((1+β)^T vs the pool's extreme value),
now operating BETWEEN areas at recall rather than within one area during
recurrence. The class asymmetry rides on the same comparison: the SG
attractor's per-word connections compete against 362 other words'
boosts into the same 30 columns under w_max, thinning the frequent
class's per-word margin (G3) — the [[hebbian-mass-follows-frequency]]
law measured INSIDE a single shared assembly.

## Instrument honesty

A post-hoc probe read "recall winners ∩ trained image = 0.998" — but the
snapshot was taken after diag's settled overlap probe, so it measured
the SETTLED state, not the MI step's selection. Recorded as a
fake-perfect signature caught by its own cleanness
([[fake-perfect-probe-signatures]]); the MI-step winner identity remains
unmeasured and the mechanism claim above rests on the DRIVE numbers,
which come from the registered instrument.

## The fix this dictates (next unit, registered)

Read drive at FIXED columns: score each area by the word's afferent
mass into that area's LABEL IMAGE (E12's `item_afferent_mass`, the
readout whose ρ 0.50–0.87 across five corpora is the arc's central
law). This denies the opposing area its extreme-value pick — both sides
are then measured on label-defined columns, own = boosted-typical vs
other = unboosted-typical, which is exactly the contrast training
actually wrote. Registered as `morph_readout="mass"`, with bars: SG
recovers on Brown at BOTH n with PL retained; the n=3000→10000 delta
shrinks toward 0 (n-robustness — no extreme-value term remains); the
synthetic #149-gate gate corpus does not regress (paired).
