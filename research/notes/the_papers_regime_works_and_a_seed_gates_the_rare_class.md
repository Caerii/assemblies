# The papers' regime works — word-conditioned images, 0.745 at n=10⁴, one 0.946 seed at n=10⁵ — and a per-seed factor gates the rare class

**Task #151, paper-regime unit. Mechanism `e4f6da8` (morph_label_stim,
morph_beta_gain; defaults byte-path pinned); 2×2 registered `2acc8e2`,
run on Brown, seeds 42–46, n ∈ {3000, 10000}; scale cell registered
`d963efc` BEFORE any n=10⁵ run; interior-gain probe post-hoc with its
prediction pre-stated. Axis deviation from the literature note stated
at registration: β gain instead of kp (Remark 2 tradeoff; no per-fiber
p lever).**

## The 2×2 verdicts

| arm (n=10⁴) | sg | pl | bal | img∩attractor |
|---|---|---|---|---|
| L1G1 (E-series protocol) | .218 | .865 | .542 | .998 |
| L0G1 (routing-only) | .999 | .287 | .643 | .023 |
| L1G4 (gain only) | .418 | .635 | .526 | .932 |
| **L0G4 (papers' regime)** | .999 | .491 | **.745** | .028 |

- **P1 PASSES, decisively**: routing-only training drops
  image∩class-attractor from 0.998 to 0.02–0.13. The papers' mechanism
  — the teacher routes, the word selects winners — is real in this
  engine, and it is what every score change rides on.
- **P2 PASSES as registered** (bal ≥ .70, SG ≥ .60 at both n): 0.704 /
  0.745 — the first configuration to beat the E-series synthetic
  terminal (0.727) ON REAL DATA. But the bar under-specified PL
  (.409/.491): the bias FLIPPED to the frequent class. Bar-design
  lesson recorded: "recovery" must be min(sg, pl), not the previously
  failing class.
- **P3 FAILS at the letter and refines the theory**: gain alone −0.015
  NS; routing alone −0.058 at n=3000. NEITHER axis suffices — word
  selection decides WHAT is compared, the margin gain decides whether
  the boost survives; only together +0.204 (CI excludes 0).
- **P4 directional PASS**: the class gap shrinks with n once images
  are word-specific — α* capacity is binding again, exactly when the
  attractor stops carrying the load.
- Synthetic guard −0.071 ± 0.077: marginal fail, recorded; the paper
  regime is NOT yet adoptable as default.

## The collision-load law, and what the scale cell did to it

The residual PL deficit tracks cross-word row-collision load V·k²/n
(shared core rows leak the frequent class's boosted mass): 109 / 33 /
3.3 at n = 3000 / 10⁴ / 10⁵ against k=30. The registered n=10⁵ cell:

    seed 42:  sg 1.000  pl 0.891  bal 0.946   <- the predicted regime EXISTS
    seed 43:  sg 0.997  pl 0.283  bal 0.640
    seed 44:  sg 1.000  pl 0.500  bal 0.750

**BIMODAL — the bar (PL ≥ .75) is not met at n=3, and the mean ± CI is
meaningless here** (the reference-is-bimodal lesson). The interior-gain
account (gain amplifies collisions too, so back off to G2) was tested
with its prediction pre-stated and **REFUTED**: G2 reproduces the same
per-seed pattern (seed 42: .783; seed 43: .283; seed 44: .391).

The gating factor is PER-SEED, GAIN-INVARIANT, and CONFIG-INVARIANT:
seed 43's PL sits at .28–.35 in every arm at every n where images are
word-specific, while seed 42's is high everywhere. The seed fixes the
core-row geometry of the 46 PL words (identical across arms at fixed
n), so the suspect is the PL words' own core-assembly draw — not the
value-area training at all.

## Registered continuation

Per-seed PL attribution at n=10⁵: 10+ seeds, per-word own/other drives
(the attribution unit's instrument), asking (a) is a bad seed's failure
concentrated in specific PL words or uniform, (b) does own-drive or
other-drive move between good and bad seeds, (c) does the failure
correlate with per-word core-row collision counts (directly countable
from the connectome). No mechanism changes until this is answered —
the campaign's standing rule. Adoption of the paper regime waits on
(i) that attribution, (ii) a PL-inclusive recovery bar, (iii) a
non-negative synthetic guard.
