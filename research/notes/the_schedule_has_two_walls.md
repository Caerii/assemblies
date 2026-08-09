# The schedule has two walls — K=40 sets the program's best number, and my F3 bar couldn't see the curve it predicted

**Task #148 (E19). Mechanism `f5d3d33` (`morph_flush_every`, default byte-identical, flush-count pinned by test); measurement `flush_rate_sweep.py` (bars F1–F3 before any swept cell). K ∈ {1, 40, 160, 0} × zipf-{200, 400} × seeds 42–46.**

## The curve

| K (episodes/flush) | zipf-200 | zipf-400 |
|---|---|---|
| 1 (per-episode) | 0.683 | 0.649 |
| **40** | **0.730 ± 0.031** | **0.658 ± 0.028** |
| 160 | 0.707 | 0.626 |
| 0 (per-phase, E17 regime) | 0.690 | 0.572 |

Two-walled at both budgets, peak at K=40, exactly the shape the law
predicts — and the 400-frame collapse is substantially RECOVERED by
schedule alone (0.572 → 0.658, with no corpus or architecture change).
0.730 is the program's best number, ahead of the thrice-replicated
0.700.

## Honest bar accounting

- **F1 FAILS at this power**: K=40 over the best endpoint at 400 is
  +0.009 ± 0.048 — the interior optimum is visually present at both
  budgets but not seed-powered at n=5 (K=1 is nearly as good at 400).
- **F2 FAILS, narrowly**: 0.730 < 0.75, CI reaching 0.767. The bar is
  undecided-leaning-close, not cleared; saying "basically 0.75" would
  be the exact sin this program exists to avoid.
- **F3: the MEASUREMENT confirms, the BAR was mis-designed.** Pre-flush
  column-mass concentration is cleanly monotone in K (1.02 → 1.07 →
  1.15 → 3.02) — the account's ordering prediction, confirmed. But I
  registered a rank CORRELATION between concentration and accuracy,
  and a correlation cannot see a TWO-WALLED curve: below the optimum,
  concentration and accuracy rise together (the E9 wall is about
  colliding with fast dynamics, not about mass). The proxy could not
  have confirmed the claim even where the claim is true — the E14
  lesson (assert the claim, not its proxy) landing on my own
  registration from the other side. Recorded as a design error; the
  right F3 would test monotonicity ABOVE the optimum only
  (K 40→160→0: concentration 1.07→1.15→3.02, accuracy
  0.730→0.707→0.690 and 0.658→0.626→0.572 — monotone-decreasing in
  BOTH budgets, which IS the account's prediction where it applies).
- **Guards pass**: K=0 reproduces E17 at both budgets (0.690/0.572 vs
  0.700/0.570); tense flat across all arms; roles 40/40.

## Standing, and the one cheap decision left

The timescale law now has its measured curve: homeostasis paced too
fast collides with fast Hebbian dynamics (left wall, E9/E19-K1), paced
too slow lets within-interval mass concentration run away (right wall,
E18/E19-K0), optimum near ~40 episodes/flush at this scale. Adoption:
mass-paced flushing (K≈40) is strictly better than the per-phase
default in every measured cell and becomes the recommended form
wherever deferred scaling is used.

**E19b (power extension, same bars, declared before running): seeds
42–51 on arms K ∈ {1, 40, 0} × both budgets.** Doubling seeds on the
SAME registered bars is a power increase, not bar-shopping — the
estimate is unbiased and the bars unchanged. It decides whether F1's
interior optimum and F2's 0.75 are real at n=10, and it is the last
cheap measurement before the consolidation/adoption unit and the
CHILDES graduation.
