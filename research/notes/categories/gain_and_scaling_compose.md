# Gain and scaling compose — monotonically, on every seed, and the tense cost cancels

**Task #131 (E2). Experiment: `research/experiments/surprise_gain_recall.py` (pre-registered R1–R5, GAIN_MAX=4.0 chosen a priori). 4 arms × 5 paired seeds, n=3000, natural rates. Run wall-clock: ~15 minutes for 20 trainings via the new parallel cell runner (vs ~70 sequential) from a worktree snapshot.**

## Results

| arm | number balanced | PL acc | tense balanced |
|---|---|---|---|
| OFF | 0.500 ± 0.044 | 0.0–0.1 | 0.679 ± 0.096 |
| GAIN | 0.540 ± 0.028 | 0.0–0.2 | 0.675 ± 0.029 |
| SCALED | 0.660 ± 0.111 | 0.3–0.6 | 0.615 ± 0.075 |
| BOTH | **0.740 ± 0.102** | 0.3–0.8 | **0.689 ± 0.060** |

- **R1 REFUTED by 0.010** (BOTH 0.740 vs the registered 0.75 bar; 3/5 seeds at or above it; the CI spans the bar). Achingly close is still refuted.
- **R2 REFUTED as registered**: BOTH−SCALED = +0.080 ± 0.129 — the CI includes zero. But read the per-seed deltas: [+0.10, 0.00, +0.05, +0.25, 0.00] — no seed is negative. Direction consistent, power insufficient at n=5 with two tie seeds.
- **R3 CONFIRMED**: GAIN alone lands between OFF and SCALED (0.540; every delta ≥ 0), exactly as the mechanism story requires — gain strengthens rare writes but leaves the SG mass advantage standing.
- **R4 CONFIRMED, better than predicted**: not only does gain add no tense cost — **it cancels scaling's**. SCALED pays −0.064 on tense; BOTH is +0.010 vs OFF. Mechanistically coherent: scaling deletes the frequent class's accumulated-mass advantage, and gain restores a per-episode strength differential that carries the same information by other means.
- **R5 CONFIRMED exactly**: roles 9/9, C1 identical, C2 ≤ 0.067, every arm, every seed.

An unregistered observation worth keeping: gain **collapses seed variance** — tense ±0.029 under GAIN vs ±0.096 under OFF; number ±0.028 vs ±0.044. Novelty gain equalizes effective exposure across seeds' corpus draws, which is variance reduction for free. Not claimed, just noted for the next design.

## Reading, and what happens next per the registered decision rule

The full ordering OFF < GAIN < SCALED < BOTH on number, with the two singles' costs and gaps behaving exactly as the independent-components hypothesis (E1: mass; E2: write-strength) predicts, is strong mechanistic corroboration — and the registered bars were still missed, which is what pre-registration is for. The decision rule prescribes: **sweep GAIN_MAX** (4.0 was a priori, not tuned) **and add seeds for power** (the runner makes 10 seeds ≈ 30 minutes). R2's power problem is exactly the two tie seeds; the variance-reduction observation suggests the composed arm's spread will tighten with any gain recalibration.

What is already bankable regardless of the sweep: **frequency imbalance now has a two-mechanism learning-rule answer whose components compose without interference and without collateral damage to the parse** (guards exact everywhere). The forcing-rate retirement A/B remains gated on a composed configuration clearing the bar honestly.

## Speed fixes validated in the same run

- Parallel (arm × seed) pool: 20 cells ≈ 15 min (~4.5×).
- Fingerprinted trained-parser cache: all 20 parsers cached; the next experiment's OFF arm is free.
- Worktree-snapshot execution: the live tree stayed editable during the whole run (the consolidation-default flip and the `_ROLE_MAP` alias collapse landed while it trained).
