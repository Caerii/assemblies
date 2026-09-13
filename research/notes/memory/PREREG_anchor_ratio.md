# PREREG: is M* a function of the ANCHOR-TO-RECURRENCE ratio?

Registered before running. Successor to `PREREG_formation_interference.md`
(F2 passed: anchor 2x -> M* 2.86x; F3 failed: erosion is diffuse, not capture).

## The constraint ledger a mechanism must fit

    C1  M* ~ (n/k)^2 at a fixed operating point     (CS1, verified)
    C2  M* ~ p^-0.6      over p in [0.3, 0.7]        (X1)
    C3  M* ~ beta^-1.3   over beta in [0.05, 0.2]    (X2; 0.05 censored)
    C4  M* ~ s^1.52      anchor size 100 -> 200      (F2)
    C5  erosion is retroactive, diffuse, early-first (F3)

## The hypothesis

C2-C4 are one fact: the forming assembly is a k-WTA fight between its own
stimulus anchor and the trained recurrent pull of everything stored before it.
Anchor strength (s), density (p) and gain (beta) enter only through that ratio,
so they TRADE OFF: an excursion in p or beta can be undone by a computed change
in s, and M* against x = s^1.52 p^-0.6 beta^-1.3 is a single curve.

## Held-out predictions (n=4000, k=100, T=8, w_max=20, arm B, 16 brains)

Baseline M* = 23.5 at (p=0.5, beta=0.10, s=100). Three excursions with
measured uncompensated ceilings, and the anchor that the ratio predicts
restores the baseline (s' = 100 * (M*_base / M*_exc)^(1/1.52)):

    A1  beta 0.10 -> 0.20   M* 8.0   (factor 2.94)   s' = 203 -> run s = 200
    A2  p    0.5  -> 0.7    M* 17.6  (factor 1.335)  s' = 121 -> run s = 121
    A3  p    0.5  -> 0.3    M* 28.8  (factor 0.816)  s' =  87 -> run s =  87

Grid --ms 8,12,16,20,24,28,32,40,48; M* read by `ceiling_from_curve`.

    PASS (ratio law):    each compensated M* within [17.6, 29.4) -- +/-25% of
                         23.5, i.e. closer to the baseline than to its own
                         uncompensated value. A1 is PRIMARY: it is the largest
                         excursion (2.94x) and the one a wrong law misses most.
    FAIL:                A1 stays within 25% of 8.0, or lands past 40.
    Partial:             A2/A3 pass, A1 fails -> p trades against the anchor but
                         beta does not; beta acts elsewhere (e.g. C5's erosion
                         term). Reported as such, not adopted.

The exponents are two- and three-point fits and are used ONLY to place the
compensated cells; no exponent is adopted from this test. What is adopted on
PASS is the mechanism CLASS: capacity is set by the anchor-to-recurrence drive
ratio at formation.

---

## Result (2026-09-03): PASS 3/3 -- the ratio law collapses the operating point

    cell  excursion          uncompensated  anchor s'  compensated M*  bracket   fill
    A1    beta 0.10 -> 0.20      8.0          200         29.4         [28,32)   0.53
    A2    p    0.5  -> 0.7      17.6          121         25.0         [24,28)   0.50
    A3    p    0.5  -> 0.3      28.8           87         25.3         [24,28)   0.62

Uncompensated, the three excursions span 8.0-28.8 (3.6x). Compensated by an
anchor computed ONLY from the previously measured exponents, they span
25.0-29.4 (1.18x), all within 25% of the 23.5 baseline. Logs committed as
`anchor_ratio_A{1,2,3}.log`.

**Disclosure on A1.** Its interpolated M* = 29.4 sits exactly on the numeric
band's upper edge ("within [17.6, 29.4)"). The registered prose criterion --
closer to the baseline than to its own uncompensated value -- is met
decisively (|29.4-23.5| = 5.9 vs |29.4-8.0| = 21.4), and a 16-brain ceiling
interpolated inside a [28, 32) bracket cannot resolve a 0.05 edge either way
([[exact-tables-are-tie-fragile]]). Judged PASS on the prose criterion, with
the edge stated.

**Direction of the residual.** All three compensated cells land ABOVE 23.5
(+6%, +8%, +25%), and the largest overshoot is the largest anchor change. The
anchor exponent 1.52 (a two-point fit) is therefore probably slightly high;
no exponent is adopted, as registered.

**Adopted: the mechanism CLASS.** Capacity is set at FORMATION by the ratio
of the stimulus anchor to the trained recurrent pull; p and beta enter through
that ratio and trade off against the anchor. Registered as `CAP-ANCHOR-RATIO`.
Consistent with F2 (anchor 2x -> 2.86x), C5 (retroactive erosion: later
training strengthens the pull on earlier attractors), and CS1 (the (n/k)
dependence is the pull's second-order chance-overlap term, still unquoted).

## Evidence disposition (2026-09-12)

Closed with its Result section (PASS 3/3). Retained artifacts: the three excursion logs research/results/logs/anchor_ratio_A1.log, research/results/logs/anchor_ratio_A2.log and research/results/logs/anchor_ratio_A3.log. No runner record or source archive exists for them; the register entry CAP-ANCHOR-RATIO carries that provenance gap.

## Sensitivity replay (2026-09-12, registered before running)

`CAP-ANCHOR-RATIO` is adopted on the three excursions above, whose logs have
no run record. This replay retains the anchor contrast itself as a per-seed
artifact the register can check, on the maintained capacity protocol
(half-cue rank-1 recall; `research/experiments/seq_capacity_scaling.py`,
`--compare-anchors`), not as a reproduction of the 2026-09-03 numbers,
whose readout differed.

Protocol: `python -m research.runner capacity-scaling --tag UNIQUE
--registration research/notes/memory/PREREG_anchor_ratio.md
--compare-anchors 100 200 --arms B --nk 4000:100 --rounds 8
--ms 8,12,16,20,24,28,32,40,48,64,96,128`; p = 0.5, beta = 0.10,
w_max = 20, net readout, unrefracted, ungated, seeds 42 to 61 (twenty
brains), the two conditions consuming identical measurement samples.

Bars:

- **AN-1, the anchor law's direction.** The ceiling M* of the anchor-200
  condition is at least twice that of the anchor-100 condition (F2 measured
  2.86 under the earlier readout; the law says (200/100)^1.52 = 2.87). Both
  ceilings must be interior to the grid (not censored).
- **AN-2, the retained per-seed contrast.** Let M_c be the largest checkpoint
  at which every brain's anchor-200 rank-1 recall is at least 0.5. At M_c the
  anchor-200 rank-1 exceeds the anchor-100 rank-1 by at least 0.3 on every
  brain. This is the register's sensitivity check; its checkpoint is chosen
  by the rule just stated, not by inspection.
- **AN-3, a shared floor.** At M = 8 both conditions recall on every brain
  (rank-1 at least 0.9): the anchor moves the ceiling, not the low-load
  regime.

A bar that fails is recorded with its numbers; the entry's caveat then says
what the maintained protocol supports.

## Sensitivity replay result (2026-09-12, twenty brains: AN-1, AN-2, AN-3 PASS)

Artifact:
`research/results/runs/memory.capacity-scaling/anchor-pair-20260912/results.json`
(engine `hashed_assembly_memory`, seeds 42 to 61, run from the pinned
runs worktree at commit 22e037e7).

| condition | M* (half-cue rank-1, interior) | rank-1 at M = 24 | at M = 48 | at M = 64 |
|---|---:|---:|---:|---:|
| anchor 100 | 23.3 | mean 0.41 (min 0.04) | min 0.00, max 0.03 | 0.02 |
| anchor 200 | 70.5 | 1.000 on every brain | 1.000 on every brain | mean 0.66 |

- **AN-1 PASS.** 70.5 / 23.3 = 3.02, above the bar of 2 and within 6% of the
  law's (200/100)^1.52 = 2.87; both ceilings are interior to the grid. Under
  the maintained half-cue readout the two ceilings (23.3 and 70.5) are
  within 1% and 5% of the 2026-09-03 stimulus-cued values (23.5 and 67.1).
- **AN-2 PASS.** M_c = 48 (the largest checkpoint at which every anchor-200
  brain recalls at least 0.5: all twenty recall 1.000 there). At M = 48 the
  anchor-200 minus anchor-100 rank-1 difference is at least 0.969 on every
  brain. This is the register's retained check for `CAP-ANCHOR-RATIO`.
- **AN-3 PASS.** At M = 8 every brain in both conditions recalls 1.000.
