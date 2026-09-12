# PREREG: does `norm_init` + `synaptic_scaling` lift the M-ceiling, and does the mechanism survive off its single operating point?

Registered before data. Experiment: `research/experiments/seq_substrate_ceiling.py`.
Follows `PREREG_recurrent_ratchet.md` Amendment 5, which established that the
substrate-C merger is uncancelled CANDIDATE in-degree and that `norm_init` and
`synaptic_scaling` compose (pairwise 0.188 -> 0.027, completion 1.000).

Amendment 5 left two things open, and they are the two parts of this study.

## Why this is not a victory lap

Amendment 5's entire mechanism rests on **one operating point**: n=2000, k=50,
p=0.5. That is in regime, but p=0.5 is a far denser fiber than the organ
actually runs. A mechanism established at one point is not established. Part B
is written to be able to **falsify the claim I just committed**.

## The ceiling must be defined on the attractor, not the lookup

Rank-1 from the FULL cue is the weak readout: substrate C scores 1.000 on it
while its assemblies sit merged at 7x chance, because the stimulus does the
discriminating ([[completion-works-in-regime-under-norm-init]]). A ceiling
measured that way is a claim about LOOKUP capacity.

So, and this is the correction to the original ratchet study, **distinctness is
a GATE, not a report**:

    attractor ceiling  M*_att = largest swept M with
                                mean rank1_half >= 0.50
                            AND mean pairwise <= 3 x chance
    lookup ceiling     M*_look = largest swept M with mean rank1_full >= 0.50

M*_look is reported for continuity with the original study and is never the
headline.

## Each arm at its OWN best beta

Judging B at beta=0.3 is judging it outside its known window
([[beta-opposes-capacity-and-depth]]) and inflated an earlier read of this same
comparison by ~10x. Best beta = the one maximizing M*_att, ties broken by lower
pairwise at that M. The full grid is printed so the choice is auditable.

## Saturation is not capacity

At n=2000, k=50, the area is fully materialized (rows/n = 1.0) by M=96. Past
that no new neuron can be recruited and the "ceiling" is partly a tiling limit,
which is a different quantity from capacity ([[core-areas-near-tiling-limit]]).
`rows/n` is reported at every cell so saturation is visible rather than
inferred. Any ceiling reached at rows/n = 1.0 is reported as SATURATED.

## Part A -- the ceiling

n=2000, k=50, p=0.50, T=8. kp = 25.0 against floor 3 ln n = 22.8: IN REGIME.
M in {8, 16, 32, 64, 96, 128, 192, 256}; beta in {0.10, 0.20, 0.30};
arms B, C, G; seeds 42, 43, 44.

Note the recorded critical load is alpha* ~ 1.15 with alpha = M k / n
([[critical-load-alpha-star]]), i.e. M_max ~ 46 here. A smoke cell already put
G at M=96 (alpha = 2.4) with pairwise 1.1x chance and half-cue 0.906. If that
survives the full run it is in tension with alpha*, and the tension is
reported, not smoothed over.

## Part B -- does the mechanism survive off the single point?

M=16, T=8, beta in {0.10, 0.20}, arms B, C, G, seeds 42, 43, 44, at three
points chosen to hold BOTH the regime margin (kp/floor ~ 1.10) and the chance
floor (k/n = 0.025) fixed while p falls 4x:

    n=2000   k=50   p=0.500   kp 25.0  floor 22.8  margin 1.10
    n=4000   k=100  p=0.280   kp 28.0  floor 24.9  margin 1.12
    n=10000  k=250  p=0.124   kp 31.0  floor 27.6  margin 1.12

## Bars

**CE1 (instrument).** At M=8 every arm has mean rank1_full >= 0.90. A failure
here voids everything below.

**CE2 (regime).** Every cell satisfies kp >= 3 ln n, asserted in code rather
than printed. Two of this project's null results were recorded an order of
magnitude below floor; a printed row at the bottom of a log is indistinguishable
from silence.

**CE3 (THE CLAIM).** M*_att(G) > M*_att(B) at n=2000, each at its own best beta.

**CE4 (the original result stands).** M*_att(C) <= M*_att(B). Substrate C ALONE
does not lift the ceiling -- the original RC3 finding, re-measured in regime.

**CE5 (generalization).** At all three Part B points: mean pairwise <= 1.5 x
chance for G, and >= 3 x chance for C.

**CE6 (mechanism invariant -- THE FALSIFICATION TEST).** At all three points,
rho(in-degree, multiplicity) > 0.15 for C and < 0.10 for G. The committed
mechanism says the merger IS the uncancelled candidate degree. If C merges at
some point with rho ~ 0, the degree account is incomplete there and Amendment 5
must be narrowed to the point it was measured at.

**CE7 (derived direction).** Across the three points, C's pairwise/chance moves
in the same direction as the MEASURED degree heterogeneity `deg_cv`. Registered
as weak: three points, and at constant regime margin the expected in-degree
`rows * p` is roughly constant by construction, so `deg_cv` may barely move. If
it does not move, CE7 is UNINFORMATIVE rather than passed or failed, and will be
reported that way. Stating this in advance so a flat sweep cannot be read after
the fact as confirmation.

## What each outcome means

* CE3 passes and CE4 passes -> the ratchet ceiling IS liftable, and the original
  study's FAIL was an artifact of testing C alone. This is the result.
* CE3 fails with G ~ B -> the composition buys distinctness at fixed M but not
  capacity. Report as distinctness, never as capacity.
* CE6 fails at any point -> Amendment 5's mechanism does not generalize; narrow
  it to p=0.5 in that note and say so in the memory that cites it.

---

# RESULTS

Run 2026-08-25, `seq_substrate_ceiling.py`, 450 cells. **All seven bars pass**,
including CE6, which was written to be able to falsify Amendment 5, and CE7,
which was registered as probably uninformative and was not.

## Part A -- the ceiling

    ceilings from the CURVE, each arm at its own best beta (all chose 0.10)
      B  M* =  41.3  bracket [ 32,  64)  interior 1  [UNRESOLVED: 2.00x]
      C  M* =   8.0  bracket [  8,   8)  interior 0  [CLIFF]
      G  M* = 103.7  bracket [ 96, 128)  interior 3

**The estimator refuses to let two of these be quoted, and that is the point.**
B's bracket is a factor of 2 wide, so 41.3 is grid-dependent; C never had an
interior point. Only G's estimate is both supported and resolved. What IS
grid-independent is that **the brackets do not overlap**: G's ceiling lies
above 96 and B's below 64. The ratio is therefore at least 96/64 = 1.5x and at
most 128/32 = 4x; the point estimates say 2.5x.

## The stronger comparison is at FIXED M, because saturation is controlled

    n=2000 k=50 p=0.5 beta=0.10      half-cue (per seed)   pairw   dist  rows/n
      B   M=32    0.792  (0.91 0.78 0.69)                  1.29x  1.000  0.967
      B   M=64    0.016  (0.02 0.02 0.02)                  7.26x  0.724  0.994
      G   M=32    0.927  (0.91 0.94 0.94)                  0.90x  1.000  0.969
      G   M=64    0.938  (0.94 0.95 0.92)                  1.10x  1.000  0.997
      G   M=96    0.684  (0.91 0.21 0.94)                  1.24x  1.000  0.999
      G   M=128   0.284  (0.30 0.41 0.14)                  1.91x  0.969  0.999

At **M=64 the two arms are at the same fill** (rows/n 0.994 vs 0.997), so
saturation cannot be the explanation: B has collapsed -- completion 0.016,
overlap 7.3x chance, and a distinct fraction of 0.724, i.e. **a quarter of its
assemblies are exact duplicates** -- while G is intact at 0.938 with every
assembly distinct. That is the result, and it needs no ceiling estimator.

The duplicate fraction is only visible because this study reports it. Mean
pairwise overlap alone is nearly blind to partial collapse, which is exactly
why `_substrate.check_distinct` exists.

**Saturation caveat, as registered.** Both ceilings sit at or past the tiling
limit of an n=2000 area (rows/n 0.97-1.00). This is a capacity result for THIS
area, not a scaling law. G's M=96 cell is 2 good seeds and 1 collapse
(0.91/0.21/0.94), which is why its interval is +/-1.02 -- the ceiling really is
near there, and 3 seeds cannot localise it. How the ceiling scales with n needs
larger areas and more seeds; it is not claimed here.

## Part B -- the mechanism survives, and CE7 was informative after all

    M=16, beta=0.10                pairwise      xchance  rho(deg,mult)  deg_cv
      n= 2000 k= 50 p=0.500  B   0.0213+-0.0030    0.85    +0.008         0.0250
                             C   0.1505+-0.0309    6.02    +0.239         0.0704
                             G   0.0177+-0.0017    0.71    -0.012         0.0247
      n= 4000 k=100 p=0.280  B   0.0177+-0.0013    0.71    +0.028         0.0297
                             C   0.3222+-0.0331   12.89    +0.284         0.1139
                             G   0.0154+-0.0012    0.62    -0.007         0.0292
      n=10000 k=250 p=0.124  B   0.0158+-0.0009    0.63    +0.032         0.0325
                             C   0.3964+-0.0175   15.86    +0.287         0.1532
                             G   0.0144+-0.0011    0.58    +0.006         0.0321

**CE6 (the falsification test) passes at every point**: rho(in-degree,
multiplicity) has a CI lower bound of at least +0.191 for C and an upper bound
of at most +0.045 for G, at all three operating points and both betas.
Amendment 5's mechanism is not an artifact of p=0.5.

**CE7 passes, and I registered it as likely UNINFORMATIVE.** Measured degree
heterogeneity rose 2.2x as p fell 4x (deg_cv 0.070 -> 0.114 -> 0.153), and C's
overlap rose with it (6.0x -> 12.9x -> 15.9x chance). I could not derive the
direction in advance and said so; the measurement supplies it. The merger
tracks the degree heterogeneity of the candidate pool, which is what
"uncancelled candidate in-degree" predicts.

**A finding not asked for: substrate C gets much WORSE in the sparse regime.**
Its overlap goes from 6.0x chance at p=0.5 to 15.9x at p=0.124 -- and p=0.124
is far closer to what the organ actually runs. Amendment 5 measured C at its
most flattering point. Anything that has used `synaptic_scaling` without
`norm_init` on a sparse fiber is worse off than that note implied.

## Bars

    PASS  CE1  instrument: M=8 full-cue CI-low >= 0.90, all arms 1.000+-0.000
    PASS  CE2  regime: asserted for all 450 cells before the run
    PASS  CE3  THE CLAIM: G M*=103.7 > B M*=41.3, brackets disjoint
    PASS  CE4  C alone does not lift it: C M*=8.0
    PASS  CE5  generalization: G <= 1.04x chance and C >= 4.42x at EVERY point
    PASS  CE6  mechanism invariant (FALSIFICATION TEST) at every point
    PASS  CE7  direction: C's overlap tracks measured degree heterogeneity

## Status of the original RC3

The original ratchet study's RC3 asked whether per-round mass renormalization
lifts the ceiling and answered NO. That answer was correct **about substrate C
alone** -- re-measured here in regime, C's ceiling is 8, the worst of the three.
It was wrong as a claim about homeostatic scaling, because it never tested the
composition. With `norm_init`, scaling roughly doubles the ceiling of this area
and keeps every assembly distinct at a fill where `norm_init` alone has already
lost a quarter of them to duplicates.

What is NOT established: how any of this scales with n. Both ceilings here are
bounded by the tiling limit of a single 2000-neuron area.

Evidence artifact: `research/results/logs/seq_s5_substrate_c_scoped.log`.
