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
