# PREREG: is the word-problem arc's assembly DECAY the substrate-C merger?

Registered before data. Experiment: `research/experiments/seq_organ_substrate.py`.

## The prediction, and why it is sharp

[[arc-training-is-not-batchable]] recorded a measured and unexplained shape:
the Z60 arc's identical-assembly fraction across presentations climbs to a peak
around presentation 5-8 and then **degrades**.

    pres  2:  11/120 identical    pres  8:  68/120 (0.567)
    pres  3:   0/120              pres 10:  39/120 (0.325)
    pres  5:  69/120 (0.575)      pres 12:  14/120 (0.117)   <- degrading

Three properties of that measurement are exactly the conditions
`PREREG_substrate_ceiling.md` has now characterized:

1. it was taken with **`synaptic_scaling=True` and `norm_init` OFF** -- substrate
   C alone, whose assemblies merge at 6-16x the chance floor;
2. the arc was **SATURATED** (w = 19999/20000) throughout -- the same rows/n ~ 1
   regime where `norm_init` alone loses a quarter of its assemblies to exact
   duplicates and substrate C loses far more;
3. it was **one seed**.

And substrate C is *worse* the sparser the fiber (6.0x chance at p=0.5 against
15.9x at p=0.124), while this organ runs a p=0.05 brain.

**So the hypothesis is that the decay is not a fact about deep training at all.
It is the merger, on a real organ.** If so, `norm_init` alongside
`synaptic_scaling` should remove it -- the same composition that took pairwise
overlap from 7.5x chance to the floor and roughly doubled the M-ceiling.

This is a transfer test. A substrate result that does not move a real organ is
a fact about toys.

## Design

Z60 word problem, `seq_s5_word_problem.build`, arm `trained`.

**`organ_p=0.5`, not the default 0.40.** k=70 gives kp = 35.0 against the
floor 3 ln(20000) = 29.7. The default 0.40 gives kp = 28.0, which is BELOW
floor -- the violation `regime_audit` printed on every run of this organ for
weeks. A null measured there is not evidence about the mechanism.

`w_max=None` (unclamped), `presentations=15` (the registered protocol),
refraction at the organ default, seeds 42-45.

Four substrate arms, the same four the ceiling study used:

    NONE  norm_init=False  synaptic_scaling=False   <- the organ's CURRENT default
    B     norm_init=True   synaptic_scaling=False
    C     norm_init=False  synaptic_scaling=True    <- what the decay was measured on
    G     norm_init=True   synaptic_scaling=True

## Readouts

Stability comes from `saved_winners` via `_stability_from_saved`, imported
from `seq_refraction_stability` rather than re-implemented.

* **identical fraction** per presentation; its PEAK, its TERMINAL value, and
  `decay = terminal / peak` -- the shape is the phenomenon, so a single
  terminal number would not reproduce it.
* **distinct fraction of the 120 arc assemblies** at the final presentation.
  This is the merger readout, and it is the one the original measurement did
  not have. Mean overlap is nearly blind to partial collapse: 256 items on ~58
  distinct assemblies still reads at the chance floor.
* **mean pairwise overlap across transitions**, in units of chance (k/n_arc =
  70/20000 = 0.0035).
* **trajectory accuracy** at L=100, because a substrate that stabilizes
  assemblies and does not help the task has not helped.

## Bars

**O1 (regime).** Every cell has kp >= 3 ln n_arc, asserted in code.

**O2 (the defect reproduces).** Substrate C's `decay` is below 0.80 -- the
arc's stability really does fall away from its own peak. If O2 FAILS the
phenomenon is not present at these settings and O3-O5 say nothing; report that
and stop.

**O3 (THE CLAIM).** `paired_delta(G terminal, C terminal)` beats 0.

**O4 (merger on the organ).** G's arc distinct-fraction has CI-low >= 0.90
while C's has CI-high < 0.90.

**O5 (does it matter).** `paired_delta(G accuracy, C accuracy)` beats 0.

**O6 (against the organ's CURRENT default).** `paired_delta(G terminal, NONE
terminal)` beats 0. O3 could pass while G is merely undoing damage C caused;
O6 asks whether the composition beats what the organ ships with today.

All bars judged on CONFIDENCE BOUNDS, orderings as PAIRED per-seed deltas.
A delta straddling zero is INCONCLUSIVE, reported as such.

## Disclosure: overlap with PREREG_refraction_stability R5

That note's R5 compares `synaptic_scaling` ON against OFF at equal refraction
with `norm_init=False` -- which is exactly this study's NONE-vs-C contrast, at
the organ default refraction. **This study will therefore answer R5 as a side
effect, before the refraction study runs.** Recording that in advance so the
result cannot later be presented as an independent confirmation. Whatever the
NONE-vs-C contrast shows here will be written into the refraction note, and R5
must not be re-scored as new evidence when that study runs.

The two studies remain distinct questions: refraction varies
`refracted_strength` and holds the substrate fixed; this varies the substrate
and holds refraction at the organ default.

## What each outcome means

* O2 passes, O3+O4 pass -> the arc's decay IS the merger, and it is fixable.
  The batchability precondition may then be reachable, which would reopen
  [[arc-training-is-not-batchable]] -- but batching would still need its own
  registration and must not be smuggled in as a perf change.
* O2 passes, O3 fails -> the decay is real and NOT the merger. The substrate
  account does not transfer to this organ; say so plainly and leave
  [[arc-training-is-not-batchable]] standing.
* O3 passes, O5 fails -> stability without capability. Report as stability,
  never as a task result.
* O6 fails -> the composition is not better than what the organ already does;
  no default change should be proposed on this evidence.
