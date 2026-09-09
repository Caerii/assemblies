# The P600's SCALE tracks how much of the area happens to be materialised

#104 / #32 / #102. Three arms, then a fourth, and the fourth is the one that
matters. The ordering of the ERP contrast is robust. Its MAGNITUDE is not a
property of the model at all -- it moves with a lazy-materialisation
implementation detail, which is why every absolute threshold built on it has
gone stale.

## The measurements

`calibrate_erp_thresholds`, SENTENCES, paired per seed, on top of #80 and #103.

**Three arms (10 seeds, `erp_growth_neutrality.log`)** -- what `read_only()`
actually bundles:

    metric      A frozen          B no_recruit      C read_only       A-B        B-C
    grew        1482.7 +/- 87.6   120.0 +/- 0.0     120.0 +/- 0.0     +1362.7    0.0000
    p600_auc    0.9722 +/- 0.063  1.0000 +/- 0.0    1.0000 +/- 0.0    -0.0278    0.0000
    p600_span   0.0061 +/- 0.0013 0.0084 +/- 0.0014 0.0084 +/- 0.0014 -0.0022    0.0000
    p600_gap    0.0022 +/- 0.0006 0.0046 +/- 0.0005 0.0046 +/- 0.0005 -0.0024    0.0000

B and C are BIT-IDENTICAL on every metric and every seed. Restoring winners is
host hygiene with zero effect on the measurement, so `read_only()`'s two halves
are not equally interesting and only recruitment moves numbers.

**Four arms (5 seeds, `erp_full_substrate.log`)** -- adding the substrate the AC
actually specifies, every neuron materialised:

    metric        frozen [shipped]  no_recruit        full [AC]          full-frozen
    materialized  18083 +/- 111     16640 +/- 57      141030 +/- 0       +122947   CHANGED
    p600_auc      1.0000 +/- 0.0    1.0000 +/- 0.0    0.9222 +/- 0.1799  -0.0778   no change
    p600_span     0.0065 +/- 0.0011 0.0087 +/- 0.0015 0.0011 +/- 0.0003  -0.0054   CHANGED
    p600_gap      0.0025 +/- 0.0004 0.0046 +/- 0.0004 0.0002 +/- 0.0001  -0.0023   CHANGED
    n400_auc      0.8889 +/- 0.1951 0.8667 +/- 0.1799 0.8222 +/- 0.2092  -0.0667   no change

The prediction recorded in the script before running was
`gap(full) < gap(frozen) < gap(no_recruit)`, monotone in how much of the area
competes. That is exactly what happened: **0.0002 < 0.0025 < 0.0046**.

## What it means, stated carefully

**The ORDERING survives.** p600 AUC goes 1.000 -> 0.922 with a CI that spans
zero change and a lower bound of 0.742, still well above the 0.5 null. The
grammatical/violation contrast is real on the full substrate. That is the claim
the ERP work exists to make and it is intact.

**The MAGNITUDE does not, and it is not a small effect.** The absolute gap falls
12.5x between the shipped probe and the AC's substrate, on a quantity already
confined to 0.7% of [0,1].

**The prime suspect is the normaliser, not the model.** `_self_recurrent_energy`
returns `pre_kwta_total / area.w`, and `w` is the MATERIALISED COUNT -- an
artifact of lazy instantiation, not a model parameter. Between these arms `w`
grows 7.8x (18083 -> 141030) and the gap falls 12.5x, the span 5.9x. Those are
the same order. A rank statistic is invariant to a per-seed common divisor,
which is precisely why AUC barely moved while everything absolute collapsed.

I am NOT claiming the ratio proves it: 7.8 vs 12.5 vs 5.9 do not match closely
enough to call it arithmetic, and the numerator's candidate set changes between
arms too. The claim is narrower and sufficient: **the P600's scale is a function
of how much of the area has been materialised, so no threshold on it can mean
anything stable.**

### There are exactly two sites, and one was flagged in July

    erp/adapters.py:107     w = max(int(brain.areas[area].w), 1)   # _self_recurrent_energy
    binding.py:312          v / max(int(brain.areas[a].w), 1)      # input_drive

`input_drive` IS the P600 measurement (adapters calls it), so this is one
divisor reached by two paths rather than two independent choices. Its docstring
explains the intent -- normalise per candidate neuron so areas of different
sizes are commensurable -- and the intent is right. `w` is simply the wrong
stand-in for "how many candidates", because it counts every neuron ever
materialised rather than anything the model defines.

A session note from 2026-07-27 already recorded "input_drive normalizes by w
(total recruited neurons), not k -- confirms self-energy divisor bug". The
observation was correct and eight days old. What was missing was the
consequence: that it makes the metric's SCALE track training history, which is
what kills the threshold. An observation without its consequence does not
protect anything.

## This is the third and deepest defect in the same metric

The other two are in [[erp_metric_is_clipped]]:

1. `p600_excess` is clipped against the GRAMMATICAL MEDIAN, so the null arm sits
   on a 0.0 floor and Cohen's d inflates when noise falls.
2. The raw quantity is SATURATED, using 0.7% of [0,1].
3. **(this note)** and its scale is set by `w`, an implementation detail.

(3) explains (2) mechanically. `p600 = 1 - drive/w`; as `w` grows, `drive/w`
falls, and every condition is pushed toward 1.0. **The saturation is the
normaliser.** It was never a normalisation constant that needed retuning -- it
was the choice of denominator.

And (3) is why `P600_EXCESS_MARGIN = 0.152` is 11.9x above anything observable.
The constant was set when areas were small and the quantity was unbounded churn.
Every subsequent training made `w` larger and the metric smaller. The threshold
did not drift away from the data; **the data drifted away from the threshold, by
construction, every time the parser grew.**

## MEASURED: no denominator fixes it, and that is the finding

`erp_denominator_invariance.log`, n=3000 k=30 p=0.05, five seeds, three
materialisation levels on one controlled projection. Six candidate statistics
off the SAME sum, scored by max/min across levels (1.00 = the number stopped
moving):

    statistic         lazy       half       full    max/min
    w              0.08474    0.01914    0.77660      40.58   [shipped]
    k              0.11892    0.11892    1.08901       9.16
    cand           0.08474    0.01914    0.01089       7.78
    null           0.05650    0.01276    0.00726       7.78
    topk/mean      1.23106    5.47940    9.56747       7.77
    max/mean       1.61388    7.17839   12.52743       7.76

    candidates          42        188       3000

**Every scale-free candidate lands on 7.76-7.78.** `cand`, `null`, `topk/mean`
and `max/mean` are four structurally different statistics -- two are ratios of
two quantities drawn from the SAME pool, so pool size cancels algebraically --
and they drift by the same factor. A quantity whose normalisation cannot matter
still moves, so **the drift is not in the normalisation.**

Two corrections to my own reasoning, both worth keeping:

* I nominated `null = total / (count * k * p)` as the principled favourite. It
  is `cand` divided by `k*p`, **a constant**, so its drift is IDENTICAL to
  `cand`'s by construction. Dividing by a constant cannot change a ratio of
  means. The prediction was not merely wrong, it was unfalsifiable.
* `w` is the worst by a wide margin (40.58), which the earlier note got right.
  But replacing it buys a factor of five, not a fix.

### What is actually happening: the pool cannot express concentration

Read the `topk/mean` row. In the lazy arm the top-k drive is **1.23x** the pool
mean; at full materialisation it is **9.57x**. The trained assembly does not
stand out at all when the area is lazily materialised -- and the reason is
arithmetic:

    lazy pool = 42 candidates, k = 30   ->   71% OF THE POOL IS THE ASSEMBLY

There is nothing for the assembly to stand out FROM. The mean is dominated by
the very neurons the statistic is trying to distinguish. Exactly the vacuity
that `Stability.trustworthy` guards in `parse_errors` (pool <= k makes the
k-cap unable to move), reached independently from the ERP side on the same day.

**So the requirement is a POOL, not a divisor.** Any concentration measure needs
`pool >> k` before it means anything, and the shipped probe runs at pool/k ~ 1.4.

A residual remains and should not be swept up: between 188 and 3000 candidates
`topk/mean` still moves 1.75x. That is the sampler changing the DISTRIBUTION,
not just the count -- [[sampler-merges-at-low-load]] and
[[sampler-is-the-whole-discrepancy]] from the ERP side. Pool size explains most
of the 7.8x; it does not explain all of it.

### An `area.w` desync found in passing

In the `full` arm `area.w` reads **42** while the candidate vector is **3000**:
`engine.materialize_area()` does not update the `Area` descriptor's `w`. That
is why the shipped `w` divisor reads 0.777 there -- a sum over 3000 candidates
divided by 42. Another instance of [[two-index-spaces-compact-vs-neuron-id]]'s
sibling, `.w` meaning two things, and it inflates the shipped statistic by ~70x
in exactly the arm that is supposed to be ground truth.

## AND UNDER ALL OF IT: THE VIOLATION ARM PROBES AN AREA THAT DOES NOT EXIST

Found while wiring the pool-ratio guard. On a PRISTINE fork, before any parse:

    area           self-fiber extent   materialised
    ROLE_PATIENT   960                 675
    VP             0                   0
    NP             0                   0

    _self_recurrent_energy(brain, "VP")  ->  0.000000
        total 0.0 over 0 candidates, while VP holds 30 winners

**`VP` has never been built.** Its self-fiber has zero columns, so the
projection carries no drive, sums nothing, and returns exactly zero. `p600 =
1 - energy`, so the VP arm reads a CONSTANT 1.0 regardless of the sentence.

This sits beneath the other two findings and largely subsumes them:

  * the `area.w` divisor is wrong -- still true, still worth fixing;
  * no divisor helps, because the pool cannot express concentration -- still
    true, measured, and the reason a 5x improvement looked like a fix;
  * **but one arm of the contrast is structurally empty**, and no choice of
    statistic recovers information from a fiber with no columns.

It also explains the saturation more directly than the denominator does. The
violation arm sits pinned near 1.0 because its energy is ZERO, not because the
metric is compressed against a ceiling.

And it is the missing half of [[erp-p600-sign-inverted]], which recorded that
the two arms "measure DIFFERENT AREAS (ROLE_PATIENT vs VP)" and that "the
documented untrained-pathway mechanism never runs". This is WHY it never ran.

Classic [[silent-no-op-dead-fibers]]: zero drive, k-WTA still returns k
winners, nothing raises, and the number reads like a measurement. Pinned as
xfail(strict) in `tests/test_erp_probe_is_alive.py` with ROLE_PATIENT as the
live positive control, so "everything reads zero" cannot pass for a fix.

## RESOLVED: it is a METRIC/ARCHITECTURE MISMATCH, not a broken fiber

Read the blocks directly (`engine._area_conns[src][dst]` -- nested, not
tuple-keyed, which is why an earlier probe of mine returned None and produced a
wrong theory about frozen row extents):

    fiber                          shape        synapses
    VERB_CORE    -> VP           (2543, 480)      50032
    SUBJ         -> VP            (296, 480)       4360
    OBJ          -> VP            (120, 656)       2736
    VP           -> VP                (0, 0)           0
    ROLE_PATIENT -> ROLE_PATIENT    (960, 960)     24173

**VP is a well-built area with three healthy afferent fibers. The only fiber it
lacks is self-recurrence -- and self-recurrence is the only fiber the P600
metric reads.**

Nothing is broken. `_self_recurrent_energy` asks for a quantity the parser's
architecture never builds for VP. The parser has no `VP -> VP` in
`_build_circuit`; `train_phrases` uses it, but only for sentences WITH an
object, and only on the direct training path.

AND THAT IS THE ASYMMETRY BETWEEN THE ARMS. The grammatical arm probes
ROLE_PATIENT, which HAS a self-fiber because `_pregrow_role_pathways`
explicitly opens `{core: [role], role: [role]}`. The violation arm probes VP,
which never gets one. [[erp-p600-sign-inverted]] recorded that "the two arms
measure DIFFERENT AREAS"; the reason it mattered is that one of those areas has
the probed fiber and the other does not. The contrast was never between two
comparable measurements.

### The fix is to measure a fiber that exists

Not to build `VP -> VP`. That was tried and reverted: pre-growing it inverted
seed 42 to `p600_auc = 0.444`, below the null, because the bootstrap recruits
neurons that never win and displaces the real assembly.

`binding.input_drive` already computes AFFERENT drive -- what an area receives
from its sources -- and VERB_CORE -> VP carries 50,032 synapses. That quantity
is well defined for BOTH arms and needs no new structure. Whether it separates
grammatical from violation is an open empirical question and must be measured
paired over seeds before adoption, with seed 42 in the set, since that is the
seed that caught the last attempt.

## Consequences

**For #104.** The fix is not a better margin. It is a denominator that is a
property of the MODEL rather than of the run. The drive into a role area comes
from `k` firing neurons through a fiber of density `p`, so `k` (or `k*p`, or the
count of contributing sources) are all defensible; `w` is not, because nothing
in the calculus knows about it -- in the AC an area has a fixed `n` and there is
no such thing as "materialised". Whatever is chosen must be checked for
`w`-invariance by re-running the four-arm table, which is now cheap.

**For every absolute ERP number in this repository.** They were measured at one
materialisation level and do not transfer to another. Rank statistics
(`*_auc`) do transfer and are unaffected. This is the third time a "result" here
turned out to be a property of the apparatus, and the second time in two days
([[backbone-fingerprint-gap]], [[erp_probe_isolation]]'s retracted attenuation).

**For [[sampler-is-the-whole-discrepancy]].** That note says bypassing the
sampler makes sparse equal explicit. This is the same boundary reached from the
ERP side, and [[sampler-error-changes-sign-across-arms]]'s warning -- an A/B is
only safe when the manipulation leaves recruitment alone -- turns out to apply
to the shipped probe comparison itself, where BOTH arms touch recruitment.

**What is NOT concluded.** That the full-substrate arm is the "true" value to
publish. It is the AC's substrate, but the metric is broken in the ways above at
every materialisation level, so 0.0002 is not a corrected magnitude -- it is
evidence that magnitudes here are not yet meaningful. Fix the denominator first,
then measure.
