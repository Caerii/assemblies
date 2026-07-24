# Can an explicit recruitment mechanism defeat Hebbian rich-get-richer?

Generated from `research/experiments/recruitment/`:
`recruitment_mechanisms.py` (sweep), `analyze.py` (aggregation),
`diagnose_synaptic_scaling.py` and `smoke.py` (mechanism engagement checks).
Raw data: `results_recruitment_shard*.json` (32 cells x 5 seeds, sharded only
to get a larger scheduler share on a contended host) and
`results_beta_pass.json` (12 cells x 5 seeds). Regenerate this file with
`python -m research.experiments.recruitment.analyze`; the prose lives in
`PROSE.md` and every table below is recomputed from the JSON on each run.

This is a direct follow-up to `research/experiments/capacity/REPORT.md`, whose
closing recommendation was to test two mechanisms that are present in the
codebase but were never engaged: `refracted` suppression and
`synaptic_scaling`. The harness is the same shared-substrate lexicon model
(one `PHON -> LEX` connectome for the whole vocabulary, k=30, p=0.05, 6
training rounds, 400 synthetic words, 5 seeds); `_drive`, `_probe_checkpoint`
and `build_phon_patterns` are imported from `capacity/lexicon_capacity.py`
rather than re-implemented, and the baseline cells reproduce the prior numbers
to within noise (V* = 17 / 22 / 32 here against 17 / 22 / 33 there; final
w/n = 0.586 / 0.394 / 0.174 against 0.599 / 0.399 / 0.179; capacity exponent
0.29 +/- 0.03 against 0.29).

---

## Summary

**Yes -- `refracted` suppression restores the capacity exponent, and then
some. It does it by overriding winner selection rather than by making it fair,
and the price is that the area saturates and can no longer be read out.**

**1. Weak refracted suppression buys capacity but not scaling.** At
beta=0.05, strengths of 0.005-0.05 raise V* significantly at every n
(n=10000: 32 -> 45, +13.2 +/- 3.3, p=0.0009) but leave the *slope* untouched:

| refracted strength | 0 | 0.005 | 0.02 | 0.05 | 0.15 | 0.5 |
|---|---|---|---|---|---|---|
| capacity exponent | 0.29 | 0.31 | 0.32 | 0.29 | 0.60 | **1.15** |

Up to rs=0.05 this is a level shift, not a change of scaling law: a bigger
area still buys `n^0.3` words. Only when the penalty becomes comparable to the
whole synaptic drive does the exponent move.

**2. At rs=0.5 the exponent is 1.15 +/- 0.03 -- and the mechanism has stopped
being a bias correction.** Every seed at every n runs the area to the hard
combinatorial wall (`w/n` = 0.96 / 0.99 / 0.997), and V* coincides with the
number of words actually learned: 11 / 36 / 152. Recruit fraction over the
tail of the run is now *above* the coupon-collector null, not below it
(+0.241, +0.279, +0.067; all p < 0.001). Pairwise overlap between stored
assemblies falls **below chance** (0.005 against 0.010 at n=3000, d = -12.5;
0.002 against 0.003 at n=10000). The area is no longer competing for winners
at all -- it is tiling, and tiling slightly better than random. That is not
"Hebbian rich-get-richer defeated"; it is Hebbian selection switched off and
replaced by an anti-incumbency scan.

**3. The best result is at low plasticity: beta=0.01 with rs=0.05.** This is
the cell the prior report was implicitly asking for -- keep beta > 0 and
recover the beta=0 exponent. It does better than that:

| beta=0.01 | V* n=1000 | V* n=3000 | V* n=10000 | exponent | final w/n @ 10k |
|---|---|---|---|---|---|
| rs=0 (control) | 38 +/- 3 | 60 +/- 7 | 125 +/- 10 | 0.52 | 0.523 |
| rs=0.05 | 14 +/- 1 | 65 +/- 3 | 207 +/- 28 | **1.16 +/- 0.06** | 0.997 |

(The beta=0 reference from the prior study is 0.84.) Words actually learned
before the wall: 14 / 112 / 359, an exponent of 1.39. And the representations
are transformed: at n=10000, V=300, mean pairwise overlap between stored
assemblies is **0.012 against a baseline 0.448** (chance 0.003), and
identification accuracy **0.92 +/- 0.03 against 0.25 +/- 0.05**. Same
plasticity rate, same area, same vocabulary; 37x less interference.

**4. The cost is read-out, and it is severe.** Filling the area is the same
event as destroying the ability to probe it. Retrieval in this engine drives
the area afresh, so it needs k never-fired neurons to be available; once
`w/n -> 1` that is no longer true. In the beta=0.01, rs=0.05 cells only
21 of 100 (n=3000) and 88 of 300 (n=10000) words could be probed at all, and
the probes that succeed are the *earliest* ones in each checkpoint sweep
(each probe materialises a few more neurons on the clone). **The
discriminability numbers in finding 3 are therefore measured on the first
~30% of the vocabulary, not on all of it, and must be read as such.** At
beta=0.05, rs=0.5 the same thing happens (probes ok 15/25 at n=3000, 57/100 at
n=10000, 0/149+ at the final checkpoint). Whether this is a property of the
model or of the sparse engine's lazy materialisation is a real open question:
a dense simulation of a full area would simply select among materialised
neurons instead of raising `Remaining size of area too small to sample k new
winners`. Either way, in *this* engine an area driven to w/n > 0.99 is a
write-only area.

**5. There is a usable middle.** At beta=0.05, rs=0.15, n = 3000 and 10000,
all 400 words train, all 400 probe successfully, and the mechanism still buys
a lot: V* 22 -> 45 and 32 -> 86 (+22.8 and +54.0, p <= 1.1e-05), pairwise
overlap 0.780 -> 0.619 and 0.780 -> 0.624, exponent over the uncensored
3000 -> 10000 range 0.32 -> 0.55. Identification is flat to slightly worse
(0.585 -> 0.605 at n=3000, 0.597 -> 0.461 at n=10000). So roughly *double the
scaling exponent with the read-out intact*, at the price of some
identification at large n. At n=1000 the same setting already hits the wall
(all 5 seeds, at V = 22 +/- 2), which is the whole story in miniature: the
right strength is a function of n, and there is no single value that is safe
across the range.

**6. The mechanism does not abolish the recruitment shutdown; it postpones
it.** Table 8: at beta=0.05 and rs=0.15, n=10000, observed recruit fraction
by eighths of the vocabulary is 0.54, 0.08, 0.00, 0.00, ... against a null of
0.72, 0.48, 0.45, ... Recruitment is still at zero from word ~100 onward while
55% of the area has never fired. The bias accumulates on the neurons that fire
most, but Hebbian potentiation accumulates on the same neurons faster; the
crossing point just moves later. Only rs=0.5 escapes, and it escapes by
hitting the wall before the crossing point exists.

**7. Refracted bias is a recruitment pressure and *not* a read-out
distortion -- a clean null.** The accumulated `_cumulative_bias` is inherited
by the probe clone, and early assemblies carry more of it than late ones, so
it could have been confounding the forgetting gradient. Repeating the final
probe on a clone with the bias zeroed changes identification by at most 0.007
and retrieval self-overlap by at most 0.002, in every cell where probing works
at all (Table 4). Whatever the mechanism does, it does it during storage.

**8. `synaptic_scaling` is inert in this configuration, for a locatable
reason.** In the feedforward model it changes nothing: paired delta V* is
+0.2 +/- 0.4, +0.0 +/- 1.9, +1.8 +/- 2.6 (all ns), overlap and identification
are unchanged, and the recruit-fraction curve is identical to baseline to two
decimals (Table 8). This is not a tuning failure. `_normalize_area_columns`
skips any fiber whose source has `rows = self._areas[src].w <= 0`, and `PHON`
is an *explicit* area whose mirror state in the sparse engine keeps `w = 0`.
`diagnose_synaptic_scaling.py` instruments the call and confirms it: 59 calls
on the `PHON -> LEX` fiber, `src.w > 0` on **0** of them. The only fiber it can
reach here is `LEX -> LEX` in recurrent mode (50 calls, `src.w > 0` on 50).

**9. Where synaptic scaling *can* act, it repairs some of the damage the
recurrent fiber does -- but does not restore capacity.** With `LEX -> LEX`
engaged, scaling raises V* by +3.4, +6.6, +9.8 (p = 0.0026, 0.0016, 0.0014)
and lifts the recurrent exponent from 0.10 to 0.20 -- still well below the
plain feedforward 0.29, and far below beta=0's 0.84. It also costs
identification at the largest area (0.458 -> 0.275 at n=10000, V=400). So the
documented status holds: the per-fiber setpoint is doing something real and
directionally sensible to runaway potentiation, and it is not a capacity
mechanism.

**10. LRI (finite-memory refractory suppression) is a weaker version of the
same trade.** `refractory_period=20, inhibition_strength=0.5` raises V* to
21 / 35 / 47 (exponent 0.36 over three points, 0.24 over the uncensored two)
and lowers pairwise overlap 0.780 -> 0.715 at n=10000. Pushing it to
`strength=2.0` at n=3000 drives w/n to 0.981 and V* to 53, cuts pairwise
overlap to 0.566 -- and destroys the oldest memories completely: retrieval
self-overlap for the first decile falls to **0.045** (baseline 0.570). Because
the penalty decays, it cannot build the standing barrier that makes rs=0.5
tile; because it applies within a word's own 6 training rounds, it fights the
assembly it is trying to form.

**11. Every mechanism tested pays the same hidden tax: churn.** Neurons
materialised per neuron that survives into the stored assembly is ~5 in *every*
cell (4.1-6.5), including the baseline. Each word throws off roughly five
assemblies' worth of first-time winners across its 6 training rounds and keeps
one. That is why capacity at the wall is 11 / 36 / 152 words when `n/k` is
33 / 100 / 333 -- about a third of the nominal slot count. The mechanisms
convert transient churn into *permanent* consumption, because a neuron that
fires once during round 2 and is then discarded is nonetheless materialised
and nonetheless penalised forever after.

**12. At beta=0.2 the mechanism is a complete null.** rs=0.05 against rs=0:
exponent 0.15 vs 0.14, V* 15/16/21 vs 14/16/20, overlap 0.784/0.794/0.819 vs
0.767/0.793/0.824, identification 0.714/0.710/0.671 vs 0.752/0.750/0.655. A
fixed per-firing penalty is simply negligible against drive that has been
potentiated by (1.2)^r. The mechanism's strength has to be indexed to beta,
which is the same statement as finding 1 from the other direction.

---

## What this implies for scaling emergent capability

* **The capacity exponent is recoverable, so the `n^0.29` ceiling is not a
  law of the architecture.** It is a consequence of unregulated Hebbian
  incumbency, and a single scalar penalty on recently-active neurons is enough
  to move it to 1.15. That is the most important result here: the earlier
  finding "scaling `n` is the most expensive and least effective lever" is
  conditional on the plasticity regime, not fundamental.

* **But recruitment pressure and plasticity strength are one knob, not two.**
  The useful setting is (low beta, moderate rs): beta=0.01 + rs=0.05 gives
  exponent 1.16 and 37x less assembly overlap; beta=0.2 + rs=0.05 gives
  nothing at all; beta=0.05 + rs=0.5 gives the exponent but a write-only area.
  Any curriculum that wants this has to schedule the two together, and the
  right rs depends on n as well -- at rs=0.15 the n=1000 area is destroyed
  while the n=10000 area is at its best operating point.

* **`n/k` is still the wrong capacity budget, now for a second reason.**
  Before, recruitment stopped at 10-51% of the slots. Now recruitment can be
  forced all the way to the wall, and the wall itself lands at ~1/3 of `n/k`
  because multi-round training materialises ~5x more neurons than it keeps.
  Reducing training rounds, or gating plasticity to the final round, is a
  cheaper lever than either mechanism tested here and was not varied.

* **Storage capacity and retrieval capacity have to be budgeted separately.**
  The single most surprising number in this study is the pair
  (pairwise overlap 0.012, probes ok 88/300) in the same cell. The area
  succeeded completely at keeping 359 words apart and failed completely at
  giving them back. An architecture that runs areas near w/n = 1 needs a
  read-out path that does not itself require fresh neurons.

* **Fix `synaptic_scaling`'s reachability before re-testing it.** The
  documented objection to the mechanism (per-fiber setpoint cancels net
  potentiation) is about correctness; what this study found is a separate,
  simpler bug in *scope*: it silently cannot see any fiber whose source is an
  explicit area, because the sparse engine's mirror of an explicit area never
  advances `w`. Any future test of homeostatic scaling on a stimulus-driven
  lexical area will measure nothing until that is addressed. (Not fixed here:
  `neural_assemblies/` was treated as read-only.)

---

## Method notes and threats to validity

* **Shared substrate, deliberately.** One `PHON -> LEX` connectome for the
  whole vocabulary. Per-word private stimulus matrices reverse the direction
  of interference (finding 6 of the prior report) and would have hidden every
  effect measured here.
* **Baselines were re-run, not reused.** Every mechanism cell has an rs=0 /
  scaling-off cell at the same n, beta and mode inside this sweep, and the
  paired tests in Tables 1 and 5 use those, not the prior study's numbers.
* **The `probes ok` column gates every discriminability claim.** Where it is
  below V, identification and retrieval are computed on the subset that could
  be probed, which is biased toward words probed early in the sweep. Cells
  with `probes ok` = 0 are reported as `n/a` rather than as zeros.
* **V* is right-censored at the pool wall.** `recruit_horizon` cannot exceed
  the number of words a run got through, so in any cell with `exh > 0` the
  reported V* is a lower bound. Those rows are marked, `V learned` is reported
  alongside, and Table 2 carries a two-point 3000 -> 10000 exponent for the
  cases where only n=1000 is censored.
* **Statistics.** 5 seeds per cell; mean +/- sd throughout;
  `research.experiments.base.ttest_vs_null`, which returns
  `significant=False` with a `degenerate` marker at zero variance. Exponents
  are reported both as a single fit to the seed means and as the mean +/- sd
  of per-seed fits; the two agree everywhere to within 0.01.
* **The host is nondeterministic across processes.** Two runs of the same
  config in separate processes do not produce identical winner sequences even
  with a fixed `Brain(seed=)` and `PYTHONHASHSEED=0` (checked explicitly while
  diagnosing `synaptic_scaling`). This is why no claim here rests on a
  single-run comparison.
* **Single parameter point per unswept axis.** p=0.05, k=30, 6 training
  rounds, TopK only. E%-WTA was not re-tested: the prior report established
  that in this drive regime it settles at an assembly size of 1.1-1.6 neurons
  and forms no assemblies at all, so a recruitment comparison against it would
  be measuring the degenerate selection window, not the rule.
* **Refracted strength was swept over 0.005-0.5, six values.** The transition
  from "level shift" to "wall" happens between 0.05 and 0.5 and is not finely
  resolved; the exponent between rs=0.15 and rs=0.5 is interpolated by two
  points only.
* **`neural_assemblies/` was not modified.** The one instrumentation of
  engine internals (`diagnose_synaptic_scaling.py`) monkeypatches
  `_normalize_area_columns` inside the experiment process only.

---

## Tables

### Table 1 -- Refracted suppression: capacity and recruitment

`rs` is `refracted_strength`, the per-firing increment to a neuron's cumulative input penalty. `V*` is the recruitment horizon (no 10-word window past it has mean recruit fraction >= 0.05). `V learned` is how many words trained before the sparse engine ran out of never-fired neurons; `exh` counts seeds that hit that hard wall (where V* is right-censored). `tail recruit` / `tail null` are the mean recruit fraction and the `1 - w/n` coupon-collector null over the LAST HALF of the words each run got through.

`churn` is neurons materialised per neuron that survives into the stored assembly, summed over the run: 1.0 means every first-time winner stayed in the assembly, >1 means the assembly never settled within its 6 training rounds.

`delta V*` is paired by seed against the rs=0 cell at the same n.

| n | rs | V learned | exh/5 | V* | delta V* vs rs=0 | final w/n | churn | tail recruit | tail null | tail excess | excess test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.0 | 400 +/- 0 | 0 | 17 +/- 1 | -- | 0.586 +/- 0.013 | 4.97 +/- 0.12 | 0.00 +/- 0.00 | 0.41 +/- 0.01 | -0.414 +/- 0.013 | p=2.1e-07 d=-32.67 * |
| 1000 | 0.005 | 400 +/- 0 | 0 | 17 +/- 1 | +0.4 +/- +1.7 p=0.62 d=0.24 ns | 0.614 +/- 0.029 | 4.92 +/- 0.06 | 0.00 +/- 0.00 | 0.39 +/- 0.03 | -0.386 +/- 0.029 | p=7.4e-06 d=-13.39 * |
| 1000 | 0.02 | 400 +/- 0 | 0 | 18 +/- 1 | +1.8 +/- +0.8 p=0.0086 d=2.15 * | 0.663 +/- 0.041 | 5.07 +/- 0.15 | 0.00 +/- 0.00 | 0.34 +/- 0.04 | -0.337 +/- 0.041 | p=5.2e-05 d=-8.21 * |
| 1000 | 0.05 | 400 +/- 0 | 0 | 23 +/- 2 | +6.6 +/- +1.5 p=0.00062 d=4.35 * | 0.760 +/- 0.038 | 4.97 +/- 0.12 | 0.00 +/- 0.00 | 0.24 +/- 0.04 | -0.240 +/- 0.038 | p=0.00015 d=-6.24 * |
| 1000 | 0.15 | 22 +/- 2 | 5 | 21 +/- 2 | +4.8 +/- +1.5 p=0.0019 d=3.24 * | 0.967 +/- 0.001 | 4.99 +/- 0.12 | 0.06 +/- 0.02 | 0.09 +/- 0.02 | -0.026 +/- 0.006 | p=0.00074 d=-4.16 * |
| 1000 | 0.5 | 11 +/- 1 | 5 | 11 +/- 1 | -6.0 +/- +1.0 p=0.00018 d=-6.00 * | 0.961 +/- 0.008 | 4.70 +/- 0.16 | 0.46 +/- 0.06 | 0.21 +/- 0.02 | 0.241 +/- 0.049 | p=0.00038 d=4.94 * |
| 3000 | 0.0 | 400 +/- 0 | 0 | 22 +/- 1 | -- | 0.394 +/- 0.014 | 5.48 +/- 0.07 | 0.00 +/- 0.00 | 0.61 +/- 0.01 | -0.606 +/- 0.014 | p=6.7e-08 d=-43.51 * |
| 3000 | 0.005 | 400 +/- 0 | 0 | 25 +/- 1 | +3.2 +/- +2.0 p=0.025 d=1.56 * | 0.446 +/- 0.030 | 5.41 +/- 0.04 | 0.00 +/- 0.00 | 0.55 +/- 0.03 | -0.554 +/- 0.030 | p=2.2e-06 d=-18.17 * |
| 3000 | 0.02 | 400 +/- 0 | 0 | 25 +/- 2 | +3.4 +/- +2.2 p=0.026 d=1.55 * | 0.463 +/- 0.032 | 5.47 +/- 0.09 | 0.00 +/- 0.00 | 0.54 +/- 0.03 | -0.537 +/- 0.032 | p=3e-06 d=-16.86 * |
| 3000 | 0.05 | 400 +/- 0 | 0 | 30 +/- 2 | +7.8 +/- +1.3 p=0.00018 d=5.98 * | 0.522 +/- 0.026 | 5.63 +/- 0.11 | 0.00 +/- 0.00 | 0.48 +/- 0.03 | -0.478 +/- 0.026 | p=2.1e-06 d=-18.34 * |
| 3000 | 0.15 | 400 +/- 0 | 0 | 45 +/- 1 | +22.8 +/- +1.3 p=2.6e-06 d=17.49 * | 0.869 +/- 0.010 | 5.99 +/- 0.08 | 0.00 +/- 0.00 | 0.13 +/- 0.01 | -0.131 +/- 0.010 | p=7.1e-06 d=-13.56 * |
| 3000 | 0.5 | 36 +/- 1 | 5 | 36 +/- 1 | +14.0 +/- +1.6 p=3.8e-05 d=8.85 * | 0.989 +/- 0.001 | 4.45 +/- 0.13 | 0.39 +/- 0.03 | 0.11 +/- 0.01 | 0.279 +/- 0.023 | p=1e-05 d=12.41 * |
| 10000 | 0.0 | 400 +/- 0 | 0 | 32 +/- 1 | -- | 0.174 +/- 0.005 | 5.77 +/- 0.08 | 0.00 +/- 0.00 | 0.83 +/- 0.01 | -0.826 +/- 0.005 | p=3.8e-10 d=-158.14 * |
| 10000 | 0.005 | 400 +/- 0 | 0 | 35 +/- 2 | +2.6 +/- +2.3 p=0.065 d=1.13 ns | 0.196 +/- 0.010 | 5.75 +/- 0.05 | 0.00 +/- 0.00 | 0.80 +/- 0.01 | -0.804 +/- 0.010 | p=7e-09 d=-76.62 * |
| 10000 | 0.02 | 400 +/- 0 | 0 | 38 +/- 3 | +6.0 +/- +3.4 p=0.017 d=1.77 * | 0.212 +/- 0.011 | 5.76 +/- 0.10 | 0.00 +/- 0.00 | 0.79 +/- 0.01 | -0.788 +/- 0.011 | p=8.6e-09 d=-72.76 * |
| 10000 | 0.05 | 400 +/- 0 | 0 | 45 +/- 3 | +13.2 +/- +3.3 p=0.00091 d=3.94 * | 0.234 +/- 0.014 | 5.81 +/- 0.15 | 0.00 +/- 0.00 | 0.77 +/- 0.01 | -0.766 +/- 0.014 | p=2.7e-08 d=-54.74 * |
| 10000 | 0.15 | 400 +/- 0 | 0 | 86 +/- 5 | +54.0 +/- +4.4 p=1.1e-05 d=12.23 * | 0.546 +/- 0.017 | 5.88 +/- 0.04 | 0.00 +/- 0.00 | 0.45 +/- 0.02 | -0.454 +/- 0.017 | p=4.4e-07 d=-27.22 * |
| 10000 | 0.5 | 152 +/- 3 | 5 | 148 +/- 3 | +116.0 +/- +2.5 p=5.6e-08 d=45.50 * | 0.997 +/- 0.000 | 4.82 +/- 0.03 | 0.10 +/- 0.01 | 0.03 +/- 0.00 | 0.067 +/- 0.011 | p=0.00016 d=6.22 * |

### Table 2 -- The headline: capacity exponent `d log V* / d log n`

Least-squares slope of `log V*` against `log n` over n = 1000 / 3000 / 10000. Prior work (`capacity/REPORT.md` Table 4b) measured 0.84 at beta=0 and 0.29 at beta=0.05; the question is whether a mechanism recovers the beta=0 exponent while keeping beta > 0. `exp (per-seed)` refits the slope separately for each seed, so the sd is real rather than a fit residual. Rows where some seed hit the hard pool wall are marked `censored` -- V* there is a lower bound and the exponent is not interpretable.

`V learned` is words trained before the hard pool wall (400 = the vocabulary ran out first, i.e. right-censored by the protocol rather than by the model), and `exp(V learned)` is the same slope fitted to it. When a mechanism drives the area to the wall, `V*` and `V learned` coincide and BOTH are the honest capacity; when it does not, `V learned` is pinned at 400 and only `V*` means anything. Read the pair, not either alone.

`exp 3k->10k` is the two-point slope over the uncensored part of the range: at n=1000 a strong mechanism hits the pool wall and V* is a lower bound, which drags the three-point fit. Where the three-point and two-point slopes disagree, the two-point one is the safer read.

| condition | V* n=1000 | V* n=3000 | V* n=10000 | exponent | exp (per-seed) | exp 3k->10k | V learned 1k/3k/10k | exp(V learned) | wall |
|---|---|---|---|---|---|---|---|---|---|
| beta=0.05 refracted rs=0.0 | 17 +/- 1 | 22 +/- 1 | 32 +/- 1 | 0.29 | 0.29 +/- 0.03 | 0.32 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 refracted rs=0.005 | 17 +/- 1 | 25 +/- 1 | 35 +/- 2 | 0.31 | 0.31 +/- 0.04 | 0.27 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 refracted rs=0.02 | 18 +/- 1 | 25 +/- 2 | 38 +/- 3 | 0.32 | 0.32 +/- 0.04 | 0.35 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 refracted rs=0.05 | 23 +/- 2 | 30 +/- 2 | 45 +/- 3 | 0.29 | 0.29 +/- 0.04 | 0.36 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 refracted rs=0.15 | 21 +/- 2 | 45 +/- 1 | 86 +/- 5 | 0.60 | 0.60 +/- 0.04 | 0.55 | 22 +/- 2 / 400 +/- 0 / 400 +/- 0 | 1.24 | 5/15 |
| beta=0.05 refracted rs=0.5 | 11 +/- 1 | 36 +/- 1 | 148 +/- 3 | 1.15 | 1.15 +/- 0.03 | 1.18 | 11 +/- 1 / 36 +/- 1 / 152 +/- 3 | 1.16 | 15/15 |
| beta=0.05 synaptic_scaling (ff) | 17 +/- 0 | 22 +/- 1 | 34 +/- 1 | 0.31 | 0.31 +/- 0.02 | 0.37 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 recurrent (rec), no scaling | 14 +/- 1 | 16 +/- 1 | 17 +/- 1 | 0.10 | 0.10 +/- 0.01 | 0.07 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 recurrent + synaptic_scaling | 17 +/- 1 | 23 +/- 1 | 27 +/- 3 | 0.20 | 0.20 +/- 0.05 | 0.15 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0/15 |
| beta=0.05 LRI period=20 strength=0.5 | 21 +/- 2 | 35 +/- 2 | 47 +/- 1 | 0.36 | 0.36 +/- 0.04 | 0.24 | 324 +/- 169 / 400 +/- 0 / 400 +/- 0 | 0.09 | 1/15 |

### Table 3 -- What it costs: overlap, identification, forgetting

Measured at the last checkpoint every seed of the cell reached (`V meas`). `pairwise` is mean overlap between stored assemblies (chance = k/n). `ident` is the fraction of words retrieved nearest to their OWN stored assembly (chance = 1/V). `retr early` / `retr late` are retrieval self-overlap for the first and last tenth of the vocabulary: early << late is retrograde forgetting.

`probes ok` is how many of the V words could be probed at all: retrieval needs the area to still have k never-fired neurons available, so a mechanism that fills the area destroys its own read-out. 0 there means every metric to its right is undefined.

| n | condition | V meas | probes ok | pairwise (chance) | pairwise test | ident (chance) | ident test | retr early | retr late |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | baseline | 400 | 400 +/- 0/400 | 0.766 +/- 0.022 (0.030) | p=2e-07 d=33.23 * | 0.684 +/- 0.052 (0.003) | p=8.1e-06 d=13.09 * | 0.676 +/- 0.013 | 0.997 +/- 0.002 |
| 1000 | synaptic_scaling | 400 | 400 +/- 0/400 | 0.760 +/- 0.017 (0.030) | p=7.5e-08 d=42.31 * | 0.696 +/- 0.052 (0.003) | p=7.4e-06 d=13.38 * | 0.676 +/- 0.023 | 0.997 +/- 0.001 |
| 1000 | rs=0.005 | 400 | 400 +/- 0/400 | 0.745 +/- 0.030 (0.030) | p=7.3e-07 d=23.97 * | 0.721 +/- 0.083 (0.003) | p=4.2e-05 d=8.64 * | 0.671 +/- 0.014 | 0.996 +/- 0.002 |
| 1000 | rs=0.02 | 400 | 400 +/- 0/400 | 0.752 +/- 0.009 (0.030) | p=5.4e-09 d=81.57 * | 0.606 +/- 0.063 (0.003) | p=2.8e-05 d=9.59 * | 0.606 +/- 0.014 | 0.996 +/- 0.002 |
| 1000 | rs=0.05 | 400 | 400 +/- 0/400 | 0.727 +/- 0.018 (0.030) | p=1.1e-07 d=38.56 * | 0.653 +/- 0.041 (0.003) | p=3.8e-06 d=15.83 * | 0.542 +/- 0.023 | 0.996 +/- 0.001 |
| 1000 | rec | 400 | 400 +/- 0/400 | 0.808 +/- 0.032 (0.030) | p=6.8e-07 d=24.34 * | 0.630 +/- 0.088 (0.003) | p=9.1e-05 d=7.13 * | 0.747 +/- 0.053 | 0.995 +/- 0.004 |
| 1000 | rec+scaling | 400 | 400 +/- 0/400 | 0.816 +/- 0.030 (0.030) | p=4.9e-07 d=26.46 * | 0.642 +/- 0.035 (0.003) | p=2.2e-06 d=18.15 * | 0.715 +/- 0.033 | 0.996 +/- 0.002 |
| 3000 | baseline | 400 | 400 +/- 0/400 | 0.780 +/- 0.032 (0.010) | p=7e-07 d=24.17 * | 0.585 +/- 0.098 (0.003) | p=0.00019 d=5.92 * | 0.570 +/- 0.037 | 0.997 +/- 0.002 |
| 3000 | synaptic_scaling | 400 | 400 +/- 0/400 | 0.781 +/- 0.023 (0.010) | p=1.8e-07 d=34.17 * | 0.577 +/- 0.112 (0.003) | p=0.00033 d=5.14 * | 0.582 +/- 0.030 | 0.997 +/- 0.002 |
| 3000 | LRI 5/0.5 | 400 | 400 +/- 0/400 | 0.775 +/- 0.015 (0.010) | p=3.8e-08 d=50.18 * | 0.547 +/- 0.029 (0.003) | p=1.9e-06 d=18.76 * | 0.535 +/- 0.028 | 0.998 +/- 0.002 |
| 3000 | LRI 20/0.5 | 400 | 400 +/- 0/400 | 0.730 +/- 0.032 (0.010) | p=9.1e-07 d=22.66 * | 0.546 +/- 0.057 (0.003) | p=3e-05 d=9.45 * | 0.381 +/- 0.049 | 0.995 +/- 0.002 |
| 3000 | LRI 20/2.0 | 400 | 400 +/- 0/400 | 0.566 +/- 0.020 (0.010) | p=3.9e-07 d=27.97 * | 0.419 +/- 0.053 (0.003) | p=6.2e-05 d=7.83 * | 0.045 +/- 0.005 | 0.997 +/- 0.002 |
| 3000 | rs=0.005 | 400 | 400 +/- 0/400 | 0.769 +/- 0.028 (0.010) | p=4.3e-07 d=27.37 * | 0.617 +/- 0.100 (0.003) | p=0.00016 d=6.13 * | 0.550 +/- 0.037 | 0.996 +/- 0.002 |
| 3000 | rs=0.02 | 400 | 400 +/- 0/400 | 0.746 +/- 0.033 (0.010) | p=1e-06 d=22.08 * | 0.666 +/- 0.100 (0.003) | p=0.00012 d=6.63 * | 0.530 +/- 0.023 | 0.997 +/- 0.002 |
| 3000 | rs=0.05 | 400 | 400 +/- 0/400 | 0.721 +/- 0.018 (0.010) | p=1e-07 d=39.28 * | 0.656 +/- 0.034 (0.003) | p=1.7e-06 d=19.27 * | 0.443 +/- 0.020 | 0.996 +/- 0.003 |
| 3000 | rs=0.15 | 400 | 400 +/- 0/400 | 0.619 +/- 0.024 (0.010) | p=5.4e-07 d=25.77 * | 0.605 +/- 0.033 (0.003) | p=2.2e-06 d=18.24 * | 0.189 +/- 0.021 | 0.996 +/- 0.001 |
| 3000 | rs=0.5 | 25 | 15 +/- 1/25 | 0.005 +/- 0.000 (0.010) | p=9.7e-06 d=-12.52 * | 0.516 +/- 0.106 (0.040) | p=0.00055 d=4.51 * | 0.273 +/- 0.060 | 0.100 +/- 0.062 |
| 3000 | rec | 400 | 400 +/- 0/400 | 0.858 +/- 0.029 (0.010) | p=3.4e-07 d=28.90 * | 0.536 +/- 0.103 (0.003) | p=0.00032 d=5.18 * | 0.779 +/- 0.050 | 0.997 +/- 0.002 |
| 3000 | rec+scaling | 400 | 400 +/- 0/400 | 0.836 +/- 0.028 (0.010) | p=3.1e-07 d=29.64 * | 0.520 +/- 0.081 (0.003) | p=0.00014 d=6.40 * | 0.661 +/- 0.045 | 0.996 +/- 0.003 |
| 10000 | baseline | 400 | 400 +/- 0/400 | 0.780 +/- 0.024 (0.003) | p=2.1e-07 d=32.63 * | 0.597 +/- 0.090 (0.003) | p=0.00012 d=6.61 * | 0.535 +/- 0.030 | 0.997 +/- 0.002 |
| 10000 | synaptic_scaling | 400 | 400 +/- 0/400 | 0.777 +/- 0.019 (0.003) | p=8e-08 d=41.63 * | 0.630 +/- 0.049 (0.003) | p=9.1e-06 d=12.70 * | 0.533 +/- 0.020 | 0.999 +/- 0.001 |
| 10000 | LRI 20/0.5 | 400 | 400 +/- 0/400 | 0.715 +/- 0.022 (0.003) | p=2.2e-07 d=32.26 * | 0.560 +/- 0.039 (0.003) | p=5.5e-06 d=14.42 * | 0.313 +/- 0.044 | 0.996 +/- 0.001 |
| 10000 | rs=0.005 | 400 | 400 +/- 0/400 | 0.769 +/- 0.024 (0.003) | p=2.2e-07 d=32.30 * | 0.591 +/- 0.081 (0.003) | p=8.5e-05 d=7.24 * | 0.494 +/- 0.039 | 0.997 +/- 0.001 |
| 10000 | rs=0.02 | 400 | 400 +/- 0/400 | 0.743 +/- 0.023 (0.003) | p=2.2e-07 d=32.18 * | 0.667 +/- 0.085 (0.003) | p=6.3e-05 d=7.81 * | 0.457 +/- 0.034 | 0.997 +/- 0.001 |
| 10000 | rs=0.05 | 400 | 400 +/- 0/400 | 0.706 +/- 0.018 (0.003) | p=9.5e-08 d=39.89 * | 0.657 +/- 0.036 (0.003) | p=2.1e-06 d=18.43 * | 0.377 +/- 0.034 | 0.996 +/- 0.001 |
| 10000 | rs=0.15 | 400 | 400 +/- 0/400 | 0.624 +/- 0.024 (0.003) | p=5.4e-07 d=25.86 * | 0.461 +/- 0.105 (0.003) | p=0.00062 d=4.36 * | 0.149 +/- 0.010 | 0.997 +/- 0.001 |
| 10000 | rs=0.5 | 100 | 57 +/- 2/100 | 0.002 +/- 0.000 (0.003) | p=3.5e-05 d=-9.09 * | 0.583 +/- 0.040 (0.010) | p=5.9e-06 d=14.20 * | 0.049 +/- 0.024 | 0.104 +/- 0.028 |
| 10000 | rec | 400 | 400 +/- 0/400 | 0.897 +/- 0.020 (0.003) | p=6.3e-08 d=44.25 * | 0.458 +/- 0.128 (0.003) | p=0.0013 d=3.57 * | 0.816 +/- 0.032 | 0.993 +/- 0.013 |
| 10000 | rec+scaling | 400 | 400 +/- 0/400 | 0.883 +/- 0.024 (0.003) | p=1.3e-07 d=37.21 * | 0.275 +/- 0.131 (0.003) | p=0.0097 d=2.08 * | 0.662 +/- 0.046 | 0.996 +/- 0.003 |

### Table 4 -- Refracted bias as a read-out distortion

The cumulative bias is not cleared for retrieval by default: the clone inherits it, and an assembly learned early carries more accumulated suppression than one learned late. This table repeats the final probe on a clone whose bias has been zeroed (`Brain.clear_refracted_bias`), which separates the mechanism's effect on STORAGE from its effect on READ-OUT.

`n/a` rows are ones where EVERY probe failed: the mechanism had filled the area past the point where a retrieval drive can still find k never-fired neurons.

| n | rs | V meas | probes ok | ident (bias on) | ident (bias cleared) | retr early on | retr early cleared | retr late on | retr late cleared |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.005 | [400] | 400 +/- 0 | 0.721 +/- 0.083 | 0.720 +/- 0.083 | 0.671 +/- 0.014 | 0.671 +/- 0.014 | 0.996 +/- 0.002 | 0.995 +/- 0.002 |
| 1000 | 0.02 | [400] | 400 +/- 0 | 0.606 +/- 0.063 | 0.604 +/- 0.066 | 0.606 +/- 0.014 | 0.606 +/- 0.014 | 0.996 +/- 0.002 | 0.995 +/- 0.002 |
| 1000 | 0.05 | [400] | 400 +/- 0 | 0.653 +/- 0.041 | 0.649 +/- 0.045 | 0.542 +/- 0.023 | 0.542 +/- 0.023 | 0.996 +/- 0.001 | 0.996 +/- 0.001 |
| 1000 | 0.15 | [19, 21, 22, 23, 24] | 0 +/- 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 1000 | 0.5 | [10, 11] | 0 +/- 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 3000 | 0.005 | [400] | 400 +/- 0 | 0.617 +/- 0.100 | 0.613 +/- 0.104 | 0.550 +/- 0.037 | 0.550 +/- 0.037 | 0.996 +/- 0.002 | 0.996 +/- 0.002 |
| 3000 | 0.02 | [400] | 400 +/- 0 | 0.666 +/- 0.100 | 0.661 +/- 0.104 | 0.530 +/- 0.023 | 0.530 +/- 0.023 | 0.997 +/- 0.002 | 0.995 +/- 0.002 |
| 3000 | 0.05 | [400] | 400 +/- 0 | 0.656 +/- 0.034 | 0.653 +/- 0.034 | 0.443 +/- 0.020 | 0.443 +/- 0.020 | 0.996 +/- 0.003 | 0.996 +/- 0.002 |
| 3000 | 0.15 | [400] | 400 +/- 0 | 0.605 +/- 0.033 | 0.598 +/- 0.035 | 0.189 +/- 0.021 | 0.189 +/- 0.021 | 0.996 +/- 0.001 | 0.995 +/- 0.001 |
| 3000 | 0.5 | [35, 36, 37] | 0 +/- 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| 10000 | 0.005 | [400] | 400 +/- 0 | 0.591 +/- 0.081 | 0.591 +/- 0.081 | 0.494 +/- 0.039 | 0.494 +/- 0.039 | 0.997 +/- 0.001 | 0.997 +/- 0.002 |
| 10000 | 0.02 | [400] | 400 +/- 0 | 0.667 +/- 0.085 | 0.662 +/- 0.091 | 0.457 +/- 0.034 | 0.457 +/- 0.034 | 0.997 +/- 0.001 | 0.995 +/- 0.002 |
| 10000 | 0.05 | [400] | 400 +/- 0 | 0.657 +/- 0.036 | 0.656 +/- 0.036 | 0.377 +/- 0.034 | 0.377 +/- 0.034 | 0.996 +/- 0.001 | 0.995 +/- 0.001 |
| 10000 | 0.15 | [400] | 400 +/- 0 | 0.461 +/- 0.105 | 0.456 +/- 0.107 | 0.149 +/- 0.010 | 0.148 +/- 0.010 | 0.997 +/- 0.001 | 0.996 +/- 0.001 |
| 10000 | 0.5 | [149, 151, 153, 156] | 0 +/- 0 | n/a | n/a | n/a | n/a | n/a | n/a |

### Table 5 -- Synaptic scaling (paired against its own control)

`synaptic_scaling=True` renormalises area->area fibers. In `ff` mode the only area->area fiber is PHON->LEX; in `rec` mode LEX->LEX is added. Paired by seed against the matching scaling-off cell.

| n | mode | V* off | V* on | delta V* (test vs 0) | final w/n off | final w/n on | pairwise off | pairwise on | ident off | ident on |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | ff | 17 +/- 1 | 17 +/- 0 | +0.2 +/- +0.4 p=0.37 d=0.45 ns | 0.586 +/- 0.013 | 0.590 +/- 0.016 | 0.766 +/- 0.022 | 0.760 +/- 0.017 | 0.684 +/- 0.052 | 0.696 +/- 0.052 |
| 1000 | rec | 14 +/- 1 | 17 +/- 1 | +3.4 +/- +1.1 p=0.0026 d=2.98 * | 0.521 +/- 0.014 | 0.664 +/- 0.004 | 0.808 +/- 0.032 | 0.816 +/- 0.030 | 0.630 +/- 0.088 | 0.642 +/- 0.035 |
| 3000 | ff | 22 +/- 1 | 22 +/- 1 | +0.0 +/- +1.9 p=1 d=0.00 ns | 0.394 +/- 0.014 | 0.388 +/- 0.012 | 0.780 +/- 0.032 | 0.781 +/- 0.023 | 0.585 +/- 0.098 | 0.577 +/- 0.112 |
| 3000 | rec | 16 +/- 1 | 23 +/- 1 | +6.6 +/- +1.9 p=0.0016 d=3.39 * | 0.334 +/- 0.013 | 0.422 +/- 0.019 | 0.858 +/- 0.029 | 0.836 +/- 0.028 | 0.536 +/- 0.103 | 0.520 +/- 0.081 |
| 10000 | ff | 32 +/- 1 | 34 +/- 1 | +1.8 +/- +2.6 p=0.19 d=0.70 ns | 0.174 +/- 0.005 | 0.178 +/- 0.005 | 0.780 +/- 0.024 | 0.777 +/- 0.019 | 0.597 +/- 0.090 | 0.630 +/- 0.049 |
| 10000 | rec | 17 +/- 1 | 27 +/- 3 | +9.8 +/- +2.8 p=0.0014 d=3.53 * | 0.133 +/- 0.010 | 0.186 +/- 0.013 | 0.897 +/- 0.020 | 0.883 +/- 0.024 | 0.458 +/- 0.128 | 0.275 +/- 0.131 |

### Table 6 -- LRI (finite-memory refractory suppression)

`refractory_period` steps of linearly decaying penalty `inhibition_strength` on neurons that fired within the window. Unlike `refracted` this forgets, so it cannot accumulate an unbounded barrier -- but it also penalises the CURRENT word's own assembly during its 6 training rounds.

| n | period/strength | V learned | exh/5 | V* | final w/n | churn | tail recruit | tail null | pairwise | ident |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0/0.0 (baseline) | 400 +/- 0 | 0 | 17 +/- 1 | 0.586 +/- 0.013 | 4.97 +/- 0.12 | 0.00 +/- 0.00 | 0.41 +/- 0.01 | 0.766 +/- 0.022 | 0.684 +/- 0.052 |
| 1000 | 20/0.5 | 324 +/- 169 | 1 | 21 +/- 2 | 0.933 +/- 0.030 | 4.97 +/- 0.15 | 0.01 +/- 0.02 | 0.07 +/- 0.02 | n/a | n/a |
| 3000 | 0/0.0 (baseline) | 400 +/- 0 | 0 | 22 +/- 1 | 0.394 +/- 0.014 | 5.48 +/- 0.07 | 0.00 +/- 0.00 | 0.61 +/- 0.01 | 0.780 +/- 0.032 | 0.585 +/- 0.098 |
| 3000 | 5/0.5 | 400 +/- 0 | 0 | 24 +/- 2 | 0.509 +/- 0.035 | 6.54 +/- 0.26 | 0.00 +/- 0.00 | 0.49 +/- 0.03 | 0.775 +/- 0.015 | 0.547 +/- 0.029 |
| 3000 | 20/0.5 | 400 +/- 0 | 0 | 35 +/- 2 | 0.682 +/- 0.028 | 5.43 +/- 0.10 | 0.00 +/- 0.00 | 0.32 +/- 0.03 | 0.730 +/- 0.032 | 0.546 +/- 0.057 |
| 3000 | 20/2.0 | 400 +/- 0 | 0 | 53 +/- 5 | 0.981 +/- 0.003 | 4.06 +/- 0.07 | 0.00 +/- 0.00 | 0.02 +/- 0.00 | 0.566 +/- 0.020 | 0.419 +/- 0.053 |
| 10000 | 0/0.0 (baseline) | 400 +/- 0 | 0 | 32 +/- 1 | 0.174 +/- 0.005 | 5.77 +/- 0.08 | 0.00 +/- 0.00 | 0.83 +/- 0.01 | 0.780 +/- 0.024 | 0.597 +/- 0.090 |
| 10000 | 20/0.5 | 400 +/- 0 | 0 | 47 +/- 1 | 0.317 +/- 0.018 | 5.61 +/- 0.06 | 0.00 +/- 0.00 | 0.68 +/- 0.02 | 0.715 +/- 0.022 | 0.560 +/- 0.039 |

### Table 8 -- Recruit fraction against the `1 - w/n` null, in eighths of the vocabulary

Only cells that trained all 400 words are shown (an exhausted run has no late eighths). beta=0.05, TopK, feedforward.

| n | condition | source | 1-50 | 51-100 | 101-150 | 151-200 | 201-250 | 251-300 | 301-350 | 351-400 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | baseline | observed | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | baseline | null | 0.47 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 |
| 1000 | synaptic_scaling | observed | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | synaptic_scaling | null | 0.47 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 |
| 1000 | LRI 20/0.5 | observed | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | LRI 20/0.5 | null | 0.17 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| 1000 | rs=0.005 | observed | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | rs=0.005 | null | 0.44 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 | 0.39 |
| 1000 | rs=0.02 | observed | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | rs=0.02 | null | 0.41 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 |
| 1000 | rs=0.05 | observed | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1000 | rs=0.05 | null | 0.34 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 |
| 3000 | baseline | observed | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | baseline | null | 0.66 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 3000 | synaptic_scaling | observed | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | synaptic_scaling | null | 0.66 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 3000 | LRI 5/0.5 | observed | 0.16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | LRI 5/0.5 | null | 0.57 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 |
| 3000 | LRI 20/0.5 | observed | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | LRI 20/0.5 | null | 0.46 | 0.32 | 0.32 | 0.32 | 0.32 | 0.32 | 0.32 | 0.32 |
| 3000 | LRI 20/2.0 | observed | 0.48 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | LRI 20/2.0 | null | 0.28 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| 3000 | rs=0.005 | observed | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | rs=0.005 | null | 0.61 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 |
| 3000 | rs=0.02 | observed | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | rs=0.02 | null | 0.60 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 |
| 3000 | rs=0.05 | observed | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | rs=0.05 | null | 0.57 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 |
| 3000 | rs=0.15 | observed | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3000 | rs=0.15 | null | 0.39 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| 10000 | baseline | observed | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | baseline | null | 0.86 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| 10000 | synaptic_scaling | observed | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | synaptic_scaling | null | 0.86 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 | 0.82 |
| 10000 | LRI 20/0.5 | observed | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | LRI 20/0.5 | null | 0.77 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 | 0.68 |
| 10000 | rs=0.005 | observed | 0.23 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | rs=0.005 | null | 0.84 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| 10000 | rs=0.02 | observed | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | rs=0.02 | null | 0.84 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 |
| 10000 | rs=0.05 | observed | 0.27 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | rs=0.05 | null | 0.83 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |
| 10000 | rs=0.15 | observed | 0.54 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10000 | rs=0.15 | null | 0.72 | 0.48 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |

### Table 7 -- Does the mechanism survive a change of beta?

Second pass: the same refracted strength at beta = 0.01 and 0.2, with matched rs=0 controls.

| beta | rs | V* n=1000 | V* n=3000 | V* n=10000 | exponent | exp (per-seed) | V learned 1k/3k/10k | exp(V learned) | exh/15 |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.0 | 38 +/- 3 | 60 +/- 7 | 125 +/- 10 | 0.52 | 0.52 +/- 0.06 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0 |
| 0.01 | 0.05 | 14 +/- 1 | 65 +/- 3 | 207 +/- 28 | 1.16 | 1.15 +/- 0.06 | 14 +/- 1 / 112 +/- 5 / 359 +/- 14 | 1.39 | 15 |
| 0.2 | 0.0 | 14 +/- 1 | 16 +/- 1 | 20 +/- 1 | 0.14 | 0.14 +/- 0.06 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0 |
| 0.2 | 0.05 | 15 +/- 1 | 16 +/- 1 | 21 +/- 3 | 0.15 | 0.15 +/- 0.04 | 400 +/- 0 / 400 +/- 0 / 400 +/- 0 | -0.00 | 0 |

Consequence metrics for the beta pass, at the last common checkpoint of each cell:

| n | beta | rs | V meas | probes ok | final w/n | pairwise | ident | retr early | retr late |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0.01 | 0.0 | 400 | 400 +/- 0/400 | 0.943 +/- 0.018 | 0.568 +/- 0.026 | 0.323 +/- 0.036 | 0.295 +/- 0.031 | 0.994 +/- 0.004 |
| 1000 | 0.01 | 0.05 | (no common checkpoint: every seed hit the pool wall at a different V) | 0.966 +/- 0.001 | | | | |
| 1000 | 0.2 | 0.0 | 400 | 400 +/- 0/400 | 0.318 +/- 0.054 | 0.767 +/- 0.024 | 0.752 +/- 0.046 | 0.841 +/- 0.007 | 0.995 +/- 0.004 |
| 1000 | 0.2 | 0.05 | 400 | 400 +/- 0/400 | 0.366 +/- 0.039 | 0.784 +/- 0.031 | 0.714 +/- 0.052 | 0.849 +/- 0.018 | 0.993 +/- 0.004 |
| 3000 | 0.01 | 0.0 | 400 | 400 +/- 0/400 | 0.802 +/- 0.035 | 0.481 +/- 0.028 | 0.326 +/- 0.023 | 0.157 +/- 0.024 | 0.995 +/- 0.003 |
| 3000 | 0.01 | 0.05 | 100 | 21 +/- 4/100 | 0.989 +/- 0.001 | 0.020 +/- 0.001 | 0.900 +/- 0.052 | 0.317 +/- 0.073 | 0.567 +/- 0.057 |
| 3000 | 0.2 | 0.0 | 400 | 400 +/- 0/400 | 0.155 +/- 0.020 | 0.793 +/- 0.017 | 0.750 +/- 0.040 | 0.861 +/- 0.015 | 0.988 +/- 0.008 |
| 3000 | 0.2 | 0.05 | 400 | 400 +/- 0/400 | 0.164 +/- 0.016 | 0.794 +/- 0.019 | 0.710 +/- 0.024 | 0.842 +/- 0.014 | 0.991 +/- 0.003 |
| 10000 | 0.01 | 0.0 | 400 | 400 +/- 0/400 | 0.523 +/- 0.036 | 0.448 +/- 0.031 | 0.247 +/- 0.049 | 0.143 +/- 0.012 | 0.994 +/- 0.001 |
| 10000 | 0.01 | 0.05 | 300 | 88 +/- 25/300 | 0.997 +/- 0.000 | 0.012 +/- 0.001 | 0.922 +/- 0.027 | 0.115 +/- 0.027 | 0.619 +/- 0.100 |
| 10000 | 0.2 | 0.0 | 400 | 400 +/- 0/400 | 0.074 +/- 0.012 | 0.824 +/- 0.022 | 0.655 +/- 0.074 | 0.850 +/- 0.027 | 0.992 +/- 0.005 |
| 10000 | 0.2 | 0.05 | 400 | 400 +/- 0/400 | 0.083 +/- 0.004 | 0.819 +/- 0.005 | 0.671 +/- 0.048 | 0.835 +/- 0.024 | 0.991 +/- 0.011 |
