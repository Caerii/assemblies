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
