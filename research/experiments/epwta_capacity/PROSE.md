# E%-WTA vs fixed-k WTA: memory capacity of one area, measured where the E% window does not collapse

Generated from `research/experiments/epwta_capacity/`:
`regime_map.py` (Phase 1, the drive-regime map), `shared_substrate.py`
(multi-assembly storage on one shared connectome), `capacity.py` (Phase 2,
V<=128), `beta_control.py` (Phase 3, plasticity control), `deep.py`
(Phase 4, V<=512), `analyze.py` (aggregation). Raw data in the
`results_*.json` files. Regenerate this file's tables with
`python -m research.experiments.epwta_capacity.analyze`; the prose lives in
`PROSE.md`.

`neural_assemblies/` was treated as strictly read-only: selection
(`_select`), potentiation (`_potentiate`) and density (`assembly_density`)
are imported from `neural_assemblies/assembly_calculus/epwta.py`, so the
dynamics here are the library's, not a re-implementation. Only the wiring
around them (one shared stimulus pool instead of a private matrix per item)
is local to this experiment.

---

## Why this experiment exists

`research/experiments/capacity/` asked whether E%-WTA has higher or lower
capacity than fixed-k WTA and correctly **refused to answer**: at its
k_s=30, p=0.05 feedforward drive both window rules settled at a mean
assembly size of 1.1-1.6 neurons and identification was at chance in all six
cells. Any capacity number from that regime measures a degenerate selection
window, not a capacity property.

This experiment does two things that report could not: it **maps the drive
regime** in which E%-WTA forms assemblies at all, and then runs the capacity
comparison **inside** that regime, with fixed-k arms matched to the E% arms'
own emergent size.

---

## Summary

**The comparison is now answerable, and the answer is that E%-WTA has lower
capacity than fixed-k WTA at matched assembly size — but only because
plasticity closes the E% window. With no plasticity the two are equal.**

**1. The E% window needs dense drive, and the requirement is quantitative
(Tables 1-2).** Sweeping p_s x k_s on a fresh substrate, the paper's
`epsilon` rule reaches a usable assembly (mean size >= the paper's
`min_size` of 6, formation rate >= 0.7) only when `k_s * p_s >= ~40`, i.e.
when a neuron receives at least ~40 active afferents so that
`mu/sigma ~ sqrt(k_s p_s) >= ~6`. Below that it collapses: 1.4 +/- 0.5
neurons at p_s=0.1, k_s=30. The scale-invariant `sigma` rule is usable over
almost the entire grid, down to p_s=0.05, k_s=30 (6.3 +/- 2.7, formed 0.70).
This is the mechanism `winner_policies.py` predicts, measured.

**2. The prior experiment's regime was simply outside the map.** k_s=30,
p=0.05 gives `k_s * p_s = 1.5`. Nothing could have formed there. The
non-collapsed operating point used below is p_s=0.5, k_s=100, n=1000, where
`epsilon` gives 10.5 +/- 4.3 (formed 0.90) and `sigma` gives 28.2 +/- 13.9
(formed 1.00) on a fresh substrate.

**3. Fixed-k wins at matched size, by 4-16x in stored items (Tables 3-4,
7).** Storing items sequentially on one shared substrate, with capacity
defined as the largest tested V at which identification stays >= 0.9:

| arm (V=1 size) | V_cap, no metaplasticity | V_cap, metaplasticity 5 |
|---|---|---|
| epsilon (11.6) | 32 | 128 |
| topk k=9 (size-matched to epsilon) | **192** (6.0x) | **>=512** (>=4.0x) |
| sigma (29.3) | 16 | 32 |
| topk k=25 (size-matched to sigma) | **96** (6.0x) | **>=512** (>=16x) |

Every E% arm is beaten by the fixed-k arm matched to its own emergent size,
by 4-16x. Both fixed-k arms do eventually fail (topk9 at meta=0 falls to
identification 0.28 by V=384, topk25 to 0.06 by V=192), so this is a
comparison of two finite capacities, not a ceiling artefact.
The paired per-seed tests (Table 7) put epsilon 0.71 below topk9 at V=128
(meta=0) and sigma 0.46 below topk25. The effect is not a size artefact:
epsilon's assemblies at V=128 are *smaller* than topk9's (6.9 vs 9.0) and
smaller assemblies are the easier case — topk30 outperforms topk9 nowhere.

**4. The mechanism is a load-induced window collapse, and it is visible in
the read-out (Table 3).** Mean stored assembly size decays monotonically with
the number of items stored: epsilon 11.6 -> 6.9 and sigma 29.3 -> 5.7 over
V = 2 -> 128, with the fraction of stored assemblies below `min_size` rising
to 0.52 and 0.77. Retrieval is worse than storage, because the retrieval
window is recomputed on the *current* substrate: at V=128 `sigma` retrieves
**1.2 neurons** on average against a stored assembly of 5.7. Multiplicative
Hebbian potentiation grows `h_max` faster than the bulk of `h`, and a window
pinned to a fraction of the peak (or to a standard deviation that the same
outliers inflate) keeps narrowing. Fixed-k is immune by construction: it
retrieves exactly k neurons regardless of what the weights have become.

**5. The deficit is entirely plasticity-induced (Table 8).** At beta=0 --
assemblies read off a static substrate, nothing written -- all four arms
identify at 0.99-1.00 with V=64, and epsilon holds 19.5 neurons, sigma 65.5.
Raising beta collapses E%-WTA monotonically while fixed-k barely moves:

| beta | 0.0 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| epsilon: size / ident | 19.5 / 1.00 | 11.0 / 0.99 | 9.8 / 0.93 | 7.8 / 0.75 |
| sigma: size / ident | 65.5 / 0.99 | 14.7 / 0.32 | 11.9 / 0.16 | 8.0 / 0.05 |
| topk9: size / ident | 9.0 / 1.00 | 9.0 / 1.00 | 9.0 / 1.00 | 9.0 / 1.00 |
| topk25: size / ident | 25.0 / 1.00 | 25.0 / 1.00 | 25.0 / 1.00 | 25.0 / 0.99 |

So E%-WTA is not intrinsically a worse code. It is a worse *writable* code:
the same plasticity that stores an item deforms the statistic the selection
rule depends on.

**6. Metaplasticity buys a lot but does not change the ranking (Tables 3-4).**
The Fusi-style cascade `beta/(1 + meta|w|)` quadruples epsilon's capacity
(V_cap 32 -> 128) and doubles sigma's (16 -> 32), and it is exactly the
predicted mechanism: it holds epsilon's assembly size at 12.8 out to V=128
instead of letting it decay to 6.9, and it holds the retrieved size at 11.1
instead of 2.9. It also raises fixed-k, and by more (topk25 96 -> >=512, topk9
192 -> >=512, topk30 64 -> >=128). **Fixed-k still leads at matched size in
every cell, and the gap widens rather than closes.** This
generalises the module docstring's 0.52 -> 0.83 first-of-10 retrieval figure:
metaplasticity is a capacity knob for both rules, not an E%-specific rescue.

**7. E%-WTA forgets retrogradely and catastrophically; fixed-k does not
(Table 5).** At V=128, meta=0, recovery of the first fifth of the items
against the last fifth is 0.31 vs 0.90 for epsilon and 0.08 vs 0.91 for
sigma, against 0.89 vs 0.91 (flat) for topk9. Old items are not overwritten
so much as made *unreadable*: the window that could resolve a 29-neuron
assembly no longer exists once the substrate has been written 128 times.
This is a distinct failure mode from the overlap-driven forgetting fixed-k
shows at its own limit (topk25 at V=128: 0.46 early vs 0.89 late, with
pairwise overlap risen from 0.055 to 0.133).

**8. Pairwise overlap tells the same story with a different instrument
(Table 5).** sigma's stored assemblies go from 0.095 overlap (chance 0.027)
at V=2 to 0.461 (chance 0.004) at V=128 -- a 115x excess -- while topk9 sits
at 0.026 against chance 0.009 across the whole range. The E% arms do not
merely shrink; the survivors converge on the same small set of
rich-get-richer neurons.

---

## What this implies for the language stack

* **Do not swap fixed-k for E%-WTA in the lexical core areas.** In the one
  place where the emergent rule has a clear claim -- letting assembly size be
  set by the data rather than by a hyperparameter -- it costs 4-16x in items
  stored per area at matched size, and it fails in the worst possible way:
  silently, by making the oldest entries unreadable rather than by refusing
  to store new ones. The parser's lexical areas are exactly the shared-
  substrate, many-items, plasticity-on regime measured here.

* **If E%-WTA is used, metaplasticity is not optional.** `epsilon` with
  `metaplasticity=5` is the only E% configuration in this study that survives
  V=128 (ident 0.94, size held at 12.8). Without it, epsilon is done by V=64
  and sigma by V=32. The knob belongs in any curriculum that engages the
  emergent rule.

* **Prefer `epsilon` to `sigma` under load, and `sigma` on fresh drive.**
  The regime map favours `sigma` (it forms assemblies almost everywhere,
  `epsilon` needs `k_s p_s >= ~40`), but the capacity result reverses that:
  sigma's larger assemblies overlap sooner and its window collapses faster
  (V_cap 16 vs 32 at meta=0; 32 vs 128 at meta=5). The two rules are good at
  different things -- sigma for one-shot formation in sparse drive, epsilon
  for sequential storage in dense drive -- and nothing here recommends a
  single default.

* **The useful export from E%-WTA is the diagnostic, not the selector.**
  Emergent assembly size is a *read-out of how loaded a substrate is*: it
  decays monotonically with V, and the collapse fraction (assemblies below
  `min_size`) tracks identification failure closely. Running an E%-WTA probe
  alongside a fixed-k area would give the curriculum a cheap saturation
  signal -- something `w/n` demonstrably does not provide
  (`capacity/REPORT.md`, finding 5) -- without paying the capacity cost.

* **A concrete next test.** The deficit vanishes at beta=0, so the question
  is whether a *decaying* plasticity schedule (or per-fiber gating, or the
  `refractory`/`synaptic_scaling` mechanisms flagged in `capacity/REPORT.md`)
  recovers E%-WTA's capacity while keeping beta>0 learning. Metaplasticity is
  one point on that axis and it already recovers 4x.

---

## Method notes and threats to validity

* **Shared substrate, and why the wiring matters.** All items drive one
  n=1000 area through one shared 1000-row stimulus pool; item i is a fixed
  random 100-row subset, so subsets overlap by ~10% and the Hebbian update
  for item i does touch afferents that item j depends on. Handing each item
  its own `stim_weights` -- or its own disjoint slice of rows -- would make
  the feedforward path private and hide forgetting entirely, the trap
  documented in `capacity/REPORT.md` finding 6.
* **Size matching is by V=1 emergent mean, and drifts.** topk9 and topk25 are
  matched to epsilon's and sigma's fresh-substrate sizes (11.6 and 29.3
  measured on the shared substrate at V=2). Because the E% arms shrink under
  load and fixed-k does not, the match degrades as V grows -- in the
  direction that *favours* E%-WTA (its assemblies become smaller than the
  fixed-k comparison, and smaller assemblies interfere less). Assembly size
  is reported next to every capacity number for this reason.
* **Convergence, not iteration count, is held fixed.** Every arm runs the
  same Eq. 8-9 loop and stops when the firing set is stationary and
  synchronized, so different policies take different numbers of plastic
  steps (epsilon ~9, sigma ~15, topk ~5 at this operating point). Equalising
  the iteration budget instead would equalise total potentiation but change
  the paper's formation criterion; this was not run.
* **Retrieval is 3 non-plastic rounds**, matching
  `epwta.recover_assembly`, and never writes to the substrate.
* **Formation rate is reported but not used as a filter.** Items that fail
  Eq. 8-11 (usually `too_small`, or `density_below_area`) are still stored
  and still counted in every metric. Filtering to formed-only assemblies
  would flatter the E% arms by discarding exactly the collapsed cases that
  constitute the finding.
* **Identification uses cosine overlap** `|A_i n R| / sqrt(|A_i||R|)` against
  all stored templates, with chance 1/V. This normalisation is needed because
  assembly sizes vary within an E% run; it reduces to the usual overlap
  fraction for fixed-k.
* **Statistics.** 10 seeds (regime map), 8 (V<=128 capacity), 6 (beta
  control), 4 (V<=512). Mean +/- sd throughout; one-sample tests against
  explicit nulls via `research.experiments.base.ttest_vs_null`, which returns
  `significant=False` with a `degenerate` marker at zero variance -- the
  ceiling-bound fixed-k cells (identification 1.00 in every seed) are
  reported that way, not as p=0. Arm-vs-arm comparisons use `paired_ttest`
  on the seed-matched values.
* **One area size, one operating point for the capacity run.** n=1000,
  p_s=0.5, k_s=100, pool=1000, beta=0.01, p_i=0.2, w_inh=-0.2. The regime map
  covers p_s x k_s but only on a fresh substrate; how the *usable region*
  moves as the substrate fills is measured only at this one point (it
  shrinks, finding 4). Scaling in n was not run.
* **Recovery is not a capacity metric for the E% arms.** Once an assembly has
  collapsed to one or two neurons it is trivially "recovered", so E% recovery
  is non-monotone in V (sigma, meta=0: 0.91 at V=2, 0.52 at V=64, back to 0.85
  at V=512 against a stored size of 2.3). `V_cap` therefore uses first-drop
  semantics -- the largest V for which *every* earlier checkpoint is above
  threshold -- and identification, which is size-normalised and monotone, is
  the headline metric. The topk30/meta=5 recovery cell reads `2` only because
  its recovery sits at 0.68-0.70, just under the 0.7 threshold, at every V.

* **`V_cap` is a grid quantity**, the largest tested checkpoint with mean
  identification >= 0.9, so it is quantised to
  {2,4,8,16,32,64,96,128,192,256,384,512} and `>=512` means the run never
  fell below threshold.
