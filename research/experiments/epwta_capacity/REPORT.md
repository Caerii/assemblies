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

---

## Tables

### Table 1 - Regime map: emergent assembly size (formation rate)
Fresh substrate, one assembly, n=1000, beta=0.01, 10 seeds. Cell is `mean size +/- sd (formation rate)`; formation requires all of Eq. 8-11 plus `|A| >= 6`.

#### `window="epsilon"`

| p_s \ k_s | 30 | 60 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| **0.05** | 2.4+/-1.3 (0.10) | 2.3+/-1.3 (0.00) | 4.0+/-3.0 (0.30) | 4.0+/-3.1 (0.20) | 3.4+/-2.4 (0.20) |
| **0.1** | 1.4+/-0.5 (0.00) | 2.6+/-1.3 (0.00) | 2.6+/-1.7 (0.00) | 5.6+/-2.8 (0.40) | 6.5+/-3.1 (0.40) |
| **0.2** | 3.5+/-1.2 (0.00) | 4.7+/-2.7 (0.30) | 3.3+/-2.3 (0.20) | 10.8+/-4.6 (0.90) | 12.2+/-6.0 (0.70) |
| **0.3** | 3.0+/-1.9 (0.10) | 3.8+/-2.9 (0.20) | 6.2+/-3.9 (0.40) | 9.2+/-6.6 (0.70) | 22.9+/-12.0 (0.80) |
| **0.5** | 3.5+/-2.1 (0.20) | 6.0+/-3.6 (0.50) | 10.5+/-4.3 (0.90) | 19.5+/-13.2 (0.70) | 71.0+/-29.5 (1.00) |

#### `window="sigma"`

| p_s \ k_s | 30 | 60 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| **0.05** | 6.3+/-2.7 (0.70) | 14.0+/-8.7 (0.80) | 14.0+/-11.3 (0.70) | 24.8+/-10.8 (1.00) | 22.3+/-10.6 (1.00) |
| **0.1** | 7.6+/-5.8 (0.60) | 10.6+/-4.0 (0.80) | 15.2+/-6.2 (0.90) | 27.0+/-12.8 (0.90) | 30.9+/-21.6 (0.90) |
| **0.2** | 11.7+/-8.0 (0.70) | 13.4+/-5.3 (0.90) | 21.7+/-13.4 (0.90) | 42.2+/-18.8 (1.00) | 33.4+/-14.0 (1.00) |
| **0.3** | 10.9+/-4.4 (0.90) | 18.7+/-9.9 (1.00) | 28.1+/-16.1 (1.00) | 21.2+/-14.1 (0.80) | 48.8+/-27.3 (0.90) |
| **0.5** | 9.4+/-8.6 (0.70) | 17.7+/-9.3 (0.90) | 28.2+/-13.9 (1.00) | 31.2+/-16.6 (0.90) | 50.4+/-21.2 (1.00) |

### Table 2 - Which cells are usable
`usable` = mean emergent size >= 6 (the paper's `min_size`) AND formation rate >= 0.7.
| p_s | k_s | epsilon size (formed) | sigma size (formed) | usable for epsilon? | usable for sigma? |
|---|---|---|---|---|---|
| 0.05 | 30 | 2.4 (0.10) | 6.3 (0.70) | no | **yes** |
| 0.05 | 60 | 2.3 (0.00) | 14.0 (0.80) | no | **yes** |
| 0.05 | 100 | 4.0 (0.30) | 14.0 (0.70) | no | **yes** |
| 0.05 | 200 | 4.0 (0.20) | 24.8 (1.00) | no | **yes** |
| 0.05 | 400 | 3.4 (0.20) | 22.3 (1.00) | no | **yes** |
| 0.1 | 30 | 1.4 (0.00) | 7.6 (0.60) | no | no |
| 0.1 | 60 | 2.6 (0.00) | 10.6 (0.80) | no | **yes** |
| 0.1 | 100 | 2.6 (0.00) | 15.2 (0.90) | no | **yes** |
| 0.1 | 200 | 5.6 (0.40) | 27.0 (0.90) | no | **yes** |
| 0.1 | 400 | 6.5 (0.40) | 30.9 (0.90) | no | **yes** |
| 0.2 | 30 | 3.5 (0.00) | 11.7 (0.70) | no | **yes** |
| 0.2 | 60 | 4.7 (0.30) | 13.4 (0.90) | no | **yes** |
| 0.2 | 100 | 3.3 (0.20) | 21.7 (0.90) | no | **yes** |
| 0.2 | 200 | 10.8 (0.90) | 42.2 (1.00) | **yes** | **yes** |
| 0.2 | 400 | 12.2 (0.70) | 33.4 (1.00) | **yes** | **yes** |
| 0.3 | 30 | 3.0 (0.10) | 10.9 (0.90) | no | **yes** |
| 0.3 | 60 | 3.8 (0.20) | 18.7 (1.00) | no | **yes** |
| 0.3 | 100 | 6.2 (0.40) | 28.1 (1.00) | no | **yes** |
| 0.3 | 200 | 9.2 (0.70) | 21.2 (0.80) | **yes** | **yes** |
| 0.3 | 400 | 22.9 (0.80) | 48.8 (0.90) | **yes** | **yes** |
| 0.5 | 30 | 3.5 (0.20) | 9.4 (0.70) | no | **yes** |
| 0.5 | 60 | 6.0 (0.50) | 17.7 (0.90) | no | **yes** |
| 0.5 | 100 | 10.5 (0.90) | 28.2 (1.00) | **yes** | **yes** |
| 0.5 | 200 | 19.5 (0.70) | 31.2 (0.90) | **yes** | **yes** |
| 0.5 | 400 | 71.0 (1.00) | 50.4 (1.00) | **yes** | **yes** |

### Table 3 - Capacity curves on a shared substrate
n=1000, pool=1000, k_s=100, p_s=0.5, beta=0.01, 8 seeds. Fixed-k arms are matched to each E% arm's V=1 emergent size.

#### metaplasticity = 0.0

| arm | metric | V=2 | V=4 | V=8 | V=16 | V=32 | V=64 | V=96 | V=128 |
|---|---|---|---|---|---|---|---|---|---|
| epsilon | assembly size | 11.6+/-6.0 | 11.4+/-3.9 | 10.8+/-2.8 | 9.9+/-1.6 | 9.6+/-1.5 | 9.2+/-1.1 | 8.1+/-1.1 | 6.9+/-1.0 |
| epsilon | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.98+/-0.03 | 0.86+/-0.13 | 0.59+/-0.16 | 0.29+/-0.11 |
| epsilon | recovery | 0.92+/-0.11 | 0.94+/-0.07 | 0.92+/-0.05 | 0.91+/-0.04 | 0.89+/-0.03 | 0.76+/-0.06 | 0.62+/-0.04 | 0.54+/-0.04 |
| epsilon | formed rate | 0.75+/-0.46 | 0.75+/-0.33 | 0.72+/-0.24 | 0.65+/-0.15 | 0.65+/-0.13 | 0.62+/-0.10 | 0.53+/-0.10 | 0.43+/-0.08 |
| sigma | assembly size | 29.3+/-11.2 | 28.6+/-9.8 | 25.4+/-7.1 | 22.1+/-3.4 | 16.5+/-2.6 | 10.1+/-1.8 | 7.2+/-1.2 | 5.7+/-0.9 |
| sigma | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.98+/-0.03 | 0.69+/-0.14 | 0.18+/-0.11 | 0.04+/-0.04 | 0.02+/-0.01 |
| sigma | recovery | 0.91+/-0.06 | 0.93+/-0.05 | 0.86+/-0.11 | 0.75+/-0.08 | 0.57+/-0.08 | 0.52+/-0.04 | 0.56+/-0.06 | 0.59+/-0.11 |
| sigma | formed rate | 0.94+/-0.18 | 0.97+/-0.09 | 0.92+/-0.09 | 0.88+/-0.06 | 0.73+/-0.10 | 0.43+/-0.08 | 0.29+/-0.06 | 0.22+/-0.04 |
| topk9 | assembly size | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 |
| topk9 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| topk9 | recovery | 0.93+/-0.06 | 0.92+/-0.04 | 0.93+/-0.02 | 0.92+/-0.03 | 0.93+/-0.01 | 0.92+/-0.01 | 0.91+/-0.01 | 0.90+/-0.01 |
| topk9 | formed rate | 0.88+/-0.23 | 0.88+/-0.13 | 0.83+/-0.09 | 0.82+/-0.06 | 0.85+/-0.05 | 0.85+/-0.04 | 0.86+/-0.03 | 0.85+/-0.02 |
| topk25 | assembly size | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 |
| topk25 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.54+/-0.27 |
| topk25 | recovery | 0.88+/-0.07 | 0.88+/-0.04 | 0.89+/-0.03 | 0.89+/-0.02 | 0.88+/-0.01 | 0.86+/-0.01 | 0.79+/-0.03 | 0.63+/-0.05 |
| topk25 | formed rate | 0.94+/-0.18 | 0.97+/-0.09 | 0.98+/-0.04 | 0.99+/-0.02 | 0.99+/-0.01 | 1.00+/-0.01 | 0.99+/-0.01 | 1.00+/-0.01 |
| topk30 | assembly size | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 |
| topk30 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.82+/-0.21 | 0.20+/-0.06 |
| topk30 | recovery | 0.88+/-0.04 | 0.88+/-0.02 | 0.87+/-0.02 | 0.88+/-0.02 | 0.87+/-0.01 | 0.84+/-0.01 | 0.69+/-0.05 | 0.53+/-0.02 |
| topk30 | formed rate | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |

#### metaplasticity = 5.0

| arm | metric | V=2 | V=4 | V=8 | V=16 | V=32 | V=64 | V=96 | V=128 |
|---|---|---|---|---|---|---|---|---|---|
| epsilon | assembly size | 15.9+/-9.2 | 14.9+/-6.6 | 14.4+/-3.6 | 13.6+/-1.7 | 13.3+/-1.3 | 13.4+/-0.9 | 13.0+/-0.8 | 12.8+/-0.7 |
| epsilon | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.99+/-0.01 | 1.00+/-0.01 | 0.99+/-0.01 | 0.94+/-0.06 |
| epsilon | recovery | 0.78+/-0.14 | 0.84+/-0.09 | 0.85+/-0.06 | 0.86+/-0.04 | 0.83+/-0.03 | 0.81+/-0.03 | 0.78+/-0.01 | 0.74+/-0.04 |
| epsilon | formed rate | 0.81+/-0.26 | 0.81+/-0.18 | 0.80+/-0.15 | 0.77+/-0.11 | 0.78+/-0.07 | 0.77+/-0.06 | 0.76+/-0.05 | 0.75+/-0.04 |
| sigma | assembly size | 33.6+/-11.5 | 35.8+/-9.7 | 33.9+/-6.6 | 31.6+/-3.4 | 29.6+/-3.0 | 24.4+/-2.3 | 19.3+/-1.9 | 15.7+/-1.7 |
| sigma | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.01 | 0.84+/-0.08 | 0.49+/-0.12 | 0.30+/-0.07 |
| sigma | recovery | 0.82+/-0.13 | 0.76+/-0.06 | 0.80+/-0.07 | 0.77+/-0.06 | 0.71+/-0.02 | 0.58+/-0.05 | 0.48+/-0.03 | 0.48+/-0.03 |
| sigma | formed rate | 0.75+/-0.27 | 0.78+/-0.21 | 0.81+/-0.15 | 0.80+/-0.11 | 0.83+/-0.06 | 0.79+/-0.05 | 0.72+/-0.06 | 0.60+/-0.08 |
| topk9 | assembly size | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 |
| topk9 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| topk9 | recovery | 0.79+/-0.08 | 0.80+/-0.07 | 0.80+/-0.04 | 0.78+/-0.04 | 0.79+/-0.03 | 0.79+/-0.03 | 0.79+/-0.02 | 0.79+/-0.02 |
| topk9 | formed rate | 0.69+/-0.37 | 0.78+/-0.16 | 0.81+/-0.09 | 0.77+/-0.07 | 0.81+/-0.05 | 0.84+/-0.03 | 0.84+/-0.03 | 0.84+/-0.03 |
| topk25 | assembly size | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 |
| topk25 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| topk25 | recovery | 0.71+/-0.05 | 0.71+/-0.04 | 0.71+/-0.04 | 0.71+/-0.02 | 0.71+/-0.02 | 0.71+/-0.01 | 0.70+/-0.01 | 0.70+/-0.01 |
| topk25 | formed rate | 1.00+/-0.00 | 1.00+/-0.00 | 0.98+/-0.04 | 0.99+/-0.02 | 0.99+/-0.01 | 0.99+/-0.01 | 0.99+/-0.01 | 0.99+/-0.01 |
| topk30 | assembly size | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 | 30.0+/-0.0 |
| topk30 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| topk30 | recovery | 0.70+/-0.06 | 0.68+/-0.04 | 0.70+/-0.03 | 0.68+/-0.02 | 0.68+/-0.01 | 0.69+/-0.01 | 0.69+/-0.01 | 0.69+/-0.01 |
| topk30 | formed rate | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.99+/-0.01 | 0.99+/-0.01 | 0.99+/-0.01 |

### Table 4 - Capacity summary
`V_cap` is the largest tested V whose mean stays at/above threshold; it is quantised to the checkpoint grid.
| arm | meta | max V tested | V_cap (ident >= 0.9) | V_cap (recovery >= 0.7) | size @ V_cap | ident @ max V |
|---|---|---|---|---|---|---|
| epsilon | 0.0 | 512 | 32 | 64 | 9.6 | 0.00 +/- 0.00 |
| sigma | 0.0 | 512 | 16 | 16 | 22.1 | 0.00 +/- 0.00 |
| topk9 | 0.0 | 512 | 192 | 256 | 9.0 | 0.04 +/- 0.04 |
| topk25 | 0.0 | 512 | 96 | 96 | 25.0 | 0.00 +/- 0.00 |
| topk30 | 0.0 | 128 | 64 | 64 | 30.0 | 0.20 +/- 0.06 |
| epsilon | 5.0 | 512 | 128 | 128 | 12.7 | 0.04 +/- 0.04 |
| sigma | 5.0 | 512 | 32 | 32 | 29.6 | 0.01 +/- 0.00 |
| topk9 | 5.0 | 512 | >=512 | >=512 | 9.0 | 1.00 +/- 0.00 |
| topk25 | 5.0 | 512 | >=512 | 192 | 25.0 | 0.94 +/- 0.06 |
| topk30 | 5.0 | 128 | >=128 | 2 | 30.0 | 1.00 +/- 0.00 |

### Table 5 - Forgetting gradient and pairwise overlap
`early`/`late` are the first and last fifth of the stored items. A negative gradient means EARLY items retrieve better than LATE ones (anterograde); positive means classic retrograde forgetting.
| arm | meta | V | recovery early | recovery late | gradient (late-early) | size early | size late | pairwise (chance) |
|---|---|---|---|---|---|---|---|---|
| epsilon | 0.0 | 32 | 0.85 +/- 0.09 | 0.96 +/- 0.04 | +0.11 p=0.0055 d=1.40 * | 11.2 | 8.9 | 0.033 +/- 0.015 (0.009) |
| sigma | 0.0 | 32 | 0.37 +/- 0.17 | 0.87 +/- 0.11 | +0.50 p=2.9e-05 d=3.38 * | 26.3 | 7.5 | 0.142 +/- 0.048 (0.014) |
| topk25 | 0.0 | 32 | 0.88 +/- 0.03 | 0.87 +/- 0.03 | -0.00 p=0.5 d=-0.25 ns | 25.0 | 25.0 | 0.055 +/- 0.005 (0.025) |
| topk30 | 0.0 | 32 | 0.86 +/- 0.03 | 0.87 +/- 0.03 | +0.01 p=0.63 d=0.18 ns | 30.0 | 30.0 | 0.064 +/- 0.007 (0.030) |
| topk9 | 0.0 | 32 | 0.93 +/- 0.03 | 0.95 +/- 0.02 | +0.02 p=0.015 d=1.14 * | 9.0 | 9.0 | 0.023 +/- 0.004 (0.009) |
| epsilon | 0.0 | 128 | 0.31 +/- 0.09 | 0.90 +/- 0.07 | +0.59 p=6e-06 d=4.28 * | 9.8 | 2.7 | 0.155 +/- 0.081 (0.006) |
| sigma | 0.0 | 128 | 0.08 +/- 0.03 | 0.91 +/- 0.24 | +0.83 p=2.1e-05 d=3.53 * | 19.0 | 1.0 | 0.461 +/- 0.088 (0.004) |
| topk25 | 0.0 | 128 | 0.46 +/- 0.08 | 0.89 +/- 0.01 | +0.44 p=1.7e-06 d=5.15 * | 25.0 | 25.0 | 0.133 +/- 0.028 (0.025) |
| topk30 | 0.0 | 128 | 0.27 +/- 0.04 | 0.92 +/- 0.02 | +0.64 p=1.1e-08 d=10.74 * | 30.0 | 30.0 | 0.206 +/- 0.035 (0.030) |
| topk9 | 0.0 | 128 | 0.89 +/- 0.02 | 0.91 +/- 0.02 | +0.02 p=0.013 d=1.18 * | 9.0 | 9.0 | 0.026 +/- 0.004 (0.009) |
| epsilon | 5.0 | 32 | 0.81 +/- 0.06 | 0.87 +/- 0.06 | +0.06 p=0.12 d=0.63 ns | 14.7 | 12.2 | 0.027 +/- 0.004 (0.012) |
| sigma | 5.0 | 32 | 0.62 +/- 0.11 | 0.79 +/- 0.07 | +0.17 p=0.022 d=1.03 * | 34.1 | 24.8 | 0.075 +/- 0.006 (0.027) |
| topk25 | 5.0 | 32 | 0.71 +/- 0.04 | 0.71 +/- 0.04 | -0.00 p=0.93 d=-0.03 ns | 25.0 | 25.0 | 0.041 +/- 0.004 (0.025) |
| topk30 | 5.0 | 32 | 0.69 +/- 0.04 | 0.67 +/- 0.06 | -0.02 p=0.44 d=-0.29 ns | 30.0 | 30.0 | 0.050 +/- 0.003 (0.030) |
| topk9 | 5.0 | 32 | 0.81 +/- 0.05 | 0.82 +/- 0.05 | +0.01 p=0.71 d=0.14 ns | 9.0 | 9.0 | 0.020 +/- 0.005 (0.009) |
| epsilon | 5.0 | 128 | 0.64 +/- 0.06 | 0.85 +/- 0.05 | +0.20 p=0.00028 d=2.37 * | 13.5 | 12.0 | 0.036 +/- 0.006 (0.012) |
| sigma | 5.0 | 128 | 0.15 +/- 0.03 | 0.89 +/- 0.04 | +0.74 p=1.4e-09 d=14.44 * | 30.6 | 4.6 | 0.186 +/- 0.016 (0.012) |
| topk25 | 5.0 | 128 | 0.70 +/- 0.02 | 0.70 +/- 0.02 | -0.01 p=0.55 d=-0.22 ns | 25.0 | 25.0 | 0.044 +/- 0.002 (0.025) |
| topk30 | 5.0 | 128 | 0.69 +/- 0.03 | 0.70 +/- 0.02 | +0.01 p=0.48 d=0.27 ns | 30.0 | 30.0 | 0.053 +/- 0.002 (0.030) |
| topk9 | 5.0 | 128 | 0.80 +/- 0.03 | 0.79 +/- 0.03 | -0.01 p=0.7 d=-0.14 ns | 9.0 | 9.0 | 0.018 +/- 0.001 (0.009) |

### Table 6 - Identification against chance (1/V)
| arm | meta | V | ident | chance | test vs chance |
|---|---|---|---|---|---|
| epsilon | 0.0 | 32 | 0.98 +/- 0.03 | 0.031 | p=1.2e-11 d=28.53 * |
| epsilon | 0.0 | 128 | 0.29 +/- 0.11 | 0.008 | p=0.00022 d=2.45 * |
| sigma | 0.0 | 32 | 0.69 +/- 0.14 | 0.031 | p=3e-06 d=4.74 * |
| sigma | 0.0 | 128 | 0.02 +/- 0.01 | 0.008 | p=0.051 d=0.83 ns |
| topk25 | 0.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk25 | 0.0 | 128 | 0.54 +/- 0.27 | 0.008 | p=0.0008 d=1.99 * |
| topk30 | 0.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk30 | 0.0 | 128 | 0.20 +/- 0.06 | 0.008 | p=5.9e-05 d=3.03 * |
| topk9 | 0.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk9 | 0.0 | 128 | 1.00 +/- 0.00 | 0.008 | degenerate:zero_variance |
| epsilon | 5.0 | 32 | 0.99 +/- 0.01 | 0.031 | p=3.2e-14 d=66.43 * |
| epsilon | 5.0 | 128 | 0.94 +/- 0.06 | 0.008 | p=1.3e-09 d=14.53 * |
| sigma | 5.0 | 32 | 1.00 +/- 0.01 | 0.031 | p=4.7e-15 d=87.33 * |
| sigma | 5.0 | 128 | 0.30 +/- 0.07 | 0.008 | p=1.1e-05 d=3.89 * |
| topk25 | 5.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk25 | 5.0 | 128 | 1.00 +/- 0.00 | 0.008 | degenerate:zero_variance |
| topk30 | 5.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk30 | 5.0 | 128 | 1.00 +/- 0.00 | 0.008 | degenerate:zero_variance |
| topk9 | 5.0 | 32 | 1.00 +/- 0.00 | 0.031 | degenerate:zero_variance |
| topk9 | 5.0 | 128 | 1.00 +/- 0.00 | 0.008 | degenerate:zero_variance |

### Table 7 - E%-WTA vs fixed-k, paired by seed
| comparison | meta | V | E% ident | fixed-k ident | difference | paired test |
|---|---|---|---|---|---|---|
| epsilon vs topk9 | 0.0 | 32 | 0.98 +/- 0.03 | 1.00 +/- 0.00 | -0.02 | p=0.23 d=-0.47 ns |
| sigma vs topk25 | 0.0 | 32 | 0.69 +/- 0.14 | 1.00 +/- 0.00 | -0.31 | p=0.00042 d=-2.22 * |
| epsilon vs topk25 | 0.0 | 32 | 0.98 +/- 0.03 | 1.00 +/- 0.00 | -0.02 | p=0.23 d=-0.47 ns |
| sigma vs topk9 | 0.0 | 32 | 0.69 +/- 0.14 | 1.00 +/- 0.00 | -0.31 | p=0.00042 d=-2.22 * |
| epsilon vs topk9 | 0.0 | 128 | 0.29 +/- 0.11 | 1.00 +/- 0.00 | -0.71 | p=4.4e-07 d=-6.29 * |
| sigma vs topk25 | 0.0 | 128 | 0.02 +/- 0.01 | 0.54 +/- 0.27 | -0.53 | p=0.00091 d=-1.94 * |
| epsilon vs topk25 | 0.0 | 128 | 0.29 +/- 0.11 | 0.54 +/- 0.27 | -0.26 | p=0.022 d=-1.04 * |
| sigma vs topk9 | 0.0 | 128 | 0.02 +/- 0.01 | 1.00 +/- 0.00 | -0.98 | p=3.1e-15 d=-92.81 * |
| epsilon vs topk9 | 5.0 | 32 | 0.99 +/- 0.01 | 1.00 +/- 0.00 | -0.01 | p=0.17 d=-0.54 ns |
| sigma vs topk25 | 5.0 | 32 | 1.00 +/- 0.01 | 1.00 +/- 0.00 | -0.00 | p=0.35 d=-0.35 ns |
| epsilon vs topk25 | 5.0 | 32 | 0.99 +/- 0.01 | 1.00 +/- 0.00 | -0.01 | p=0.17 d=-0.54 ns |
| sigma vs topk9 | 5.0 | 32 | 1.00 +/- 0.01 | 1.00 +/- 0.00 | -0.00 | p=0.35 d=-0.35 ns |
| epsilon vs topk9 | 5.0 | 128 | 0.94 +/- 0.06 | 1.00 +/- 0.00 | -0.06 | p=0.025 d=-1.01 * |
| sigma vs topk25 | 5.0 | 128 | 0.30 +/- 0.07 | 1.00 +/- 0.00 | -0.70 | p=2.5e-08 d=-9.51 * |
| epsilon vs topk25 | 5.0 | 128 | 0.94 +/- 0.06 | 1.00 +/- 0.00 | -0.06 | p=0.025 d=-1.01 * |
| sigma vs topk9 | 5.0 | 128 | 0.30 +/- 0.07 | 1.00 +/- 0.00 | -0.70 | p=2.5e-08 d=-9.51 * |

### Table 8 - beta control (V=64, no metaplasticity, 6 seeds)
At beta=0 nothing is written: assemblies are read off a static substrate.
| beta | arm | assembly size | ident | recovery | pairwise (chance) | formed rate |
|---|---|---|---|---|---|---|
| 0.0 | epsilon | 19.5 +/- 1.7 | 1.00 +/- 0.01 | 0.68 +/- 0.04 | 0.030 +/- 0.002 (0.017) | 0.60 +/- 0.05 |
| 0.0 | sigma | 65.5 +/- 4.1 | 0.99 +/- 0.01 | 0.53 +/- 0.02 | 0.086 +/- 0.007 (0.060) | 0.28 +/- 0.06 |
| 0.0 | topk9 | 9.0 +/- 0.0 | 1.00 +/- 0.01 | 0.75 +/- 0.02 | 0.018 +/- 0.003 (0.009) | 0.82 +/- 0.04 |
| 0.0 | topk25 | 25.0 +/- 0.0 | 1.00 +/- 0.00 | 0.63 +/- 0.01 | 0.040 +/- 0.001 (0.025) | 0.98 +/- 0.02 |
| 0.005 | epsilon | 11.0 +/- 0.4 | 0.99 +/- 0.01 | 0.82 +/- 0.04 | 0.034 +/- 0.002 (0.010) | 0.72 +/- 0.07 |
| 0.005 | sigma | 14.7 +/- 1.4 | 0.32 +/- 0.26 | 0.44 +/- 0.08 | 0.168 +/- 0.033 (0.012) | 0.68 +/- 0.09 |
| 0.005 | topk9 | 9.0 +/- 0.0 | 1.00 +/- 0.00 | 0.85 +/- 0.01 | 0.020 +/- 0.003 (0.009) | 0.83 +/- 0.05 |
| 0.005 | topk25 | 25.0 +/- 0.0 | 1.00 +/- 0.00 | 0.80 +/- 0.01 | 0.051 +/- 0.003 (0.025) | 1.00 +/- 0.01 |
| 0.01 | epsilon | 9.8 +/- 0.8 | 0.93 +/- 0.08 | 0.76 +/- 0.08 | 0.040 +/- 0.009 (0.009) | 0.70 +/- 0.06 |
| 0.01 | sigma | 11.9 +/- 1.8 | 0.16 +/- 0.15 | 0.45 +/- 0.10 | 0.227 +/- 0.070 (0.009) | 0.49 +/- 0.11 |
| 0.01 | topk9 | 9.0 +/- 0.0 | 1.00 +/- 0.00 | 0.91 +/- 0.02 | 0.022 +/- 0.003 (0.009) | 0.84 +/- 0.05 |
| 0.01 | topk25 | 25.0 +/- 0.0 | 1.00 +/- 0.00 | 0.86 +/- 0.01 | 0.065 +/- 0.005 (0.025) | 0.99 +/- 0.01 |
| 0.02 | epsilon | 7.8 +/- 0.8 | 0.75 +/- 0.14 | 0.68 +/- 0.05 | 0.067 +/- 0.023 (0.007) | 0.54 +/- 0.10 |
| 0.02 | sigma | 8.0 +/- 1.9 | 0.05 +/- 0.03 | 0.50 +/- 0.10 | 0.319 +/- 0.085 (0.006) | 0.36 +/- 0.12 |
| 0.02 | topk9 | 9.0 +/- 0.0 | 1.00 +/- 0.00 | 0.97 +/- 0.01 | 0.029 +/- 0.005 (0.009) | 0.85 +/- 0.05 |
| 0.02 | topk25 | 25.0 +/- 0.0 | 0.99 +/- 0.02 | 0.80 +/- 0.04 | 0.117 +/- 0.018 (0.025) | 1.00 +/- 0.01 |

### Table 9 - Deep run, V up to 512 (4 seeds)
Same operating point; separate seeds from Table 3, so the V=128 column is an independent replication of it.

#### metaplasticity = 0.0

| arm | metric | V=128 | V=192 | V=256 | V=384 | V=512 |
|---|---|---|---|---|---|---|
| epsilon | assembly size | 7.6+/-1.4 | 5.7+/-1.1 | 4.6+/-0.9 | 3.4+/-0.6 | 2.8+/-0.4 |
| epsilon | identification | 0.38+/-0.26 | 0.05+/-0.05 | 0.01+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 |
| epsilon | recovery | 0.53+/-0.05 | 0.43+/-0.02 | 0.47+/-0.08 | 0.62+/-0.09 | 0.71+/-0.07 |
| sigma | assembly size | 6.0+/-1.2 | 4.4+/-0.9 | 3.6+/-0.7 | 2.7+/-0.5 | 2.3+/-0.4 |
| sigma | identification | 0.03+/-0.04 | 0.01+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 | 0.00+/-0.00 |
| sigma | recovery | 0.64+/-0.01 | 0.73+/-0.04 | 0.78+/-0.05 | 0.83+/-0.08 | 0.85+/-0.10 |
| topk9 | assembly size | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 |
| topk9 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 0.83+/-0.24 | 0.28+/-0.24 | 0.04+/-0.04 |
| topk9 | recovery | 0.91+/-0.02 | 0.85+/-0.04 | 0.74+/-0.10 | 0.55+/-0.04 | 0.52+/-0.08 |
| topk25 | assembly size | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 |
| topk25 | identification | 0.42+/-0.34 | 0.06+/-0.06 | 0.01+/-0.01 | 0.00+/-0.00 | 0.00+/-0.00 |
| topk25 | recovery | 0.58+/-0.07 | 0.53+/-0.05 | 0.63+/-0.05 | 0.75+/-0.03 | 0.82+/-0.03 |

#### metaplasticity = 5.0

| arm | metric | V=128 | V=192 | V=256 | V=384 | V=512 |
|---|---|---|---|---|---|---|
| epsilon | assembly size | 12.7+/-0.9 | 11.9+/-0.8 | 10.7+/-1.0 | 8.5+/-1.2 | 6.9+/-1.0 |
| epsilon | identification | 0.98+/-0.01 | 0.81+/-0.13 | 0.53+/-0.23 | 0.16+/-0.10 | 0.04+/-0.04 |
| epsilon | recovery | 0.76+/-0.01 | 0.65+/-0.06 | 0.53+/-0.09 | 0.47+/-0.02 | 0.49+/-0.03 |
| sigma | assembly size | 16.1+/-0.9 | 11.7+/-0.8 | 9.3+/-0.7 | 6.8+/-0.5 | 5.5+/-0.4 |
| sigma | identification | 0.25+/-0.06 | 0.10+/-0.03 | 0.04+/-0.01 | 0.01+/-0.01 | 0.01+/-0.00 |
| sigma | recovery | 0.44+/-0.02 | 0.50+/-0.01 | 0.55+/-0.03 | 0.63+/-0.08 | 0.68+/-0.08 |
| topk9 | assembly size | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 | 9.0+/-0.0 |
| topk9 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 |
| topk9 | recovery | 0.79+/-0.00 | 0.79+/-0.01 | 0.80+/-0.00 | 0.79+/-0.00 | 0.79+/-0.01 |
| topk25 | assembly size | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 | 25.0+/-0.0 |
| topk25 | identification | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 1.00+/-0.00 | 0.94+/-0.06 |
| topk25 | recovery | 0.70+/-0.00 | 0.70+/-0.00 | 0.70+/-0.00 | 0.68+/-0.01 | 0.63+/-0.01 |
