# The candidate sampler, measured against an engine that does not sample

The sparse engine never materializes the neurons that have not yet fired, so
when it has to decide whether a never-fired neuron outbids a settled incumbent
it *invents* that neuron's drive. Every version of this repo has done that with
some approximation. This note records what those approximations are worth,
measured against the explicit engine, which computes
`drive_i = |{j in x : synapse j -> i}|` exactly and samples nothing.

The arbiter matters because the two obvious sanity checks both lie:

* the recorded goldens were produced by *our own* old sampler, so they cannot
  tell a fix from a regression; and
* "the assembly is stable" is satisfied by degenerate states -- a sealed area
  scores 1.000 on every stability metric there is.

Everything below is at `norm_init=False` (the un-normalized PNAS 2020 model, so
the literature reproductions are comparable), `beta=0.05`, and scaled down
preserving the two invariants that govern the dynamics: the expected afferent
count `k*p = 3.17` and `k ~ sqrt(n)`.

## The three samplers

| | what a candidate's drive is |
| --- | --- |
| **old** | `k` fresh draws from the top-`k/n` tail of a truncated normal, redrawn every call |
| **keyed** | order statistics of ONE pool of `n` draws, keyed on the projection's content, offset by what that exact content already recruited |
| **keyed + cumulative** | the same, but the offset is discounted by the *correlation* between this input and the last one through the same fibers |

`keyed` exists because `old` is not idempotent: at `beta=0` with no recurrence,
drive is a fixed property of the random graph, so re-projecting the same
stimulus must elect the same winners. It did not -- overlap against the first
round ran 1.000, 0.620, 0.340, 0.320, 0.260 -- because two independent top-`k`
draws from a barely-moving tail each displace about half the other's winners.

## CORRECTION: the merge table below compared two different quantities

The explicit arm reports DISTINCT WINNERS over the run; the sparse arm reports
`area.w`. Those are not the same number -- `w` counts neurons materialized,
which runs ~20% ahead of neurons that ever won. Re-measured on distinct winners
for both, at the same sizes:

| n, k | explicit | sparse (keyed) |
| ---: | ---: | ---: |
| 1000, 32 | 6.1 | 8.9 |
| 2000, 45 | 6.1 | 8.2 |
| 4000, 63 | 6.9 | 10.8 |

So the real gap is about **1.4x**, not the 2-4x the table below implies. The
sampler is biased toward over-recruitment and the direction of every conclusion
here survives, but the magnitude was overstated by the metric mismatch. Left
in place, corrected rather than rewritten, because the mistake -- comparing an
arbiter to a subject on quantities that merely have similar names -- is the
same class as everything else in this note.

## Merge support (`merge_sim`, 50 rounds), in units of k

Support is the count of neurons that have ever fired: the quantity
`test_merge` asserts on, and the one that exposes over-recruitment.

| n, k | explicit (truth) | old | keyed | keyed + cumulative |
| ---: | ---: | ---: | ---: | ---: |
| 1000, 32 | **6.1** | 10.0 | 18.4 | 8.9 |
| 2000, 45 | **6.1** | 14.2 | 30.9 | -- |
| 4000, 63 | **6.9** | 12.2 | 25.9 | -- |

Swept across `p` at n=1000 (area A), explicit reads 5.7-7.2 and
`keyed + cumulative` reads 6.0-9.5; `keyed` alone read 6.8-15.8.

The truth is `~6.4k` and **flat in n**. `keyed` alone is not merely inaccurate,
it has the wrong shape: its support grows with `n`.

## Pattern completion (50% cue, 25 rounds), recovered fraction of k

| n, k | explicit (truth) | old | keyed | keyed + cumulative |
| ---: | ---: | ---: | ---: | ---: |
| 1000, 32 | **0.844** | 1.000 | 0.969 | -- |
| 2000, 45 | **0.778** | 1.000 | 0.978 | -- |
| 4000, 63 | **0.889** | 0.984 | 0.937 | -- |
| 10000, 100 | **0.670** | 1.000 | 0.950 | -- |

The old sampler recovers the assembly **perfectly** from a half cue, at three
of four sizes, which the exact model never does. `test_pattern_completion`
asserts `>= 300` of `k = 317` (0.946) -- a threshold calibrated against that
artifact.

## Why the correlation discount, and why it must not decay

`keyed` treats any input it has not seen byte-for-byte as brand new, and hands
it candidates from rank ~0: the extreme top of a pool of `n`. But in `merge_sim`
the source assemblies shift by a few neurons per round, so *every* round mints a
fresh key and draws a fresh maximum. Since drive is
`|{j in x : synapse j -> i}|`, two inputs overlapping 90% have drives correlated
0.9 -- the neurons already taken are still near the top, and only the genuinely
new 10% can reach past them.

So the offset is `rho * (what this fiber has ever recruited)`, with `rho` the
drive-weighted fraction of this projection's input shared with the previous one
through the same fibers.

The first version of this decayed the count, `eff <- rho * eff + recruited`,
which models the correlation between round `t` and round `t-d` as `rho^d`. That
is wrong whenever any part of the drive **persists**, and here some always does:
a stimulus fires the identical pattern every round, so a neuron wired to it is
favoured in all of them. Traced at n=1e5, k=317:

```
  t    rho     eff  offset      w   dw   winner stability
  2  0.333   418.7   139.6   1383  314            0.000
 12  0.334   470.8   157.4   4497  315            0.003
 24  0.344   439.3   151.1   8151  290            0.022
```

`rho` pinned at exactly `1/3 = stim/(stim + A + C)` for 25 consecutive rounds at
**zero** winner stability. The geometric sum capped the offset near 155 while
`w` ran past 8000, so candidates kept outbidding settled incumbents and the
assembly never converged at all. Summing without the decay lets the offset track
`w`; the transient then terminates and `w_A` at paper scale falls 8725 -> 4273.

Two accumulators are kept and the offset is the larger of them. The per-key
count is exact for an input repeated byte-for-byte **and survives interleaving
with other inputs**, which the per-fiber count does not; the per-fiber count
covers drift, which the per-key count cannot see. The per-fiber table is bounded
by the number of distinct projection shapes; the per-key table is now an LRU.

## `separate`: the offset buys idempotence and pays in cross-talk

`separate` drives two stimuli into ONE area and asks how much their assemblies
share. Same arbiter, at the regime the cross-language parity check fails on
(n=1e4, k=100, p=0.01) plus two k*p-preserving sizes, in units of chance:

| | persistence | separate (x chance) |
| --- | ---: | ---: |
| **explicit (truth)** | **0.746-0.860** | **0.00-1.98** |
| old | 1.0000 | 0.00-2.00 |
| keyed + cumulative | 1.0000 | **6.05-9.00** |
| keyed, offset hard-zeroed | 0.98-1.00 | 1.00-1.01 |

Two things fall out.

**`project_persistence = 1.0` is another artifact.** The exact model reads
0.75-0.86 at `beta=0.05`; both samplers read exactly 1.0000. Idempotence is a
`beta=0` property -- with plasticity on, re-projecting *should* move some
winners, and the goldens record a substrate that does not.

**The offset is what causes the cross-talk, and it is load-bearing for
idempotence.** Zeroing it at the sampler restores separate to 1.00x chance
exactly, and costs persistence (0.98, not 1.000). Note the first attempt at
this isolation was VOID: patching `rho` to 0 changed nothing, because the
offset is `max(per-key count, rho * fiber count)` and the per-key term still
carried it. The `max` has to be bypassed at the sampler to test the claim.

Mechanism: in the exact model, rounds 2-10 of s2 re-elect the same neurons and
recruitment simply stops. In the sampler, residual churn keeps minting fresh
keys, so the offset advances every round and s2's candidates sink deeper into
the tail -- until s1's incumbents, potentiated by (1+beta)^20 ~ 2.65, outrank
them. The overlap is s1 neurons winning slots in s2's assembly.

This is a genuine unresolved tension in the keyed approach, not a threshold to
retune: offset > 0 gives idempotence and cross-talk, offset = 0 gives clean
separation and churn. The principled resolution is for the offset to apply only
against the neurons THIS input recruited rather than against every incumbent,
which the current per-key/per-fiber counters cannot express.

## What is still open

At paper scale (n=1e5, k=317) merge support reads 13.5k, against 6-9.5k at the
sizes where the explicit engine can be run at all. Whether that is residual
n-dependence in the sampler or a genuine property of the protocol at that scale
is **not settled here** -- explicit at n=1e5 is 3 areas x 1e10 synapses, so the
arbiter does not reach it. `test_merge`'s `<= 4000` is a tier-C golden recorded
from the old sampler (which reads 3505); the flag reads 4273, 6.8% over. It has
NOT been retuned to pass -- a threshold moved to accommodate the code it is
meant to check stops being a check.

The `separate` cross-talk above is the blocking issue for making
`stable_candidates` the default. Idempotence and clean separation are both
correctness properties and the current offset cannot deliver both.

## The deeper core: dense wiring is a function of DECLARATION ORDER

Chasing the `separate` gap down produced two dead ends and one real answer.

**Dead end 1 -- the tie-break.** Drive is an integer count, so at low `k*p` the
top-k band is two or three distinct values and hundreds of neurons tie at the
cutoff. `heapq_select_top_k` has no tie-break policy, and
`all_inputs = concatenate([prev_winner_inputs, potential_new])` puts incumbents
at the low indices, so **on a tie an incumbent always beats a candidate**.
Censused over the regimes this repo runs, that decides 5.5-12.8% of the average
assembly and up to 92% of individual rounds, under BOTH samplers:

| regime | k*p | mean decided by index | worst round |
| --- | ---: | ---: | ---: |
| parity paper_canonical | 1.00 | 12.8% / 12.6% | 71% / 92% |
| parity ci_parity | 4.00 | 5.7% / 8.9% | 42% / 80% |
| engine_pricing | 5.00 | 5.5% / 7.1% | 39% / 70% |
| merge_sim (paper) | 3.17 | 8.8% / 9.5% | 53% / 34% |

(old sampler / keyed sampler.) This is real and worth fixing -- allocation order
is not a modelling choice anyone made. But it is NOT the cause: breaking ties
uniformly leaves persistence at exactly 1.0000 and moves separate only
9.00 -> 8.00.

**Dead end 2 -- the arbiter.** Before trusting explicit any further: at
`beta=0`, drive is a fixed function of the graph, so both engines MUST be
idempotent. Both read exactly 1.0000. The arbiter is sound.

**The real answer.** Explicit persistence measured 0.9048 in one harness and
0.7460 in another at *identical* parameters. The only difference was a second
stimulus that is declared and never projected. Minimal repro, same seed, same
sizes:

```
  s1->A identical : True
  A->A  identical : False   (23752 of 250000 synapses differ)
  A->A across two identical builds : True
```

`Connectome._initialize_weights` draws
`self._gen.binomial(1, p, size=(source, target))` -- one sequential draw per
connectome from one shared stream -- and `add_area` constructs stimulus fibers
before recurrent ones. So declaring an extra stimulus inserts a draw and
rewires **every connectome constructed after it**, 9.5% of A->A here.

This is the SAME defect class the repo already fixed once. Content-addressed
init keyed on `(row, col)` was applied to the sparse materialization path;
`Connectome._initialize_weights` -- the dense path, which is the whole explicit
engine -- still draws from a stream. The docstring on `rng` even records the
previous round of this fix (passing the Brain's generator instead of the global
one), which closed contamination BETWEEN brains but not order-dependence WITHIN
one.

Three consequences, in increasing order of seriousness:

1. Explicit persistence/separate numbers are one draw of a graph that depends
   on declaration order. The aggregate conclusions above survive -- they are
   statistics over several sizes and `p` values, and the merge invariant
   (`~6.4k`, flat in n) is stable across all of them -- but no single explicit
   number is reproducible across harnesses that declare differently.
2. Exact explicit-vs-sparse parity is unreachable by construction: the two
   engines are not wiring the same graph.
3. **An ablation that removes an EARLY-declared area rewires every area
   declared after it.** The confound is directional, and the direction matters:

   ```
   lesion an early area, read a later one : C->C differs, 15256/160000 (9.5%)
   lesion a late area,   read an earlier one : A->A identical
   ```

   So an ablation arm can differ from its control by more than the ablated
   component -- but only downstream of the removal. Studies that lesion by
   zeroing weights rather than by not declaring the area are unaffected.

The fix is to key dense init on content the way the sparse path already does,
which requires `Connectome` to receive the (source, target) names it currently
never sees. It is NOT landed here: it rewires every explicit-engine golden in
the repo and needs its own re-recording pass. Pinned as an xfail regression
test so it cannot be forgotten or silently "fixed" by a reordering.

## What the flag is actually worth

The same question, asked of the SPARSE engine -- the production default -- is
the strongest argument for `stable_candidates`, and it is not the one the flag
was built for. Does declaring a stimulus you never fire change the assembly?

| n, k | flag off | flag on |
| --- | ---: | ---: |
| 4000, 63 | 62/63 shared | **63/63** |
| 2000, 45 | 37/45 shared | **45/45** |
| 10000, 100 | 85/100 shared | **100/100** |

Off, up to 18% of the assembly moves because an unrelated declaration shifted
`self._sparse_sim.rng`. On, nothing moves. The keyed sampler was justified as an
idempotence fix; it is also the removal of the last stream-order dependence in
sparse winner selection.

Which puts the whole session in one line. This repo has been fixing ONE defect
class, one door at a time -- *randomness that stands in for a fixed structural
fact must be keyed on that fact, not drawn from a stream*:

| | site | status |
| --- | --- | --- |
| 1 | global `np.random` -> the Brain's generator | fixed |
| 2 | lazy sparse synapse init -> `stable_seed(row, col)` | fixed |
| 3 | CUDA hash -> fmix32 | fixed |
| 4 | candidate sampler | **fixed by this flag** |
| 5 | dense `Connectome._initialize_weights` | **still a stream** |

## The elegant resolution: compute the drive, do not sample it

Every defect in this note lives in one place -- `sample_new_winner_inputs`
invents a drive for neurons that have not fired. An invented drive cannot carry
the correlation the model gives for free, because the model says

    d_i(x)  = d_i(S) + d_i(U)
    d_i(x') = d_i(S) + d_i(U')

for inputs sharing `S`: the shared term is THE SAME RANDOM VARIABLE. Similar
inputs get similar drives automatically -- but only if a neuron's synapses are a
fixed fact rather than a fresh draw. Three offset schemes (per-key, discounted,
derived) were three attempts to reconstruct that correlation after having
thrown it away, and each traded one protocol against another because none of
them had the information.

The repo already makes synapses a fixed fact. `_init_area_block` is addressed
by ABSOLUTE position via `hash_area_weights(r0, r1, c0, c1, seed, p)`, and its
own docstring states the invariant: *materializing does not change the brain,
only when it exists.* So the exact drive of ANY neuron, materialized or not, is
computable -- `O(|active sources| x n)` time, `O(n)` memory, no weight storage.
Doing that removes, in one move:

  * the order-statistic sequence and all three offsets
  * the correlation gap (exact by construction)
  * the candidate-vs-incumbent scale mismatch
  * the incumbent-first tie asymmetry (all n neurons compete on one footing)

and makes the sparse engine agree with the explicit one by construction rather
than by calibration. The cost is time, not memory -- which is the right trade,
since memory (`O(n^2)` weights) is what makes the explicit engine unusable at
scale, not arithmetic.

### Demonstrated, not just argued

`materialize_area` is the existing route: at `w == n` there are no
unmaterialized neurons, so the sampler offers zero candidates and k-WTA runs on
exact drive. It was half-implemented -- area fibers wired at the correct
density, stimulus fibers left at zero, so a materialized area took the
zero-drive early return and elected an empty assembly every round. Fixed (it
passed every CONNECTED stimulus where the helper wanted only the FIRING ones,
which are deliberately zero-filled for the caller). With that closed, the merge
protocol, support in units of k (distinct winners over 50 rounds):

| n, k | explicit (truth) | **materialized sparse** | sampled sparse |
| ---: | ---: | ---: | ---: |
| 1000, 32 | 6.1 / 6.0 / 8.6 | **5.9 / 5.9 / 8.3** | 8.9 / 9.5 / 10.0 |
| 2000, 45 | 6.1 / 6.8 / 10.0 | **7.2 / 5.2 / 8.7** | 8.2 / 8.3 / 8.5 |
| 4000, 63 | 6.9 / 5.7 / 8.6 | **7.2 / 7.5 / 11.1** | 10.8 / 10.1 / 10.8 |

(areas A / B / C.) With the sampler bypassed the SPARSE engine reproduces the
EXPLICIT one, on all three areas, within scatter. The systematic
over-recruitment is entirely the candidate sampler -- not pricing, not the
offset, not the tie-break, not norm_init.

The internal check that the bypass is real: those materialized rows are
byte-identical with `NEURAL_ASSEMBLIES_STABLE_CANDIDATES` on and off. The flag
is a no-op when the sampler never runs, exactly as it must be.

So the sampler is a performance shortcut with a measurable accuracy cost, and
the engine already contains everything needed to charge that cost only where it
is wanted. The remaining work is to compute the drive lazily -- per active
source row via `hash_area_weights`, accumulated into a length-n vector and
discarded -- rather than materializing the weights, which is what keeps memory
at O(n) instead of O(n^2). See task #82's follow-on.

## The habit this is an instance of

Four separate things in this area were measurement defects rather than engine
defects: association read through a plastic probe that retrained what it
measured; association read a shared area never cleared between sources; the
order-statistic offset keyed on global `w`; and a test that encoded churn as
health. A fifth -- the pattern-completion threshold -- is a golden calibrated on
an artifact.

This substrate makes it very easy to build a measurement that reports the thing
it caused. `read_only()` probes, a `beta=0` null, checking the floor before
believing the ceiling, and -- when there is one -- an arbiter that does not
share the approximation under test.

---

# The offset cannot be right (2026-08-01)

The flag was left opt-in because it broke tests. Triaging those, 5 of the 7
reproduce in isolation (`wobbly_stress readiness_gates` and
`erp calibration_separates_category_violation` PASS when run alone, so they are
order- or cache-dependent, not flag-caused).

Of the 5, two share ONE root cause: the fiber-draw offset. A third does NOT,
and that negative result is the useful part -- see "Where the single-cause
story breaks" below.

## pnas2020_scaling, `paper_canonical` cell (n=10000, k=100, p=0.01, beta=0.05)

| arm | project_persistence | separate_overlap |
| --- | ---: | ---: |
| **materialized (samples nothing)** | **1.0000** | **0.0100** |
| flag off | 1.0000 | 0.0200 |
| flag on | 1.0000 | 0.0900 |
| flag on, offset forced to 0 | 0.9800 | 0.0100 |

`chance = k/n = 0.0100`.

Read the top row first. **The exact model gets both**: an assembly that fully
persists, and two separately-projected assemblies at exactly chance overlap --
which is the PNAS 2020 claim. Every sampled arm misses at least one.

* The offset is the ENTIRE separation failure. Keying alone lands on chance.
* But removing it costs persistence (1.0000 -> 0.9800).
* So the offset buys persistence and pays in separation, and there is no
  setting of it that buys both. This is the fourth offset policy to trade one
  protocol against another; see the list above. It is not a tuning problem.
* **The golden itself is an artifact.** `separate_overlap = 0.02` is 2x the
  exact value. Flag-off passes it by being wrong in the direction the golden
  was recorded in. So "make the flag match the golden" was never the right
  target.

## The coin, 40 flips (`test_fairness_cannot_tell_the_two_apart`)

| arm | legacy | attractor |
| --- | ---: | ---: |
| flag off | 0.500 | 0.650 |
| flag on | 0.450 | **0.825** |
| flag on, offset forced to 0 | 0.575 | 0.475 |

Same lever. Binomial SE at 40 flips is 0.079, so flag-on's attractor sits 4.1
SE from fair and the offset-free arm sits 0.3 SE from it. Consistent with
[[neural-coin-is-a-scaling-law]]: fairness emerges when basin asymmetry
self-averages, and a persistent offset is exactly what stops it averaging.

## Merge points the same way

Explicit ~2030 (in the units the test bounds at 4000); flag off 3505; flag on
4273. The offset was ADDED to cut merge over-recruitment and it does -- but
flag-on still lands further from the exact model than flag-off. The offset
helps the protocol it was fitted to and hurts the others.

## Where the single-cause story breaks: next-token prediction

`test_next_token_scaling` (n=10000, k=100, 50-word vocabulary), MRR against a
chance floor of 0.0900:

| arm | MRR | top1 | top3 |
| --- | ---: | ---: | ---: |
| **materialized (samples nothing)** | **0.1159** | 0.0000 | 0.0870 |
| flag off | 0.1165 | 0.0000 | 0.0870 |
| flag on | 0.0744 | 0.0000 | 0.0652 |
| flag on, offset forced to 0 | 0.0901 | 0.0000 | 0.0652 |

Two corrections to earlier readings of this test:

* `top1 = 0.0000` holds in the EXACT model too. It is a property of the task,
  not an engine artifact, and not evidence of a collapse.
* Zeroing the offset recovers only ~37% of the gap (0.0744 -> 0.0901 against
  an exact 0.1159). Unlike separation and the coin, the offset is NOT the
  whole story here.

### Why the same lever does not fix all three

Keying makes a repeated projection draw the same candidates. That fixes the
DIAGONAL of the drive covariance -- same input, same drive. It does nothing
for the OFF-DIAGONAL -- similar inputs, similar drives -- because that
correlation was thrown away at the draw and no key can reconstruct it.

    separate()   two disjoint stimuli, no shared component   diagonal only
    the coin     one construction flipped repeatedly          diagonal only
    next-token   contexts that share most of their words     off-diagonal

So the two protocols keying fixes are exactly the two that only need the
diagonal, and the one it does not fix is the one whose whole task is graded
similarity between overlapping inputs. This is the same `d_i(S)` term as at
the top of this note, seen from the other side.

It also predicts which future protocols will fail under the flag: anything
scored on partial overlap (pattern completion from a cue, association,
graded category structure), and not the ones scored on identity or
distinctness.

## Conclusion

Do not tune the offset again. Three protocols now agree that it is a band-aid
over information the sampler threw away at the draw: the correlation between
`d_i(x)` and `d_i(x')` for inputs sharing a component. A scalar per fiber
cannot carry it, which is why every policy so far has traded protocols.

The sampler-free arm demonstrates the information is recoverable, and
`hash_area_weights` makes it computable in O(n) memory. That is the fix.

**Decision: `NEURAL_ASSEMBLIES_STABLE_CANDIDATES` stays opt-in.** Its
order-independence benefit is real (up to 18% of an assembly, see
[[content-addressed-synapse-init]]), but it is not worth 5 protocol
regressions when the principled fix is known and scoped.
