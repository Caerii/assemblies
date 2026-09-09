# Graded similarity is real, and the sampler destroys it by MERGING

Measured 2026-08-02. Scripts: `scratchpad/graded.py`, `graded_control.py`,
`graded_pool.py`. `numpy_sparse`, n=2000, k=100, p=0.1 (k·p=10, matching the
next-token studies), chance overlap = k/n = 0.0500.

This settles a question `ENGINE.md` had only *predicted*: it says keying fixes
the diagonal but not the off-diagonal, and that we should "expect the same for
pattern completion, association, and graded category structure." Expect. Here
it is measured, and the direction of the error is the opposite of what I
assumed.

## Finding 1 — the exact substrate has textbook graded similarity

Input patterns in SRC sharing a controlled fraction `f` of their neurons with a
reference pattern; project SRC→DST; read `overlap(DST(P_f), DST(P_1))`. Each
point on a **deepcopy** of one built brain, so no ordering effect can
manufacture a curve. 5 seeds.

| shared fraction f | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **explicit** | 0.0500 | 0.1080 | 0.1960 | 0.3300 | 0.5040 | 1.0000 |
| **materialized** | 0.0460 | 0.0940 | 0.1940 | 0.3180 | 0.5300 | 1.0000 |
| **sampled** | **0.9060** | 0.9000 | 0.9200 | 0.9360 | 0.9500 | 1.0000 |

The exact model starts at *exactly* chance for disjoint inputs and rises
monotonically to 1.0. `materialized` tracks it, which is the standing result
that bypassing the sampler reproduces the explicit engine.

**This is a positive result about the calculus** and it is the mechanism every
partial-overlap readout depends on. It was assumed here, not shown.

## Finding 2 — the sampler fails toward MERGING, not separation

I expected the sampled curve to sit flat *at chance* — correlation discarded, so
similar inputs no more alike than different ones. It sits flat at **0.906**:
two **fully disjoint** inputs produce nearly the same assembly. The sparse
engine's output is ~90% invariant to its input.

### The degenerate explanation, ruled out

A sealed area with only `k` materialized neurons must return all of them
whatever the input — overlap 1.0 for a trivial reason. That is not what this is:

| arm | DST neurons ever fired | distinct winners over the 6 patterns | overlap @ f=0 |
| --- | ---: | ---: | ---: |
| explicit | 100 | 311.8 | 0.0500 |
| materialized | 105 | 320.8 | 0.0460 |
| sampled | 105 | **119.8** | 0.9060 |

Identical pool sizes. The candidates **exist**; invented drives simply never
outbid already-materialized incumbents, so the area stays frozen on its
first-recruited population and re-uses 119.8 neurons where the exact engine
uses 311.8.

## Finding 3 — the error is a function of area LOAD, and it inverts

Overlap between two **fully disjoint** inputs, as the materialized population
grows. 4 seeds (8 at the two pinned points).

| distinct stimuli | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit pool | 100 | 189 | 341 | 572 | 838 | 1046 |
| **explicit overlap** | 0.0425 | 0.0600 | 0.0625 | 0.0675 | 0.0875 | **0.2625** |
| sampled pool | 105 | 203 | 392 | 701 | 1104 | 1438 |
| **sampled overlap** | **0.9025** | 0.8375 | 0.7175 | 0.4775 | 0.3450 | 0.1900 |

Pinned with 8 seeds and 95% CIs:

| | explicit | sampled |
| --- | ---: | ---: |
| 1 stimulus | 0.0475 ± 0.0177 (**0.9× chance**) | 0.9025 ± 0.0231 (**18.0× chance**) |
| 16 stimuli | 0.0925 ± 0.0208 (1.8× chance) | 0.3225 ± 0.0421 (6.5× chance) |

**The sampler's error decays with load and then INVERTS**: at 32 stimuli
sampled reads 0.1900 where explicit reads 0.2625, i.e. it now under-merges.
The crossover is near half the area materialized.

So the rule is not "the sampler is wrong by a factor". It is:

> **Distinctness measured on the sparse engine at LOW area load is not a
> measurement.** A lightly-loaded area reports a merge that the model does not
> have, by up to 18× chance. The error shrinks as the area fills and changes
> sign around 50% materialized.

This generalises the `pnas2020_scaling` trade already in `ENGINE.md`
(persistence 1.0000 / separation 0.0900 vs 0.0100 exact) from one protocol into
a load-dependent law.

## Finding 4 — the explicit arm independently reproduces α*

Ignore the sampler and read the explicit row alone: overlap is flat (0.9–1.8×
chance) through 16 stimuli and jumps to 5.25× at 32. `critical-load-alpha-star`
puts capacity at `M_max ≈ 1.15 n/k` = **23** at n=2000, k=100 — between 16 and
32, exactly where the jump is. An independent reproduction of α* from a
protocol built for something else.

## What this means for studies I–IV

* **Paired comparisons survive.** Both arms of any A/B run on the same engine at
  the same load, so the *difference* is real. Study IV's 4.3× distinctness gap
  between plastic and frozen CONTEXT stands.
* **Absolute overlaps do not.** Study II's collapse figure of 0.7566 is measured
  in the regime where the sampler inflates merging, so an unknown part of it is
  engine, not Hebbian dynamics.
* **Arm C is probably even more of a hash than measured.** Study IV read frozen
  CONTEXT at 0.1756 against chance 0.02 (8.8× chance). At comparable load the
  sampler inflates ~6.5× where exact reads 1.8×, so the true value is plausibly
  near chance. That *strengthens* the under-generalisation diagnosis: frozen
  recurrence is an even purer prefix hash than it appeared.

**Net: the sampler makes the substrate look MORE collapse-prone than it is.**
Every "shared area collapses" result in this repo was measured on an engine
that merges by up to 18× chance on its own. The collapse findings are not
thereby wrong — they have plasticity-controlled arms — but their magnitudes are
not the model's.

Task #85 (compute the drive instead of sampling it) is the fix, and this note
is the strongest case for it: it is not a fidelity nicety, it is the difference
between chance and 0.906 on the property every graded readout depends on.
