# PREREG: does word-learning CAPACITY obey the anchor law?

Registered before running. The anchor law now holds in two places: the
assembly-formation ceiling (`CAP-ANCHOR-RATIO`: anchor 2x -> M* 2.86x; p and
beta excursions undone by a computed anchor) and alignment ACCURACY under load
(`PREREG_alignment_load.md` L2: accuracy tracks the referent's drive share).
The question that decides how to SIZE a lexical area is whether the number of
word types one area can hold follows the same law as the number of
assemblies: a ratio in n/k, moved by the anchor.

## The learner, unchanged

`unaligned_scenes.Aligner` exactly as it passed U1-U3: feedforward LEX and
FEAT, one LEX -> FEAT fiber, column scaling on FEAT, reconstruction readout.
Nothing in the model changes between cells; only n, k, the phon stimulus size
s, and the vocabulary size V.

## The corpus, synthetic on purpose

V referents, each a bundle (IDENT_i, CAT_{i mod 4}) -- the lesion corpus's
shape, [DOG, ANIMAL]. Word w_i names referent i. A scene is 3 referents drawn
uniformly; its sentence is their three words. Each referent appears in about
E = 6 scenes. This is not a claim about a lexicon; it is the substrate's
capacity question asked with the mechanism that passed on real groundings.

## The ceiling

Per-type alignment against the whole inventory (chance 1/V), the stringent
metric. V* is the vocabulary at which the per-type accuracy curve crosses
0.90, interpolated in log2(V) between bracketing grid points -- the same
reading `ceiling_from_curve` gives M*; a curve that never crosses inside the
grid is CENSORED and reported as a bound, never a value.

Grid: V in (16, 32, 64, 128, 256, 512). Seeds (42, 1, 2, 3, 4). Cells:

    A  n=1000  k=50   s=k     n/k = 20
    B  n=2000  k=50   s=k     n/k = 40
    C  n=4000  k=50   s=k     n/k = 80
    D  n=4000  k=100  s=k     n/k = 40   <- the ratio-law cell, pairs with B
    E  n=2000  k=50   s=2k               <- the anchor cell, pairs with B

## Bars

    W1  DIRECTION.  V* rises with n/k at fixed k: V*(C) > V*(B) > V*(A),
        each step larger than the pooled seed CI of the two cells.
        FAIL: any inversion beyond seed noise.

    W2  RATIO LAW.  V*(D) within +/-25% of V*(B) -- same n/k, n and k
        both doubled. This is CS1 asked of word learning.
        FAIL: outside +/-40%.  Between: inconclusive.

    W3  ANCHOR.  V*(E) / V*(B) >= 1.3 (F2 gave 2.86 for assemblies; a weaker
        but same-sign effect is the bar, since alignment is a conjunction
        and the anchor enters through the LEX side only).
        FAIL: <= 1.1.

    Every cell must be UNCENSORED for its bar to be judged; a censored cell
    voids the bars it takes part in and is reported as a bound.

## What is NOT claimed

* Nothing about word ORDER or roles; V2-style induction is not re-run.
* V* here is an alignment ceiling with FEAT fixed at n=1000, k=50, so a
  FEAT-side limit could bind first at large V. If it does, the curve's shape
  will say so (accuracy falling while LEX is far from full) and W1 is void.
* Synthetic bundles have exactly one distinctive feature; real referents
  share more. The number is an upper bound on a real lexicon's V*.

## Interpretation, stated now

* W1, W2, W3 pass -> one law sizes both assemblies and words: neurons per
  word is a function of n/k moved by the anchor, and "large" becomes a
  computed quantity. CHILDES-scale sizing follows from it.
* W2 fails -> word capacity is NOT a ratio law; the lexical fiber has its own
  scaling and the anchor law's scope stops at assembly formation. Report the
  measured exponent as a fit, adopt nothing.
* W3 fails -> the anchor moves accuracy (L2) but not capacity; the two are
  different limits and the registration's premise was wrong.

---

## Amendment 1 (2026-09-03, before any bar was read): FEAT was following n; exposures 6 -> 12

The first build let FEAT scale with n although the registration fixes it at
1000 x 50. The sweep was stopped after cells A and B had been run at V <= 128
and no bar had been judged. What those partial curves showed is worth keeping:
with FEAT following n, cell B (n=2000) read LOWER than cell A (n=1000) at
V=16 (0.56-0.81 against 0.80-1.00), and an exposure ladder at V=16 gave

    FEAT follows n:   E=6   A 1.000  B 0.812  C 0.875
                      E=48  A 1.000  B 0.875  C 0.625
    FEAT fixed 1000:  E=6   A 1.000  B 1.000  C 1.000
                      E=48  A 1.000  B 1.000  C 1.000

so the degradation with n was a READOUT FLOOR -- more FEAT columns competing
in the reconstruction's k-WTA -- and not capacity. That is itself a finding
about the readout (reconstruction accuracy at fixed evidence falls with the
size of the area reconstructed into), recorded here and not judged. FEAT is
now fixed as registered. Exposures are raised from 6 to 12 so the curves
start above threshold and the ceiling is approached from above; at E=6 even
the fixed-FEAT build sat within noise of 0.90 at V=16. Bars unchanged.

## Amendment 2 (2026-09-03, pre-bar): the sweep runs on the hashed substrate, unclipped, two rounds per step

The numpy sweep was stopped by hand before any bar was judged (its partial
curves are in Amendment 1). The registered cells now run on `HashedAligner`
(`DESIGN_hashed_aligner.md`), whose gate is drive parity with the numpy
learner at < 5e-6 and U1 = 1.000 on all five brains. Three protocol facts
carried over from that gate, each measured:

* UNCLIPPED (`w_max=None`): column scaling and a clip do not commute; the
  numpy learner holds at w_max=None (0.985 / 1.000), so nothing rests on it.
* ANCHOR GAIN 1/p on the stimulus fibers, stimuli non-learning: the numpy
  learner's anchor by accident (0-or-size stimulus weights), made explicit.
* TWO cross rounds per (word, bundle) step instead of five: U1 reads 1.000 on
  every brain at 2, 3 and 5 rounds and 0.97-1.00 at 1; cost is linear in
  rounds. The registered bars and cells are unchanged.

One protocol difference from the numpy path is a nuisance, not a treatment:
the five seeds are five brains trained on ONE corpus (seeded 42) rather than
five corpora, because batched brains share a presentation sequence.

---

## Result (2026-09-03, hashed learner, dense fiber, ~7 minutes for all five cells)

    cell  n     k    s    n/k   V*                      censored
    A     1000  50   50   20    41.5 +/- 17.8 (28..65)   0/5
    B     2000  50   50   40    80.0 +/-  4.4 (76..84)   0/5
    C     4000  50   50   80    65.3 +/- 15.8 (49..79)   0/5
    D     4000  100  100  40    67.7 +/- 11.0 (53..75)   0/5
    E     2000  50   100  40    76.7 +/-  3.6 (72..79)   0/5

    W1  FAIL   A < B holds; B < C does not (65 vs 80)
    W2  PASS   V*(D)/V*(B) = 0.85 at equal n/k
    W3  FAIL   V*(E)/V*(B) = 0.96: doubling the phon anchor moves nothing

**Read with the registration's own caveat, the sweep answered a different
question than it asked.** "V* here is an alignment ceiling with FEAT fixed at
n=1000, k=50, so a FEAT-side limit could bind first at large V. If it does,
the curve's shape will say so and W1 is void." It does: V* saturates at
65-80 from n/k = 40 upward while LEX is nowhere near full, and the anchor --
which acts on LEX -- has no effect. The binding limit is the FEATURE area:
128 bundle assemblies of 50 in a 1000-neuron area overlap past what a
reconstruction readout can separate, and the shared category features make
the bundles overlap further. W1 is VOID as registered; W2's pass is then
uninformative (both cells sit on the same FEAT limit); W3's fail is what a
LEX-side lever does when LEX is not the bottleneck.

What IS established: the word ceiling of this learner is set by the area the
words are reconstructed INTO, not by the lexical area, at these sizes. The
registered question -- does the LEX-side ceiling follow the anchor law --
needs FEAT scaled with the cells (or held large enough not to bind), which
is Amendment 3 to register before running. On the dense fiber the whole
sweep took seven minutes, so that is an afternoon, not a week.

---

## Amendment 3 (2026-09-04, registered before running): the LEX-side ceiling, with FEAT measured not to bind

The Result above answered a FEAT-side question. The registered question is
LEX-side, and needs FEAT "scaled with the cells or held large enough not to
bind". Amendment 1 showed the other constraint: a LARGER feature area lowers
reconstruction accuracy at fixed evidence (a readout floor). "Large enough"
is therefore an empirical size, found first. Two parts, bars for each.

### Part 1 -- the FEAT ladder

Cells A (n/k = 20) and C (n/k = 80), LEX exactly as registered, FEAT on a
ladder held IDENTICAL across cells at each rung:

    FEAT  (n, k):  (1000, 50)  (2000, 50)  (4000, 50)  (8000, 50)  (4000, 100)  (8000, 100)

V grid (16 .. 1024), 10 seeds per (cell, rung), each seed its own corpus
(the scheduled learner gives every brain its own). V* by the registered
curve reading; censored rungs reported as bounds.

    F1  PLATEAU.  For cell C, V* is non-decreasing along the ladder at fixed
        k until a plateau: F* = the smallest rung whose V* lies within the
        pooled seed CI of the ladder's maximum. F* is then the "non-binding"
        FEAT for Part 2.
        FAIL (a): V* still rising at the last rung -- FEAT binds throughout;
        Part 2 is void and the ceiling is a FEAT law, reported as such.
        FAIL (b): V* FALLS along the ladder -- the readout floor dominates and
        this readout cannot ask the LEX question; reported, nothing adopted.

    F2  LEX VISIBLE.  V*(A) plateaus at a LOWER value than V*(C) (beyond the
        pooled CI): the LEX side is the binding limit once FEAT is not.
        FAIL: A and C plateau together -- the limit is still not LEX.

### Part 2 -- the registered cells at F*

All five cells at FEAT = F*, 20 seeds, V grid (16 .. 1024). Bars W1, W2, W3
EXACTLY as registered, judged on these curves; Part 1's rungs are not
reused as cells.

### What changes and what does not

The learner, the corpus, the exposure count (12), the two cross rounds and
the anchor gain are unchanged from Amendment 2. The substrate is the
present-only fiber (DESIGN_present_only.md), gated identical to the dense
and hashed paths and to the committed sweep tables. Seeds go from 5 to 20
for the judged cells because the instrument now affords it; the bars' CI
terms use the 20.

### Part 1 -- Result (2026-09-04, present-only substrate, 10 seeds per rung, V grid 16..1024)

    FEAT (n, k)     A (n/k = 20)                       C (n/k = 80)
    1000 x 50       51.9 +/- 11.5   (0/10 censored)    78.2 +/-  4.1
    2000 x 50       51.5 +/-  8.5                      166.4 +/- 11.2
    4000 x 50       31.3 +/- 10.9   (3/10 censored)    152.9 +/- 30.4
    8000 x 50       <= 16           (10/10 censored)   117.7 +/- 29.4
    4000 x 100      80.4 +/- 22.9                      327.2 +/- 41.1
    8000 x 100      32.3 +/- 16.9   (5/10 censored)    272.0 +/- 36.9

    F1  NO PLATEAU. At k = 50, V*(C) rises 78 -> 166 from FEAT 1000 to 2000
        and then FALLS (153, 118); at k = 100 it falls from 4000 to 8000.
        Neither of the registered FAIL branches alone: the ladder rises,
        peaks, and falls -- Amendment 1's readout floor (more feature
        columns competing in the reconstruction's k-WTA) takes over past a
        FEAT size that depends on k. Read as F1 FAIL (b) past the peak, with
        F* = the rung of maximal V*, (4000, 100), the same for both cells.
    F2  PASS at F*: A 80 +/- 23 against C 327 +/- 41, a gap of 247 against
        a pooled CI of 64; already at 2000 x 50, A 52 against C 166. The
        LEX side is the binding limit once FEAT is not.

Two things the ladder found that were not asked:

* The FEAT-side ceiling is NOT a ratio law. At equal feat_n / feat_k = 40,
  2000 x 50 gives V*(C) = 166 and 4000 x 100 gives 327: doubling both
  doubles the ceiling. What the readout needs is EVIDENCE -- k winners to
  separate assemblies whose overlap grows with V -- not a ratio; the same
  reading as Amendment 1's floor, now with its scaling.
* Cell A's V* is not fixed either: 52 at k = 50 rungs, 80 at 4000 x 100.
  The word ceiling is a CONJUNCTION of a LEX limit and a FEAT/readout
  limit, and F* only moves the second far enough for the first to show.

Part 2 therefore runs at F* = (4000, 100), as registered, with the caveat
stated now: FEAT still moves V* at F*, so cells whose LEX ceiling
approaches ~330 (the FEAT ceiling read on cell C) may be FEAT-bound there,
and a censored or converging top of the grid is reported as such.

### Part 2 -- Result (2026-09-04, FEAT = F* = 4000 x 100, 20 seeds per cell, V grid 16..1024)

    cell  n     k    s    n/k   V*                        censored
    A     1000  50   50   20    73.8 +/- 12.5 (16..135)    1/20 (one seed below the grid)
    B     2000  50   50   40    166.4 +/- 16.3 (99..231)   0/20
    C     4000  50   50   80    324.4 +/- 20.6 (239..383)  0/20
    D     4000  100  100  40    294.9 +/-  4.0 (276..310)  0/20
    E     2000  50   100  40    158.4 +/-  9.3 (118..184)  0/20

    W1  VOID by the letter (one censored seed of 20 in cell A). The
        direction is unambiguous for ANY value that seed could take:
        A 74 < B 166 < C 324, steps of 92 and 158 against pooled CIs of 29
        and 37. Reported, not judged.
    W2  FAIL. V*(D) / V*(B) = 1.77 at equal n/k = 40 (bar: within 25%;
        fail past 40%). Robust to the Part 1 caveat: D reads within a few
        percent of the FEAT ceiling with a seed CI of +/- 4, so it is
        plausibly FEAT-bound there -- which makes 295 a LOWER bound on its
        LEX ceiling, and the ratio can only be larger.
    W3  FAIL. V*(E) / V*(B) = 0.95: doubling the phon anchor moves nothing.
        B is far from the FEAT ceiling (166 against ~330), so the fail is
        informative: the anchor law does not govern word capacity.

**Reading.** Word capacity is NOT a ratio law and NOT an anchor law. At
fixed k = 50, V* is proportional to n: 74, 166, 324 for n = 1000, 2000,
4000 (a fit of n^1.07 over the 4x range); at fixed n = 4000, doubling k
leaves V* unchanged within the FEAT-ceiling caveat (295 at k = 100 against
324 at k = 50). One LEX area of n neurons holds about n / 12 word types at
this readout and corpus, whatever k in 50..100 and whatever the anchor.
This is the registration's own "W2 fails" branch: the lexical fiber has
its own scaling, the anchor law's scope stops at assembly formation
([[capacity-depends-on-n-over-k]] is about assemblies and stands), and
the fit is reported and nothing adopted.

**Why n and not n/k** (a hypothesis for the next registration, not a
claim): a word is a LEX assembly of k neurons projecting through one
fiber into FEAT; what limits separability at readout is how many
distinct k-subsets of n project to separable FEAT patterns, which grows
with the number of neurons available to differ in, n, while a larger k
adds evidence per word and interference per word in the same proportion.
The assembly-formation ceiling (M* in n/k) counts how many assemblies an
area can HOLD; the word ceiling counts how many an area can be READ from
through a fiber, and those are different quantities.

**Instrument.** Five cells, 140 brains each, in ~4 minutes of GPU time on
the present-only kernel; the same sweep was hours on numpy two days ago.

### Schema-8 migration replay (2026-09-11; software check, VOID)

The maintained entry point is now:

```bash
python -m research.experiments.word_capacity_run --tag UNIQUE
```

It requires a unique tag, records the complete version-3.3 protocol and the
scheduled aligner's two-family execution semantics, rejects incomplete or
changed fixed parameters before CUDA work, and writes an immutable run record.
The legacy `word_capacity.py` entry delegates to this same boundary.

The focused migration artifact at
[`word-capacity-cell-a-schema8-replay-20260911`](../../results/runs/aligner.word-capacity/word-capacity-cell-a-schema8-replay-20260911/results.json)
replayed cell A at FEAT = 4000 x 100 over the registered vocabulary grid and
the original twenty seeds. All 140 accuracy values are exactly equal to
`word_capacity_results_scheduled_feat4000x100.json`; the recomputed ceiling is
73.826 with one censored seed. This is a VOID software reproduction of one
registered cell, not new scientific evidence.

Protocol 3.3 replaces the parallel module constants behind that first schema-8
replay with one frozen `WordCapacityProtocol`. Its JSON form now also records the
corpus and training seed offsets, per-brain corpus scope, connectome-seed stride,
early-stop margin, interpolation band, bar thresholds, launch budget and FEAT
ladder. Corpus generation, scheduled alignment and the readout consume the
decoded value. Protocol 3.3 admits only `scheduled_aligner`; the lower hashed
path requires an explicitly different shared-batch corpus protocol. The
follow-up artifact at
[`word-capacity-protocol33-cell-a-replay-20260911`](../../results/runs/aligner.word-capacity/word-capacity-protocol33-cell-a-replay-20260911/results.json)
again matches all 140 original cell-A accuracy values exactly and recomputes the
same 73.826 ceiling with one censored seed. It too is a VOID migration check.
