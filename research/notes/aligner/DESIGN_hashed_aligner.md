# DESIGN: the aligner on the hashed substrate -- an engineering unit with a parity gate

## The cost model that motivates it (measured 2026-09-03)

`unaligned_scenes.Aligner` on `numpy_sparse`, same 13.8k projections at two
LEX sizes with FEAT fixed at 1000:

    n=1000   22 s     n=4000   31 s
    =>  T_proj ~ 1.4 ms (n-independent) + 0.2 us x n

The arithmetic in one projection is ~2e5 flops. The engine spends ~1.4 ms per
projection on work that does not scale with the problem -- dispatch, lookups,
recruitment and degree bookkeeping. That is Amdahl's serial term: at n=1000 it
is ~90% of the cost, at n=4000 ~65%, and no kernel speedup reaches it. A
capacity sweep is 110k projections per cell-seed x 25 cell-seeds: hours.

## The three levers, in order

1. **Fewer projections (algebra).** Both areas are stimulus-pinned, so the
   cross-fiber's Hebbian count over a corpus is ONE GEMM, `count = X^T C Y`
   ([[HEBB-OUTER-PRODUCT]]), column scaling is one per-column factor
   (`setpoint / colsum`, exact because scaling commutes with per-cell
   potentiation), and the readout is a second GEMM. Precondition: FEAT's
   winners under stimulus + cross-fiber equal its winners under stimulus
   alone. This unit MEASURES that precondition (`track_pinned`) rather than
   assuming it.
2. **Batch brains (Gustafson).** The hashed path runs B independent brains
   per launch at ~0.1 ms per brain-round. A study's seeds are B.
3. **Kernels.** Only after 1 and 2; the fused drive and radix select already
   give 30-50x on what remains.

## What is built

`core/torch_engine/_hashed_aligner.py`: `HashedAligner`, composed from
`HashedArea`, `StimulusFiber` (one per word into LEX, one per feature into
FEAT) and one `AreaFiber` LEX -> FEAT with column scaling. The protocol is the
numpy learner's, including the simultaneous update `Brain.project` gives its
batched targets. `overlap_table` scores every word against every bundle for
all brains with one mask GEMM.

## The gate (before any number from it is used)

`test_hashed_aligner_parity.py`:

* DRIVE REPLAY against `numpy_sparse` -- the engine's winner trajectory
  replayed through the hashed fibers, pre-k-WTA drive compared every round on
  both areas, stimulus bases injected from the engine, < 5e-6 relative. This
  is the arbiter every substrate arm was verified with.
* DECISIONS end to end on a tiny corpus: every word the numpy learner aligns,
  the hashed learner aligns identically on every brain.

Then `word_capacity.py --engine hashed` reproduces the numpy sweep's cells
before replacing it; the numpy run stays as the reference.

## What is measured, not assumed

* the pinned-winner fraction per training round -- how far the GEMM shortcut
  may be trusted;
* wall-clock per cell-seed, numpy vs hashed, so the speedup is a number.

## Found by the gate (2026-09-03): two tie facts, one of them a substrate defect

**The chain table is sized by potentiation count, not by episode.** The first
build gave the cross fiber `max_rounds=1` because it runs 1-round episodes;
every co-firing count above 1 then clamped silently to `tab[1]` and the
fiber's drive stopped growing after the first write. The drive-replay gate
caught it at the third cross round. The store now tracks its largest count and
`end_episode` raises when the table cannot price it; a regression test pins
three 1-round episodes against one 3-round episode.

**The numpy engine's stimulus into a materialized area is 0-or-size.** Its
weights take exactly two values. At n=1000, k=50, p=0.05 (the U1 operating
point) a fresh stimulus cue leaves 41 neurons strictly above the k-WTA bar and
959 tied at ZERO, so 9 of every 50 winners are zero-drive tie-fill chosen by
`argpartition`'s order. The hashed `StimulusFiber` generates the reference's
per-input-neuron Bernoulli counts instead (eight or so distinct levels), which
is the model as stated. Consequences:

* The two engines cannot agree at the WINNER level on a stimulus-driven
  area from identical bases -- LEX(dog) overlapped 1/20 between them -- while
  agreeing on the drive to 5e-6. Drive parity is the arbiter; winner-level
  comparisons on stimulus-driven areas are tie comparisons.
* The hashed selector's canonical smallest-index tie-break then picks the
  SAME low-index tie-fill for every word, correlating word assemblies; the
  aligner's toy alignment WORSENED with training (8/16 -> 5/16 over four
  brains). Fix: opt-in `HashedArea.tie_jitter`, a deterministic
  per-(input pattern, column) offset below any drive gap, on for the aligner
  and OFF by default so every existing capacity result keeps its canonical
  order.
* Every numpy learner result (U1-U3, load/Zipf) was measured with 18% of
  each assembly being tie-fill. The results stand -- the 41 carry the signal
  -- but the substrate they stand on is the odd one. Recorded, not fixed here:
  the engine's stimulus model is its own unit ([[add-stimulus-zero-or-size]]).

## Result of the gate (2026-09-03): exact, and the science holds

Four things had to be true before the port could be trusted, and each was
measured rather than assumed:

* **Drive replay < 5e-6** on both areas across the numpy learner's whole
  trajectory, unclipped, stimulus potentiation on, no anchor gain -- the
  arbiter.
* **Pricing is max-relative when scaling runs unclipped.** With column scaling
  a weight is `base (1+beta)^c s_j`, a share within its column, so only count
  DIFFERENCES matter; `(1+beta)^(c - cmax_j)` is that number bounded in (0, 1].
  A FEAT column in a frequent bundle co-fires >2,000 times in one pass over
  198 sentences, where the absolute chain is 10^80. Two kernels
  (`dev_correct_rel`, `column_mass_rel`), per-column `cmax`, tables sized by
  the corpus's total rounds. Relative equals absolute wherever both are
  representable (pinned by test).
* **Column scaling and a weight clip do not commute**, so `AreaFiber` now
  refuses the pair unless a caller opts into the shallow-count regime the
  capacity protocol lives in (T <= 8; its four-arm parity licenses it). The
  numpy learner's registration holds at `w_max=None` (U1 0.985 / 1.000), so
  the clip was never load-bearing.
* **Stimuli are anchors and do not learn; the anchor share is a parameter.**
  With stimulus potentiation on, a superordinate shared by two bundles
  swamped FEAT's k-WTA and the bundle assemblies merged (DOG stability 15/50).
  With the anchor at gain 1, the cross fiber decided FEAT's winners (pinned
  0.11) and alignment sat at 0.5-0.7. At gain 1/p -- the numpy learner's
  accidental anchor made explicit -- pinned reads 0.75 and

      U1 hashed, 5 brains:  scene-acc 1.000 1.000 1.000 1.000 1.000
                            type-acc  1.000 1.000 1.000 1.000 1.000
      U3 shuffled:          0.23 0.43 0.46 0.26 0.18   (chance 0.333)

  against the numpy learner's 0.994 +/- 0.017. The anchor law, in the
  language layer, on the exact substrate.

Wall-clock: 103 s for five brains at U1 size (8,910 cross rounds), against
15-90 s per seed on numpy -- 2-4x, not the target. The remaining cost is the
per-round `_rescale`, which walks the whole store to price winner columns, and
per-round launch overhead; measured next, not guessed.

The pinned fraction at gain 1/p is 0.75, not 1.0: the GEMM shortcut is NOT
exact on this protocol and is not adopted. It would be exact only where
FEAT's winners are fully pinned, which the anchor share does not guarantee.

## Rounds per step (2026-09-03)

    rounds_word   U1 scene-acc, 5 brains                 wall
    1             0.985 1.000 1.000 0.970 0.985           25 s
    2             1.000 1.000 1.000 1.000 1.000           31 s
    3             1.000 1.000 1.000 1.000 1.000           39 s
    5             1.000 1.000 1.000 1.000 1.000           73 s

Two rounds adopted for the capacity sweep (PREREG_word_capacity Amendment 2).

## Where the time goes now (5 brains, U1 size, 5 rounds, synchronized)

    before quick wins   101 s   4.73 ms / area-round
    _jitter cached, anchors constant   73 s   3.40 ms / area-round

    _rescale     20%   walks the whole store (2.3M entries) to price k columns
    _emit        15%   one fold per step: GEMM, local index, nonzero
    contribute   15%   scratch zero + count + relative apply per round
    append        8%
    stimuli       6%   one add per stimulus per round
    remainder    ~35%  topk_select, area glue, Python

Per brain-round ~0.7 ms against the 0.1 ms the capacity kernels reach: the
store-walking rescale and the per-round launch count are the next unit
(incremental column mass; fewer, fatter launches), registered separately.
