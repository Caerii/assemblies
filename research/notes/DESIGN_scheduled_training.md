# DESIGN: width, then the loop on the device -- layers 1 and 3

Measured (DESIGN_dense_cross_fiber.md): the dense aligner's cost per round is
FLAT in the number of brains -- 6.4 ms/round at B=5, 5.3 ms/round at B=50 --
so a round pays for launches and Python, not arithmetic (whose floor is
~1 us per brain-round). Two layers follow, each with a gate.

## Layer 1: per-brain schedules (width)

The batch dimension becomes "independent tasks with identical area shapes",
not "seeds of one corpus". Every brain carries its own schedule -- word index
and bundle index per step, padded with -1 -- its own vocabulary (word i of
brain b is brain b's word i; its phon fiber is seeded by (seed_b, i)) and
its own bundle inventory (feature indices per bundle, padded). All brains
advance one step per iteration; a brain past the end of its schedule has row
-1 and is skipped by the kernels.

Anchors stay constants: LEX winners per (brain, word) and FEAT's stimulus
drive and winners per (brain, bundle) are computed once, as tensors indexed
by the schedule. A step is exactly its cross rounds, each: gather stimulus
drive and LEX rows, one cross-drive kernel, jitter (per bundle, cached),
select, one write kernel.

    GATE   `ScheduledAligner` with every brain on the SAME corpus and step
           order equals `HashedAligner` -- the same kernels in the same order,
           so identical winners are expected, and the overlap tables must be
           identical.
    USE    word_capacity: a cell's V values x seeds in one launch (30 brains
           at k=50); cells differ in area shape and run separately.
    TARGET U1-size work at B=30 in the time B=5 takes today.

## Layer 3: the training loop on the device (persistent kernel)

With one launch per round, Python's per-step cost is the wall. The step loop
moves into a persistent kernel: one block per brain walks that brain's
schedule; per round it computes the FEAT drive over the columns (stimulus
drive + cross drive + jitter), selects k in shared memory, writes k x k cells
and updates the winner columns' max, relative mass and scale, and continues.
No host round-trip until the readout.

    GATE   the persistent kernel equals the layer-1 path on the same
           schedules: drives at every readout within 1e-5, and U1 = 1.000
           on every brain. Winner-level equality is EXPECTED (same
           arithmetic, deterministic ties under jitter) but the gate is on
           drives and outcomes, because summation order can move a float32
           ulp and ties are fragile.
    TARGET U1 (3,564 rounds) under 100 ms for five brains; the sweep in
           seconds.

Memory: `B x n_pre x n_post` int32 counts; 250 brains at 4000 x 1000 within
2 GB. Past that the store fiber returns for fibers that do not fit.

## Results (2026-09-03)

**Layer 1 gate:** overlap tables IDENTICAL (atol 0) to `HashedAligner` with
every brain on one corpus and order; a brain on a different schedule in the
same launch does not disturb another. The one discrepancy on the way was the
reconstruction readout's tie jitter (salted by the cross fiber alone); matched.

**Layer 3 gate:** count matrix, column maxima and tables IDENTICAL (atol 0)
to the layer-1 loop -- the bitonic selection uses the histogram selector's
own key, so ties break the same way, and the drive is summed in the same
order.

**Timing, U1-size schedule (3,564 rounds), same corpus for every brain:**

    width      python loop (layer 1)     device loop (layer 3)
    B=5        3066 ms   172 us/br-rnd    227 ms   12.7 us/br-rnd
    B=30       3546 ms    33 us           259 ms    2.4 us
    B=68       3401 ms    14 us           322 ms    1.33 us

The device loop's time is nearly flat in width up to the SM count (one block
per brain), and 1.33 us per brain-round is the arithmetic floor estimated in
DESIGN_dense_cross_fiber.md. Against this morning's 103 s for five brains on
the store fiber, U1 for five brains is 227 ms: ~450x; per brain-round from
~700 us to 1.3 us at width: ~500x. Against numpy (15-90 s per seed): three to
four orders of magnitude.

Targets: <= 0.05 ms per brain-round met by 40x at width; "U1 five brains
under 100 ms" missed by 2.3x -- a single block per brain serializes a brain's
rounds on one SM, so narrow launches are SM-bound. Width is the regime.

**The capacity sweep:** all five cells, thirty brains each, on the device
loop -- timing below. The science is unchanged from the hashed run: the
feature area binds first (PREREG_word_capacity.md Result), which is what a
faster instrument confirms rather than changes.

    The capacity sweep, five cells x 30 brains, device loop:   24 s total
    (numpy this morning: hours; hashed store: 7 min; layer 1: 3 min)
    V* identical to the layer-1 run, as gated.
