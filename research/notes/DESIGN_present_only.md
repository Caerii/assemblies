# DESIGN: the present-only fiber -- store what exists, one warp per brain

DESIGN_dense_floor.md ended at 0.39 us per brain-round, 1.7x a floor set by
STREAMING EVERY COUNT of 50 rows across 1,000 columns, 95% of which are not
synapses. That floor belongs to the dense layout. This unit changes the
layout and the work assignment; the arithmetic, its order, and the key that
breaks ties are unchanged, so the gate is the same: identical tables.

## Representation

The connectome is fixed by the seed. Per (brain, row): the present columns
with their counts, one packed 32-bit entry each (column in the low 16 bits,
int16 count in the high 16), padded to DMAX (the maximum row degree) with
-1. Built once from the presence bitmask (which remains the hash's stored
form and GATE-4's subject).

    per brain, 1000 x 1000, p = 0.05     dense int16 + mask    present-only
    memory                               2.1 MB                ~0.3 MB (DMAX ~80)
    bytes a round must read              ~56-100 KB            ~10 KB (K rows x DMAX x 4 B)
    bytes the write touches              ~8 KB                 ~4 KB

## Work assignment: a WARP per brain, rows in order

The dense kernel gave a brain a block and its lanes columns, because the
counts were column-contiguous. Present-only lists are row-contiguous, so a
brain gets a WARP and its lanes walk a row's entries. The warp walks the K
rows IN ORDER; a row's columns are distinct, so `drive[j] += price` is a
plain shared-memory add with no race, and every column's sum accumulates in
row order -- the SAME float sequence as `dense_drive_kernel`'s per-column
loop, hence bit-identical drives. The write's two passes (increment and new
max; price change at the new max, summed per winner column in row order)
walk rows the same way into per-slot shared accumulators. Selection is the
radix select of DESIGN_dense_floor.md at warp level: shared histogram,
warp shuffles, no block barrier anywhere in a round -- which was 17-21% of
stall cycles in the dense kernel.

Shared per warp ~10 KB (drive/keys 4 KB, staged column max 2 KB, winner
slot map 2 KB, histogram 1 KB, masks and accumulators); the staged price
head per block. 4-8 brains per block.

## Gates and bars (pre-registered)

    GATE-1  `test_device_loop_equals_python_loop`: counts, column maxima,
            tables IDENTICAL (atol 0) between the persistent kernel and the
            python loop on the present-only fiber.
    GATE-2  `test_scheduled_equals_batched_on_one_corpus`: IDENTICAL to
            `HashedAligner` (present store).
    GATE-3  present-only fiber == store fiber (`AreaFiber`) within the
            existing tolerance, the store fiber unchanged.
    GATE-4  presence mask == hash (unchanged), and the lists == the mask:
            every row's entries are exactly its set bits, ascending.
    GATE-5  the capacity sweep's results JSON byte-identical to the
            committed run.
    BAR-T2  <= 0.10 us per brain-round at width >= 200, n = 1000 cell.
    BAR-R2  kernel <= 2x a probe that streams K row lists per round in the
            same warp shape.
    BAR-M2  width 1000 at the 4000 x 1000 cell fits.

The dense kernels are deleted in a SEPARATE commit once the present-only
path is gated (dead-code deletion as a reviewable step).

## Out of scope

Incremental drive (recompute only touched columns) sits on top of this
layout and is the unit after; fewer rounds and coarser counts remain
protocol questions.

## Results (2026-09-04)

**Gates:** all pass. GATE-1 (persistent == python loop: counts, maxima,
tables at atol 0), GATE-2 (== `HashedAligner`, present store), GATE-3
(== store fiber), GATE-4b (lists == mask), plus a new exact gate: the
present fiber's drives, maxima and masses equal the DENSE fiber's at atol
0 on random distinct row/winner sets -- the row-order argument holds.
GATE-5: the capacity sweep's JSON byte-identical (three layouts, one
table). Note for the tests: winners must be DISTINCT -- the dense write
gives a column a block and counts a duplicate twice, the present write
masks it once; k-WTA never produces one.

**Bars, best of three, U1-size schedule (3,564 rounds), n = 1000:**

    width    present-only    row-list probe    ratio    dense kernel (DESIGN_dense_floor)
    B=68     1.32 us         0.141             9.3      0.60
    B=136    0.69            0.078             8.8      0.41
    B=272    0.37            0.041             9.0      0.51
    B=544    0.203           0.030 (600 GB/s)  6.7      not resident in one wave
    B=1088   0.204           0.030             6.8      --

    BAR-T2  0.20 us at width >= 544 (0.37 at 272)     FAIL (<= 0.10)
    BAR-R2  6.7                                        FAIL (<= 2.0)
    BAR-M2  width 1000 at 4000 x 1000: 1.40 GiB        PASS (7x less than dense)

Warps per block (2, 4, 8) at width 544: 0.205 / 0.203 / 0.206 -- the
per-SM warp count, not the block shape, is what binds.

**Reading.** The floor moved exactly as the layout promised: 0.23 -> 0.03
us per brain-round, the probe streaming row lists at 600 GB/s. The kernel
did not follow it: flat at 0.20 us from width 544 up, i.e. COMPUTE-BOUND
at 6.7x its floor, and 2x better than the dense kernel only because the
dense kernel could not be resident past 272 brains. A single warp's round
is ~87 us (width 5), against a ~15 us instruction estimate -- the fifth
time this session an estimate was 5x low, and the reason the profiler,
not the estimate, decides the next step (below).

**Profile at width 544 (ncu, one launch, 150 steps):**

    occupancy           16.7% theoretical = achieved: 2 blocks/SM x 4 warps
                        (dynamic shared 44.8 KB per block, 11 KB per warp);
                        107 registers per thread (block limit 4 by registers)
    issue               0.25 per scheduler-cycle; no eligible warp 75% of cycles
    stalls per issued   wait 2.43 (31%)  short_scoreboard 1.93 (24%)
                        long_scoreboard 1.00 (13%)  branch 0.89 (11%)
                        no_instruction 0.45
    DRAM                48% of peak; L1 26%; SM 25%

Read: the warp kernel is a LATENCY chain of shared-memory operations --
`drive[j] += price` is three dependent shared loads and a store per entry,
and a row must complete before the next -- run by too few warps to hide it.
Neither the memory system nor the issue slots are busy. The levers are
therefore (a) shared memory per warp, which sets warps per SM: drive 4 KB
and the staged column max 2 KB are essential, the slot map, histogram and
accumulators are not, ~6 KB is reachable and doubles the warps; (b) the
chain itself: fewer dependent shared accesses per entry (price index from
the staged max in one load; the row barrier replaced by a per-row-group
ordering only where columns collide). Per-phase cycle counts decide
which, next unit.
