# DESIGN: the dense formulation to its bandwidth floor

Measured (DESIGN_scheduled_training.md): 1.33 us per brain-round at width 68
on the RTX 3080 (68 SMs, 760 GB/s). That is the arithmetic of the CURRENT
kernel at speed. This unit changes what the kernel stores and how it selects,
never the sum a column receives or the key that breaks a tie, so the gate is
the existing one: identical tables. Four changes, each against a named cost
in `sched_train_kernel`.

## Where a brain-round goes today

    drive    each thread: ~2 columns x 50 rows; a 10-op hash per (row, column)
             pair to decide PRESENCE, 50,000 hashes to find ~2,500 synapses
    select   bitonic sort of 1,024 64-bit keys: 55 stages, a block barrier
             each, to learn 50 winners
    write    50 threads of 512 walk 50 rows, hashing twice more per pair
    launch   one 512-thread block per brain; at width 68 one block per SM,
             a quarter of an SM's threads, idle at every barrier

Measured split: drive ~75%, select ~25%, write small.

## The four changes

1. **Presence bitmask.** The connectome is fixed by the seed. Store it once
   as bits, `pres[b, i, j/32]`, built by the SAME hash at fiber construction;
   a warp reads one word for 32 columns of a row. Removes the hash from
   drive and write. Identity is by construction: the mask is the hash's
   output.
2. **Radix select.** The k-th largest 64-bit key by 8-bit digits: a shared
   256-bin histogram per pass (warp-aggregated atomics), the crossing bin
   found by one warp, refine within it; stop early when a bin's count equals
   the remainder. Keys are unique (the column index is in the low bits) so
   exactly k keys are >= the threshold: the SAME winner set as the sort's
   first k. Winners are collected unordered; the write is per column and
   commutative, so order cannot change a table.
3. **Occupancy.** 256-thread blocks with ~10 KB shared: three to four blocks
   per SM. Width past 68 at nearly the same wall time.
4. **int16 counts.** A count is bounded by the number of rounds; the fiber
   refuses a schedule deeper than 32,767 rounds (`ensure_depth`). Halves the
   count traffic and, more importantly, halves memory: width 200 at the
   4000 x 1000 cell needs 1.6 GB of counts, not 3.2.

`dense_drive_kernel` / `dense_write_kernel` (the layer-1 path and
`HashedAligner`'s dense store) take the same mask and count type: one
representation, three kernels.

## The floor, measured not guessed

After the hash is gone a round's irreducible work is streaming 50 rows of
counts across N columns: ~100 KB per brain-round at int16, ~0.13 us at
760 GB/s. A probe kernel (`stream_probe`) does exactly that read and nothing
else, on the same layout with the same block shape, so the target below is a
RATIO to a measured ceiling, not a number ([[substrate-harness-owns-the-standards]]).

## Gates and bars (pre-registered)

    GATE-1  `test_device_loop_equals_python_loop`: counts, column maxima and
            overlap tables IDENTICAL (atol 0) to the python loop.
    GATE-2  `test_scheduled_equals_batched_on_one_corpus`: IDENTICAL to
            `HashedAligner` (dense store, same mask/counts).
    GATE-3  `test_dense_fiber_equals_store_fiber`: dense == store fiber
            within the existing tolerance (the store fiber is unchanged).
    GATE-4  presence mask == hash: the mask's popcount drive equals
            `hashed_drive` exactly on random row sets.
    BAR-T   <= 0.30 us per brain-round at width >= 200, n = k*20 = 1000 cell,
            U1-size schedule.
    BAR-R   kernel time <= 2x the stream probe's time at the same width.
    BAR-M   width 200 at the 4000 x 1000 cell fits (counts + mask < 4 GiB).

## Out of scope (the units after)

Incremental drive (recompute only columns whose state changed since the last
round; rows are fixed within a word and across a sentence's bundles) and the
present-only layout (CSR counts over the fixed connectome, warp per brain)
each move the floor itself; they are registered separately once this unit
has measured where the dense floor is.

Fewer rounds and coarser counts than the counts need are PROTOCOL changes:
they move the science and are not performance work.

## Results (2026-09-04)

**Gates:** GATE-1..4 pass (58 GPU tests: fused, scheduled, aligner parity,
substrate parity). End to end, the five-cell capacity sweep on the new
kernel wrote a results JSON byte-identical to the committed run.

**Bars:**

    BAR-T  0.449 us per brain-round at width 204 (0.552 at 272)   FAIL (<= 0.30)
    BAR-R  kernel / probe = 2.18 at 204, 2.30 at 272                FAIL (<= 2.0)
    BAR-M  width 200 at 4000 x 1000: 1.49 GiB counts + 0.10 mask     PASS

**Timing, U1-size schedule (3,564 rounds), same corpus for every brain:**

    width   before (int32, hash, bitonic, 512 thr)   after            probe (streamed lines)
    B=5     227 ms   12.7 us/br-rnd                  149 ms   8.4     53 ms   3.0
    B=68    322 ms    1.33                           204 ms   0.84    68 ms   0.28
    B=136      --                                    251 ms   0.52    90 ms   0.19
    B=204      --                                    326 ms   0.45   150 ms   0.21
    B=272      --                                    535 ms   0.55   233 ms   0.24

Against the layer-3 kernel this is 1.6x at width 68 and 3x at the width it
could not reach. Against the probe, 2.2x. The probe attains 417-541 GB/s of
the card's 760.

**Where a block-round went, by cycle counter (block 0, width 5), version by
version.** This table is the unit's real product: the model that named the
four changes was wrong about which one mattered, twice, by 5x.

    version                                   drive   select   write   total
    v1 mask as a global bit test               84k      17k     46k    151k   (old hash kernel: ~same)
    v2 mask staged in shared memory            90k      19k     41k    153k
    v3 chunked loads in flight, prices shared  44k      15k     31k     93k
    v4 block-wide write, prefix-skip select    37k       9k     21k     70k
    v4 at width 204 (3 blocks per SM)         105k      12k     43k    168k

**What was learned.**

1. *The hash was never the cost.* v1 equalled the old kernel. The drive was
   a chain of dependent loads issued one at a time: a warp waits whenever
   ANY of its 32 lanes has a present cell -- 81% of rows at p = 0.05 -- so
   "skip 95% of the loads" saved bandwidth we did not lack and none of the
   latency we did. The probe, which loads everything unconditionally, was
   7x faster on more bytes.
2. *Read the SASS.* `#pragma unroll` did not put loads in flight: nvcc sank
   each unconditional load INTO the presence branch, its only use, and
   serialized them again (`cuobjdump --dump-sass`: the 16-bit loads sat 66
   instructions apart, each behind its own use). Chunked loads into a
   register array, consumed through `acc += present ? v : 0`, gave every
   load an unconditional use; the SASS then showed them grouped in tens and
   the drive halved.
3. *Blocks phase-lock.* Identical schedules keep every block on an SM in
   the same phase: all stream their drive at once, then all select at once,
   so memory time and compute time ADD (probe 72 us + ~85 us at width 272).
   Self-timed de-phasing by q/nq of the first round recovered 12%, not the
   40% the model said -- the blocks re-synchronize under memory arbitration,
   or block index does not map to SM co-residency as assumed.
4. *Exactness under restructuring.* `x + 0.0f == x` for a sum of
   non-negative prices from +0, so the select form is bit-identical; the
   write's row-order double sum was kept by staging the new counts in shared
   memory and pricing sequentially per column; the prefix-skipping select's
   first version returned a wrong winner count because its last digit did
   not reach bit 0 -- float TIES are common (the jitter is below float32
   resolution at these drive magnitudes), so the column bits must always be
   covered.

**Where the remaining 2.2x is.** At width the drive phase alone (105k
cycles) lasts as long as the probe's whole round: bandwidth is shared and
the blocks are phase-locked, so the drive IS at the floor. The select and
write (30k of 70k cycles) then run with the memory system idle. The dense
formulation is within ~2x of its own floor; the next factor is structural --
the present-only layout (30x fewer bytes, the compute becomes the floor) --
or making the compute phases negligible (warp-per-column write, a
single-pass select).

**Sweep:** five cells, 30 brains each, byte-identical tables; 67 s wall
including start-up. The cells run at width 30 -- the narrow, latency-bound
regime -- so the sweep gains less than the width table: cells D (k = 100)
and E took 5 s and 3 s.

**Tooling.** Nsight Compute is installed but GPU performance counters need
the driver's admin permission (ERR_NVGPUCTRPERM); `clock64()` per phase from
block 0 into a global buffer, in a scratch build of the same source, was the
instrument that found every one of the four lessons.

## Profiler iteration (2026-09-04, counters enabled)

With performance counters enabled (NVIDIA Control Panel > Desktop > Enable
Developer Settings > Developer > Manage GPU Performance Counters > all
users), `ncu` on one launch, 150 steps.

**Stalls per issued instruction** (share of the total):

    width 5     long_scoreboard 1.76 (22%)  barrier 1.69 (21%)  wait 1.62 (21%)
                short_scoreboard 0.77 (10%) branch 0.49 (6%)   -- 7.85 total
    width 204   long_scoreboard 6.04 (43%)  barrier 2.42 (17%)  wait 1.65 (12%)
                short_scoreboard 1.02 (7%)                     -- 14.1 total

After the batching fix the narrow kernel is no longer memory-dominated:
barriers cost as much as loads (phases where one warp works and seven
wait). At width it is memory-latency-bound with DRAM at 60% of peak --
and occupancy capped at 3 blocks per SM by shared memory (27.6 KB per
block: keys 8 + presence 6.4 + prices 8 + staged counts 5, plus 4.6 static
and driver).

**Two changes from the profile, both gated identical:**

1. *Four blocks per SM.* The staged counts are live only in the write,
   after the keys are dead, so they alias the keys' space; the staged price
   head is capped at 1024 entries (4 KB) and lookups past the table's
   nonzero head skip the load (the price IS zero there). 22.6 -> 18.4 KB;
   ncu: Block Limit Shared Mem 4, achieved occupancy 65%, DRAM 65%.
   Wall time barely moved -- which said the bottleneck was not warps in
   flight.
2. *Sector traffic.* DRAM at the probe's fraction of peak while taking 2x
   the probe's time means 2x the probe's BYTES: a 2-byte count costs a
   32-byte sector, and the write's column-wise reads and writes of 2,500
   cells were ~160 KB of sectors per brain-round against the drive's 100.
   Presence is known from shared memory before any load, so the loads are
   PREDICATED per cell: straight-line, still ten in flight, but a
   predicated-off load fetches nothing and a warp-row fetches only sectors
   holding a present lane. SASS: 20 of 20 count loads predicated and still
   grouped. This also trims the drive below the probe's bytes (a sector of
   16 lanes has a present cell with probability 0.56).

**Timing, best of three** (the GPU is shared with a screen encoder,
RustDesk, whose load follows screen activity; the minimum is the
least-contended read; width 272 stayed contended and is not reported):

    width   before unit   v4      v5 (occupancy + predicated)   probe    ratio
    B=68    1.33          0.84    0.60                          0.33     1.8
    B=136     --          0.52    0.41                          0.23     1.8
    B=204     --          0.45    0.39                          0.23     1.7

    BAR-T  0.39 us at width 204                      FAIL (<= 0.30)
    BAR-R  1.66 at width 204                         PASS (<= 2.0)
    BAR-M                                            PASS

Against the layer-3 kernel: 2.2x at width 68, 3.4x at width 204. Full
suite 58 passed; sweep tables byte-identical.

**What the profiler adds to the lessons.** (5) Read the SECTORS, not the
bytes: at 2-byte cells the unit of traffic is 32 bytes, and a "scattered
write of 5% of cells" moves more than a coalesced read of all of them.
(6) Occupancy is only a lever when the stall is latency; at width the
kernel had become bandwidth-bound on inflated traffic, so a fourth block
per SM bought nothing until the bytes were cut. (7) Predication keeps loads
in flight where a branch serializes them; the compiler honours a per-cell
select on a load when the loaded value has an unconditional use.
