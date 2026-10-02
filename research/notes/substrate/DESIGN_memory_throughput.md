# Memory throughput: the sweep is the batch

**Status:** built 2026-10-01; every path gated bit-for-bit on its solo run
(`neural_assemblies/tests/test_memory_throughput.py`). Measured speedups are
in the table below.

## The problem

The refracted-memory studies (PREREG_refraction_memory.md, Amendments 9 to
17) spend their time in `memory_learning_rate.run_beta`: store items one at a
time, read a sample of them back at every checkpoint, once per learning rate
in the sweep, one rate after another. At the studies' sizes (n = 2000 to
10000, 20 brains) a round reads a few MB. The work was LATENCY-bound, not
bandwidth-bound ([[gpu-latency-chains-not-bandwidth]],
[[gpu-lever-is-batching-not-the-kernel]]). (A first reading of 36% GPU
utilization during Amendment 17 is not evidence for this: the desktop's own
applications hold the card at 30-40% with nothing of ours running. The
evidence is the speedup from batching below, which a bandwidth-bound
workload would not show.)

* about 17 kernel launches and their Python per round, 8 rounds per item;
* three HOST SYNCS per item: the stimulus gain table copied from pageable
  host memory, the stimulus seeds copied the same way, and the k-WTA overflow
  flag read back -- each drained the CPU/GPU pipeline, so the CPU could
  never queue ahead;
* a checkpoint's 32 recalls run one after another, and a Python loop over
  brains per recalled cue (20 tiny launches each);
* the rates of a sweep, which share everything but beta, run serially.

## What changed

| lever | where | how it stays exact |
|---|---|---|
| **rates as brains** | `AssemblyMemory(beta=[...])`, `DenseOrganFiber` / `StimulusFiber` per-brain tables, `HashedArea` per-brain charge, `run_betas` | each brain reads its own chain/gain table row and its own 0.5 beta; brains never interact, so a brain in a sweep does its solo run's arithmetic |
| **finished rates leave** | `AssemblyMemory.select`, `run_betas` | a rate's stop decision is its solo decision; dropped brains' state is sliced, survivors' untouched |
| **batched recall** | `AssemblyMemory.recall_many`, the organ drive's brain map | a cue is a VIRTUAL brain that reads its brain's counts; frozen rounds write nothing, so cues cannot affect each other |
| **vectorized readings** | `readings` | the overlap of a recall with every stored item is one gather over a chunk of brains instead of a Python loop |
| **no host syncs in the write** | cached device gain tables, stimulus seeds sent a block of items at a time, overflow deferred to the checkpoint (`check_overflow`) | the overflow is still checked before anything reads a winner as a result; a truncated winner set can never be recorded |
| **the write as a CUDA graph** | `AssemblyMemory(graphs=True)`, kernels on torch's current stream | an ungated write is the same launches on the same shapes every item; everything it mutates is updated in place, so a replay is the eager write |
| **content-addressed kernel build** | `_fused_cuda.build_name()` | a pinned run worktree and an edited checkout no longer rebuild each other's extension (a rebuild cannot replace a module a running study has loaded on Windows) |

The memory budget of a launch is the organ fiber's `MAX_BYTES` (6 GiB of
int8 counts plus connectome bits): at n = 4000 a whole 9-rate sweep of 20
brains (180 brains, 3.2 GB) is one launch; at n = 8000 about four rates are;
`run_betas` splits the sweep by `launch_rates`.

## The gate

`test_memory_throughput.py` (CUDA): a swept memory equals its solo runs (items,
counts, bias, recall); `recall_many` equals `recall` per cue, masked and
unmasked, and in several passes; dropping rates leaves the survivors on their
solo trajectories; the graphed write equals the eager write through a drop;
the study loop's readings, first-item counts and stop points equal the
one-rate launch's, with rates stopping at different checkpoints; and a sweep
split across launches equals one launch. A recorded run is replayed on the
new path as the end-to-end check (below).

## Measured (RTX 3080, 2026-10-01)

One sweep at (4000, 60, 0.5), seeds 142-161, rates 0.0442-0.0884 (five),
1024 items each, no early stop:

| code | how | seconds | readings |
|---|---|---|---|
| a8b95efe (before) | one rate at a time | 57.3 | -- |
| this change | one rate at a time (`run_beta`) | 9.6 | identical, 9500 of 9500 scalars |
| this change | the five rates in one launch (`run_betas`) | 4.8 | identical, 9500 of 9500 scalars |
| + vectorized drive, short sort | the five rates in one launch | 2.9 | identical, 9500 of 9500 scalars |

12x, bit-identical to the code that ran Amendments 12-17. A nine-rate grid
(Amendment 17's) of 1024 items: 7.6 s, of which store 5.0 s and readings
1.5 s. The GPU is now busy rather than waiting on the host: by kernel time,
the organ drive is 44%, the k-WTA select 21% and the organ write 11%, and
those three kernels are the next levers.

**The kernels (second step).** The drive reads four adjacent columns per
thread with one char4 load per row (a quarter of the load instructions,
128-byte warp transactions); each column still sums its rows in row order,
so its float sequence is unchanged. The k-WTA select bitonic-sorts the
smallest power of two holding its candidates (typically 64-128) instead of
all 2048 slots: unique nonzero keys and zero padding put the same top k at
the top in the same order. Together: 4.8 s -> 2.9 s on the sweep (20x
over a8b95efe), the nine-rate grid 7.6 s -> 5.3 s (store 3.5 s, readings
0.9 s), readings still identical.

**Where this sits (Little's law, L = lambda W).** A store round reads
about 0.85 MB per brain at (4000, 60, 0.5) for about 0.16 MFLOP: 0.19
FLOP/byte, 200x below the 3080's ridge point, so only bandwidth and latency
matter. Before: L = 20 brains a launch, W = 1.4 ms a round (launches, three
host syncs, Python), lambda = 14k brain-rounds/s. After batching and graph
replay: L = 180, W = 0.61 ms, lambda = 295k/s. The bandwidth ceiling is
760 GB/s / 0.85 MB = 894k/s. Past L* = lambda_max x W_min (about 45 brains
at W_min ~ 50 us), more brains lengthen W instead of raising lambda: the
launch is now kernel-bound, not overhead-bound.

The count-saturation contract was tightened alongside
(`count_saturation_is_exact`, VERIFICATION.md#contract-organ-count-saturation):
a chain table longer than the count range is exact when its clip binds by
count 127. The memory's tables run to 256 rounds, and the old rule would
have raised on every Hebbian-control store whose hub counts pass 127, now
that `run_betas` checks the fiber at each checkpoint.

**The fused round (third step).** Two kernels replace two chains of torch
ops in the write, with the same float operations in the same order (the
_rn intrinsics stop the compiler contracting a multiply and an add into one
FMA): `stim_add`, the learned stimulus's priced drive (gather, multiply,
clamp, divide, add), and `charge`, the refraction bias and the ever-fired
record at the winners. 2.9 s -> 2.7 s on the sweep, identical readings.

## The lexicon learner

The word-capacity studies run the scheduled aligner, whose training kernel
gives a brain ONE WARP and walks its whole schedule in one launch. A launch
held one cell's 20 seeds at one vocabulary size: 20 warps, where the card
keeps about 200 resident (its ~30 KB of shared memory per warp allows three
per SM). A launch's time is its schedule length alone (36,864 steps, ~19 s
at V = 1024), and the rates, cells and vocabulary sizes ran one after
another: Amendment 5 measured 4.3 minutes per rate.

`word_capacity.run_cells_concurrent` runs every chunk of a study at once on
CUDA streams (one per worker thread; every kernel launches on torch's
current stream), the longest first, each chunk built and trained exactly as
the serial loop builds it. Two V = 1024 chunks: 27.2 s together against
20.4 s for one.

**The trap: the driver pages instead of failing.** The first concurrent run
admitted chunks against `_bytes_per_brain`'s estimate, which was half their
measured peak (1.58-2.23x), and the desktop holds ~2 GB of the card. On
Windows the driver backs a CUDA allocation that does not fit with system
memory rather than failing it (the sysmem fallback policy), and every chunk
ran ~60x slower (990-1285 s for a 20 s chunk). Admission is now against the
card's real free memory at the start, less a 1 GiB reserve, at PEAK_FACTOR
times the estimate, and the allocator is capped there so an overcommit
raises instead of paging.

**Smaller chunks.** The peak was `prepare` holding every per-word and
per-feature stimulus fiber at once ([B, V, n] + [B, F, feat_n] floats), the
anchor and cross jitters together, and one cached [B, n] jitter per word.
The anchors are now built one at a time and dropped, the jitter cache is
cleared per word (it is keyed by fiber identity, so a reused id must not
find the previous word's jitter), and each stage's inputs are freed once
consumed; the readout gathers each word's winners at the bundles' winners
instead of two dense [B, V, n] tensors. Measured peaks fell 2.0-2.6x (cell
C, V = 1024: 3.56 -> 1.39 GiB), so more chunks fit in flight.
