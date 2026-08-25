# DESIGN: generate the connectome, don't fetch it — the GPU order of magnitude

Prototype and measurements. Not integrated; this note exists so the claim, the
evidence, and the parts that are *not* yet covered are written down together.

## The argument

A cell's weight is a pure function of its position:

    w[i,j] != 0   iff   float24( fmix32( i*A ^ j*B ^ seed ) ) < p

Nothing about the connectome has to exist in memory. `VirtualWeights` already
exploits this on the CPU and it is a **losing** trade there — hashing a cell
costs about what reading it costs, so it measures ~1.7x SLOWER than dense
(after this session's fixes; it was 2.88x).

On a GPU the trade inverts, for a reason that is arithmetic rather than
engineering. An RTX 3080 has ~8700 ALUs against ~760 GB/s: an
arithmetic-to-bandwidth ratio near 100:1. Generating a cell is ~10 integer ops;
fetching one is 4 bytes off the memory bus. **The hash is nearly free and the
gather is not.**

Two consequences, and the second is the one that matters:

1. the drive becomes COMPUTE-bound rather than MEMORY-bound;
2. **there is no matrix**, so the batch dimension is not bounded by memory —
   which is what lets one kernel launch amortize across many brains, and
   per-projection launch overhead is precisely what the sequential GPU engine
   loses to (measured 0.06-0.14x vs numpy on training).

## Why pure torch cannot express it

Written as torch elementwise ops, the hash is ~8 operations over a
`[B, k, tile]` tensor and **every one of them writes its result to DRAM**. The
gather is replaced by *more* traffic, not less. Measured: 0.7-0.9x against the
stored SpMM it was meant to beat, and 43.6 ms for B=64 at n=20000 — about 900M
integer ops, which the card should do in well under a millisecond. That gap is
round trips, not compute.

So the kernel is fused: one CUDA thread per (brain, column), the `k` hashes
accumulated in a **register**, nothing intermediate leaving the SM. Same
arithmetic, same result, no traffic.

## Measured

Prototype: `research/experiments/gpu_hashed_drive_prototype.py` (nvcc 13.1,
MSVC 14.44, RTX 3080).

**Correctness — against the ENGINE's kernel, not my transcription of it.**
Checked against `na_kernels.area_rows_*`, the rust kernel the numpy engine
actually uses:

    fused GPU == na_kernels rust drive : True   (max|diff| 0)
    nonzero columns 3898/4000, mean drive 3.507 (k*p = 3.5)

A GPU kernel that agrees with my reimplementation of the hash and disagrees
with the engine would be a fast wrong answer, which is the only outcome worse
than a slow right one.

**Fused vs the same arithmetic in torch:** 215-590x.

**Per brain per round, drive + top-k, against the CPU's stored CSR matvec:**

    n= 4,000  k= 60     0.943 ms CPU     0.00202 ms GPU      466x
    n=20,000  k= 70    24.134 ms CPU     0.00617 ms GPU     3912x
    n=50,000  k=100   171.597 ms CPU     0.16341 ms GPU     1050x

**Batch scaling and memory** (drive and selection split, so the next
bottleneck is visible):

    n       k     B     drive ms   topk ms   per brain   GPU MB
    20000   70    64      0.1494    0.2997     0.00702      5.5
    20000   70   256      0.8663    0.3652     0.00481     21.8
    20000   70  1024      2.7415    1.3707     0.00402     85.2
    50000  100  1024      9.5975    2.5287     0.01184    210.4

**1024 brains at n=20,000 fit in 85 MB** — one *stored* CSR at that size is
234 MB. The connectome is not there at all; the only memory is the output.

**Top-k is now a third to a half of the time.** The next move after this would
be a batched radix-select, not a better hash.

## What this does NOT cover

Stated plainly because the numbers above are seductive:

* **base drive and selection only.** No potentiation deviations, no stimulus
  term, no plasticity write-back, no refraction.
* the CPU baseline is a full `n x n` stored matvec. The engine's fiber is only
  `w x w` for materialized `w`, so early in training the engine does far less
  work than the baseline; the comparison is fair at organ scale (`w = 19999` of
  `n = 20000`) and generous to the GPU before that.
* the deviation store is the real open question. Per projection only the ACTIVE
  rows' deviations matter — the same sparse structure the k-WTA prune uses — so
  it should be a scatter-add of order 10^4 elements, but that is reasoning, not
  a measurement.

## The structural prize, which is larger than the speed

**With a generated connectome, lazy materialization stops being necessary.**
Growth exists to avoid storing `n x n`. If the matrix is never stored, every
neuron is addressable from the start, and "recruitment" is a change of which
indices are *active*, not a reallocation. That deletes:

* `_expand_connectomes` (13% of organ training) and its buffer doubling;
* `_ensure_area_block_coverage` and the deferred/eager init paths;
* the whole compact-index / neuron-id split, which has cost this project at
  least three retracted results ([[two-index-spaces-compact-vs-neuron-id]]).

That is a simplification of the engine's most defect-prone surface, arriving as
a side effect of the representation rather than as a refactor.

## Honest status

A prototype that computes one well-defined quantity very fast and correctly.
The path from here to a usable engine is the deviation store, the stimulus
term, and plasticity write-back — each of which could erode the factor. Nothing
here licenses a claim about end-to-end study throughput yet.
