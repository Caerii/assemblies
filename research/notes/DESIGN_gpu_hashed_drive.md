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

---

## Amendment 1: the deviation store, which was the open question

Prototype: `research/experiments/gpu_hashed_deviations_prototype.py`.

### It got the model wrong first, and the error is the interesting part

The first attempt derived a deviation cell's starting value from the hash. That
is not what a fiber holds. **Recruitment OVERRIDES cells** — it assigns 1.0 to
a cell whose base may have been ABSENT — so `present(i, j)` is the wrong test,
and the drive came out 23 units low on a mean of 26.7 with a different winner
set. A fast wrong answer, caught only because it was checked against the CPU's
`row_sum` rather than against a plausibility argument.

### The correct decomposition, and why it makes the correction nearly free

From `VirtualWeights.row_sum`:

    w[i,j] = chain( eff[i,j], count[i,j] )     eff = 1.0 if overridden,
                                                     else the raw base
    drive  = base_drive + SUM over deviation cells of ( chain(eff,c) - raw )

and the simplification that matters: **the base is Bernoulli 0/1**, so `eff`
and `raw` are each 0 or 1. Therefore

    chain(eff, c) = eff * tab[c]        tab[c] = chain(1.0, c)

with `tab` replayed on the host using the dense engine's own per-step
multiply-and-clip. A deviation cell is then `(col, count, eff_bit, raw_bit)`
and **the kernel needs no hash at all for the correction** — two bits and a
table lookup.

### Verified against the CPU, not against an argument

    M= 8   43,781 cells   max|GPU-CPU| = 3.8e-06   top-k set differs 0/8
    M=32  116,647 cells   max|GPU-CPU| = 7.6e-06   top-k set differs 0/8

That residual is float32 epsilon on drives of ~27 — the reduction order differs,
which the v2 drive semantics already licenses — and **the winner set is
identical on every assembly tested**, which is the property that matters.

### Cost

The deviation pass adds **1.05x to 4.2x** over base-only at real densities,
giving **~0.003-0.007 ms per brain per round at n=20,000, B=64-256**. Against
the CPU engine's ~1.5-5 ms per projection that is still **300-1000x**.

One number in the earlier table was an artifact and is called out rather than
quietly dropped: giving every one of 20,000 rows 208 deviations is a 4.3 GB
store at B=256 and measures thrashing, not the kernel. The real store is
44k-117k cells per brain.

### Still not covered

The stimulus term (a per-column vector add, cheap) and **plasticity
write-back**, which needs a sparse insert-if-absent on device and is the one
remaining piece that could genuinely erode the factor. Everything above is a
READ-side result.

---

## Amendment 2: plasticity write-back — no insert-if-absent needed

Prototype: `research/experiments/gpu_writeback_prototype.py`.

This was the last risk: after k-WTA, training increments `count[i,j]` for every
pair in `prev_winners x new_winners`, which reads like a sparse map update, and
**insert-if-absent is the one access pattern a GPU is bad at**.

### The structure that removes the problem

Three facts, and the third is the one that matters:

1. the update is a CROSS PRODUCT — `k x k` cells described by `2k` indices;
2. `count[i,j]` is not state. It is `#{ rounds t : i in W[t-1], j in W[t] }`,
   a pure function of the winner history;
3. therefore **a write needs no knowledge of what is already stored**.
   Appending the `k^2` keys is a complete record of the event.

So: append-only writes, and a periodic compaction that sorts the buffer with
the store and segment-reduces duplicates. Both halves are what a GPU is best at
— a radix sort and a scan — and neither needs an atomic or a probe. Keys are
packed globally as `b*n*n + i*n + j`, so ONE sort covers every brain at once and
the batch dimension costs nothing extra.

### Verified

    521 unique cells, dict reference has 521 -> identical: True

against a Python dict of counts built from the same events.

### Cost, at realistic winner reuse

Random winners make nearly every key unique — 20M distinct cells where a real
M=32 run holds 117k — so a random-winner benchmark is a pessimistic bound by two
orders of magnitude, not a measurement. Modelling stable cores with churn gives
a per-brain store of ~189k cells against the real 117k, close enough to trust:

    n=20000  k=50  B=64   store 12.1M cells (189k/brain)   0.039 ms/brain/round
    n=20000  k=70  B=64   store 23.7M cells                0.042 ms/brain/round

**B=256 does not fit**: the store plus sort workspace thrashes. Batch is capped
near 64 at this store size, which is still ample.

### An optimisation that measurement refused

The store is already sorted, so re-sorting it on every compaction looked like
obvious waste — sort only the buffers and merge once at the end. **Measured
17x SLOWER** (12.58 s against 0.73 s), because eager compaction DEDUPES EARLY
and keeps the working set small, while deferring accumulates 41M keys before
the first dedupe. The naive design was already the right one.

### What this does to the headline

Write-back at 0.039 ms/brain/round DOMINATES the read side (0.003-0.007), so
the honest end-to-end figure is **~0.045 ms per brain per round**, against the
CPU engine's 1.5-5 ms per projection:

    read side alone      300-1000x
    including write-back    35-110x

Still an order of magnitude and more, but the read-side number is not the one
to quote. The last risk is retired: there is no insert-if-absent anywhere in
the design.

---

## Amendment 3: the write-back is a GEMM — the sort was never necessary

Prototype: `research/experiments/gpu_writeback_gemm_prototype.py`.

Amendment 2 used the identity `count[i,j] = #{t : i in W[t-1], j in W[t]}` only
to justify APPEND-ONLY writes. That was the weak reading of it. Written as
algebra,

    count = SUM_t  x_{t-1} x_t^T                                          (*)

is a sum of T RANK-1 OUTER PRODUCTS, and two things follow that Amendment 2
missed.

### 1. The k^2 append is redundant by a factor of k/2, provably

An event carries `2k` numbers. Writing the `k x k` cross product writes the same
information `k/2` times over. That redundancy is created BEFORE the sort is
reached, so no sorting strategy can recover it — which is why the deferred-merge
experiment in Amendment 2 failed. It was optimising the wrong side of the pipe.

### 2. Restricted to the touched rows and columns, (*) IS A MATRIX PRODUCT

    count[i,j] = ( R^T C )[i,j],   R in {0,1}^{T x r},  C in {0,1}^{T x c}
    R[t,i] = 1[i in W_{t-1}],      C[t,j] = 1[j in W_t]

Both factors are THIN (T ~ 8, r,c ~ k). So the consolidation that the whole
append-and-sort budget went into is a batched GEMM of 0/1 matrices — no keys, no
radix sort, no segment reduce. Counts stay exact integers: 0/1 operands with
T <= 255, so fp32 accumulation is exact and the result fits in uint8.

### 3. There is no global store, because count is ADDITIVE

Blocks are never merged with each other. Two items whose blocks collide in a
cell are handled by the READER adding them, which is precisely what (*) says. So
the store is a list of per-(brain, item) blocks `(rows, cols, counts)` and there
is nothing to re-sort — which was the real cost centre: eager compaction
re-sorted 189k stored cells to absorb 20k new ones, touching every element
O(M) times.

### Verified

    462 nonzero cells, dict reference has 462 -> identical: True

### Measured, same winner streams, same process

     n    k    B    M   T |  sort s  ms/br/rd     MB |  gemm s  ms/br/rd    MB | speedup
 20000   50   64   32   8 |    0.45   0.02772  145.4 |   0.056   0.00341  17.1 |    8.1x
 20000   70   64   32   8 |    0.67   0.04118  284.0 |   0.056   0.00345  32.7 |   12.0x

**The speedup column is not the finding.** The GEMM path is FLAT (0.00341 ->
0.00345) while the baseline grows with the store (0.0277 -> 0.0412). Append+sort
is O(store); the GEMM is O(k^2) per item regardless of accumulated history. A
third row at M=64 would have shown exactly that divergence — the baseline
reached 9.7 GB and was killed rather than allowed to thrash, so it is
UNMEASURED and is not reported as a number.

Memory falls 8.5x: a block is 2k indices plus a uint8 matrix, against int64 keys
and int32 counts.

### The headline, revised

Write-back drops 0.039 -> 0.0034 ms/brain/round, which puts it BELOW the read
side (0.003-0.007). It is no longer the bottleneck:

    Amendment 2 end-to-end     35-110x
    with the GEMM write-back  145-780x

The read side is the binding constraint again, and within it top-k is a third to
a half of the time — so a batched radix-select is now the next lever, as the
original note predicted before write-back displaced it.

### Still not covered

The stimulus term (a per-column vector add). And the block store's READ path is
argued, not measured: a block is already `(row, col, count)` triples in dense
form, which is what the deviation kernel consumes, but no measurement here
substitutes for the CSR read of Amendment 1.

---

## Amendment 4: top-k -- selection is a histogram problem, 2-6x

Prototype: `research/experiments/gpu_radix_select_prototype.py`.

With write-back reduced to a GEMM, top-k became the binding constraint: 0.2997
ms against the drive's 0.1494 at n=20000, B=64. `torch.topk` is a general
comparison selector over arbitrary float32, but an assembly's drive is neither
arbitrary nor really a float -- the base is a Bernoulli sum, an INTEGER in
[0, k], and potentiation perturbs it slightly. Bounded, concentrated, and
MASSIVELY TIED. Selection over such data is a counting problem.

### The key, and why tie-breaking had to be designed first

For x >= 0 the IEEE754 bit pattern is monotone in x, so the comparison becomes
an integer one. But integer drives mean the k-th largest is routinely tied with
5-18 other columns (measured), and a selector that breaks those ties differently
from the engine returns a DIFFERENT ASSEMBLY -- the failure mode
`exact-tables-are-tie-fragile` records. So the index is folded into the key:

    key(j) = ( float_bits(x_j) << 16 ) | ( 65535 - j )

Keys are now UNIQUE, and "largest key" means "largest value, ties to the
smallest index" -- exactly stable argsort. **Tie-breaking stops being a policy
and becomes an identity.** Verified exact against
`np.argsort(-x, kind='stable')[:k]` -- the engine's semantics -- not against
torch.topk, whose tie order is unspecified.

### Three versions, and the profile that redirected the work

**v1**, one block per brain, four radix passes of 12 bits: exact, but 2.4x, and
**0.7x -- a LOSS -- at n=50000**. Not the algorithm: 64 blocks of 256 threads on
68 SMs is 17% occupancy, so five passes over L2-resident data cost what
seventeen should.

**v2**, four kernels, compacting after pass 1 (the top 12 bits are sign +
exponent + 3 mantissa bits, so ~56 buckets are occupied and one pass narrows
n=20000 to ~180 candidates -- **38-194x** across configs). Exact, never a loss,
but only 1.6-2.7x. The per-stage profile said why:

      n     B     hist   thresh  collect   refine      sum
  20000    64   0.0220   0.0286   0.0329   0.0408   0.1243

**No stage dominates**, and `thresh` -- which touches 4096 buckets per brain and
does nothing else -- costs as much as the stage that reads 5.1 MB. Launch and
latency, not work. A four-kernel design has a floor near 4 x 0.025 ms however
good its algorithm is. This is the third time this session a counter has
overturned an argument about where time goes; the bandwidth reasoning was being
applied inside a latency-bound regime.

**v3**, ONE launch, one block per brain, 1024 threads, candidates staged in
SHARED memory and refined by a block-resident bitonic sort. Two full-data passes,
67% occupancy, no DRAM round trip for the refine.

### It was fast and WRONG first

v3 measured 5.3x on the first run and failed exactness on every configuration.
The bitonic comparison direction was inverted -- it sorted descending while the
tail extraction assumed ascending. A fast wrong answer is the only outcome worse
than a slow right one, and it is caught here only because the exactness check
runs against stable argsort BEFORE the timing table.

### Measured, three runs, ranges not points

    n=20000 k= 70 B=  64    4.6 - 6.3x
    n=20000 k= 70 B= 256    1.5 - 3.6x
    n=20000 k= 70 B=1024    2.5 - 2.9x
    n=50000 k=100 B=  64    2.8 - 5.1x
    n=50000 k=100 B=1024    2.8 - 3.3x

**2-6x, typically ~3x** -- NOT the ~15x the two-pass bandwidth argument
predicted, and the run-to-run spread is wide enough that a single measurement
would have been misleading. The achieved bandwidth column says why: 170-560 GB/s
against ~760 peak, because at B=64 there are only 64 blocks to hide latency
with.

### Constraints, stated because they are real

* `n <= 65536` -- the key packs a 16-bit index. Larger n needs a wider key and
  a fifth pass.
* `K <= 1024`, the shared candidate buffer.
* Candidate overflow beyond 1024 slots is **detected and reported**, never
  silently truncated into a wrong assembly. Measured max was 269, a 3.8x margin,
  but the margin is data-dependent and the check is not optional.
* requires `x >= 0`, which holds for a sum of non-negative weights.

### Where this leaves the read side

At n=20000, B=64: drive 0.1494 + select ~0.056 against 0.1494 + 0.2997, so
~0.0032 ms per brain per round against 0.0070. Combined with the GEMM
write-back at 0.0034, **no single term dominates any more** -- drive, selection
and plasticity are within a factor of ~2 of each other. That is the natural
stopping point for this line of optimisation: further work should go into
integrating the design, not into shaving any one of the three.

---

## Amendment 5: integrated -- `batched_project_hashed`, and a hash defect found on the way

Code: `neural_assemblies/core/torch_engine/_fused_cuda.py`,
`_batched.batched_project_hashed`, `tests/test_fused_cuda.py`.
Benchmark: `research/experiments/gpu_hashed_batched_bench.py`.

### The integration point was already named in the codebase

`_batched.py` says: "Batching across INDEPENDENT connectomes (data-parallel
training of different brains) is Phase 3 -- it needs a block-diagonal CSR and is
scoped separately." **It does not need a CSR at all.**
`batched_project_independent` must materialise a `[B*n, B*n]` sparse matrix; at
B=64, n=20000, p=0.05 that is 1.28e9 edges (~15 GB), so the
independent-connectome case could not be run at organ scale. Regenerating the
connectome inside the drive kernel leaves the `[B, n]` drive as the only memory.

### A defect found before the integration could be built

The kernel has to agree with the engine's hash, so the first step was to check
it -- and `torch_engine/_hash.py` turned out to carry the docstring "Same hash
function as cuda_engine._hash_bernoulli_2d" while containing **no fmix32
finalizer**. Measured, reproducing the table in `kernels/implicit.py`:

    source                   density  row disp  col disp   corr(i)
    numpy Generator (ref)    0.04998     0.969     0.926  -0.00009
    torch _hash.py (before)  0.04999     3.794     0.015  -0.05262
    torch _hash.py (after)   0.04995     0.930     0.925   0.00047

Column dispersion 0.015 means in-degree was nearly CONSTANT, and `norm_init`
divides by in-degree -- so the raw hash quietly degenerated that mechanism while
every density check stayed green. Agreement with the numpy/rust hash went
0.905072 -> 1.000000. Fixed in `beb1372`; it does NOT close #98.

### Verified against the engine, not against a transcription

    hashed_drive == engine's stored W, summed over the row set : EXACT (torch.equal)
    multi-round batched == stored-connectome reference          : EXACT, 1 and 3 rounds
    selection == np.argsort(-x, kind='stable')[:k]              : EXACT

The reference deliberately uses the SAME tie policy as the kernel, so the test
isolates generated-vs-stored connectome from canonical-vs-unspecified tie order
rather than conflating them.

### Measured

    where BOTH can run
         n    k    B  rnds | blockdiag ms         nnz       MB | hashed ms    MB |  speed     mem
      2048   40    8     3 |        4.862   1,677,168    168.7 |     0.808   0.1 |   6.0x  1238x
      4096   60    8     3 |       18.521   6,708,907    671.9 |     0.801   0.3 |  23.1x  2495x
      4096   60   16     3 |   53217. (*)  13,422,693   1342.4 |     0.831   0.5 |    (*)  2500x

    (*) NOT a speedup figure. The block-diagonal path took 53 SECONDS here --
    it is thrashing under 1.3 GB of CSR plus SpMM workspace. Reporting that
    ratio as "64000x" would be quoting a memory collapse as arithmetic.

    organ scale -- only the hashed path exists
      n=20000 k= 70 B= 64:  1.12 ms,  10.3 MB | block-diagonal: 1.28e9 edges (~15 GB)
      n=20000 k= 70 B=256:  3.17 ms,  42.2 MB | block-diagonal: 5.12e9 edges (~61 GB)
      n=50000 k=100 B= 64:  3.18 ms,  25.7 MB | block-diagonal: 8.00e9 edges (~96 GB)

**The memory ratio is the result, not the speed ratio.** 1238-2500x less memory
is what converts "cannot run" into "runs in a millisecond".

### What is NOT integrated, stated plainly

* **Plasticity.** `batched_project_hashed` runs at beta=0. The GEMM write-back
  (Amendment 3) and the deviation-corrected drive (Amendment 1) are verified as
  prototypes but are not wired into this entry point, so this is the inference /
  parse case, not training.
* **The sequential engine is untouched.** `TorchSparseEngine.project_into`
  selects over a 1-D array for one brain; a batched selector cannot help it, and
  its k-WTA still goes through `torch.topk` / the CPU `WinnerSelector`.
* **Tie order.** The selector is canonical (stable argsort by construction);
  `torch.topk` and `heapq_select_top_k` are not. With 5-18 columns tied at the
  bar this changes WHICH neurons fire. `_kwta_prune` records that making the
  tie-break canonical is SCIENCE-AFFECTING and must not be smuggled in as an
  optimisation -- so this is a SEPARATE entry point, never a faster path inside
  an existing one, and nothing existing changed behaviour.
* **Limits.** `n <= 65536` (the key packs a 16-bit index) and `k <= 1024`.
  Candidate overflow raises rather than returning a truncated winner set.
* The kernels need nvcc, a host compiler and ninja; absent those, `load()`
  returns None, `available()` is False, and the tests skip.

---

## Amendment 6: plasticity -- the learned connectome is two bit-masks

Code: `_batched.batched_project_hashed(..., beta=, w_max=)`,
`_fused_cuda.dev_correct`. Benchmark:
`research/experiments/gpu_hashed_plasticity_bench.py`.

Amendment 5 shipped the inference case. This closes it for TRAINING, which is
what the blocked studies actually need.

### The identity, in its other form

Amendment 3 used `count = SUM_t x_{t-1} x_t^T` as a GEMM, which materialises the
block. For a drive correction the block is never wanted -- only single cells,
on demand -- and the same identity gives that directly:

    count[i,j] = popcount( rowmask[i] & colmask[j] )

with bit t of `rowmask[i]` recording "i fired at t-1" and bit t of
`colmask[j]` "j fired at t". One 64-bit AND and one `__popcll`. **The entire
learned connectome is two bit-masks**, `[B, n]` int64 plus a compacted column
list, and its SIZE DOES NOT GROW WITH THE ROUND COUNT. The GEMM form and the
popcount form are the same theorem answering different questions: materialise
the block, or evaluate a cell.

The correction itself is `(tab[count] - 1) * present(i,j)`, where `tab` is
`chain(1.0, c)` replayed on the host with the engine's per-step
multiply-and-clip -- NOT `min((1+beta)**c, w_max)`, which differs once the clip
binds. The base is Bernoulli 0/1, so an absent cell stays absent however often
it is potentiated, which is why the kernel needs one hash and a table lookup.

A small identity that avoids a missing primitive: torch has no bitwise
scatter-reduce, but a column appears at most ONCE per round, so the bits landing
on it are distinct powers of two and `scatter_add_` IS the OR.

### A fourth mirror divergence, measured not inferred

The engine potentiates `hebbian_update(src.winners, winners_long, ...)` --
source winners x target winners, i.e. `prev x new` for a recurrent fiber, since
`tgt.winners` is assigned AFTER. `batched_project_independent` builds ONE mask
from the NEW winners and applies it to both endpoints. Measured on a 64x64 case:

    batched_project_independent == prev_x_new : False (41 cells differ)
    batched_project_independent == new_x_new  : True  (0 cells differ)

`batched_project_hashed` follows the ENGINE. The two functions are therefore
NOT interchangeable and results from them are not comparable. Which rule is
right for `batched_project_independent` is a separate, science-affecting
question and is deliberately left open rather than silently resolved here --
this is the fourth instance of `pricing-law-implemented-twice` found in two
sessions.

### Verified

Exact winner-set equality against a stored-connectome reference implementing
`prev x new`, across rounds 2/4/6 and beta 0.10/0.25/0.50, with and without
`w_max`. The float-ordering hazard (GPU atomicAdd vs numpy summation, on drives
where ties at the bar are the common case) did NOT materialise at these sizes;
that is a measurement, not a guarantee, and is why the assertion is on the
winner SET. A separate test pins that beta actually moves the trajectory -- a
plasticity path that silently did nothing would pass every parity test against
a reference that also did nothing.

### Measured

         n    k     B    T | beta=0 ms  beta>0 ms      x | per br/rd ms     MB
     20000   70    64    8 |      3.33      10.81   3.2x |     0.02111   21.5
     20000   70    64   32 |      8.95      80.31   9.0x |     0.03921   26.5
     20000   70   256    8 |      7.34      15.45   2.1x |     0.00754   86.0
     50000  100    64    8 |      5.21      12.57   2.4x |     0.02454   51.9

Against the CPU engine's 1.5-5 ms per projection, TRAINING throughput is
**71-238x at B=64 and 200-660x at B=256**.

The T=32 row is the honest cost: the correction's work is `B*k*C` with C the
union of winner sets so far, so it GROWS with the round count even though the
stored state does not, and `_column_index` is rebuilt each round (O(T^2)
overall). An incremental column index would remove that and is not done here.

### Still not covered

Stimulus drive is accepted as a caller-supplied vector but is not hash-generated;
`norm_init` and `synaptic_scaling` are not applied in this path, so substrate
studies that need B or C cannot use it yet; and the round-mask caps learning at
64 rounds.
