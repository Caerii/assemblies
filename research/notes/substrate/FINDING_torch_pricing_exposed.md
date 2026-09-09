# The corrected hash EXPOSED a torch pricing divergence

Not a regression from the hash fix -- a defect the biased hash was masking.
Recorded rather than fixed, because fixing it is its own unit of work and the
measurement should exist first.

## What happened

`beb1372` added the missing `fmix32` finalizer to `torch_engine/_hash.py`. That
is correct and stays: without it the torch engine disagreed with the
numpy/rust hash about 9.5% of cells, and column dispersion was 0.015 against
Binomial's ~0.93, i.e. IN-DEGREE WAS NEARLY CONSTANT.

`norm_init` divides each column's drive by its in-degree. When in-degree is
nearly constant that divisor is nearly a constant, so a mispriced divisor is
INVISIBLE -- every column is scaled the same way. Restoring the real Binomial
spread made the divisor vary, and the torch engine's handling of it stopped
being hidden.

## Measured

`test_engine_pricing.py::test_area_neither_seals_nor_exhausts`, K = 100.
`w` = neurons ever fired after converging on one input; `w_new` = after
switching to a different input.

    engine         n_src  n_tgt |  pre-fix w / w_new |  post-fix w / w_new
    numpy_sparse    1000  10000 |       154 /   345  |       154 /   345
    numpy_sparse    5000   5000 |       117 /   229  |       117 /   229
    numpy_sparse   10000   1000 |       111 /   218  |       111 /   218
    torch_sparse    1000  10000 |       478 /   510  |       520 /   492
    torch_sparse    5000   5000 |       156 /   184  |       249 /   199
    torch_sparse   10000   1000 |       135 /   144  |       243 /   239

Read the numpy column first: it is the reference and the hash fix does not
touch it (numpy already finalized). **The torch engine over-recruits against
numpy by 1.4-3.4x BEFORE the fix and 2.2-3.4x after.** The divergence is
pre-existing; the fix widened it and broke the test's `w_new > w` proxy.

## What is NOT claimed

* NOT "the area seals". Sealing means `w == k == 100`; torch sits at 243-520.
  The three failing assertions are the `w_new > w` half, which held only
  marginally before (510 > 478, 184 > 156, 144 > 135).
* NOT that the hash fix caused the divergence. It existed at every size ratio
  beforehand.

## A candidate cause, unverified

`_engine._norm_scale_area` passes `self.p` -- the BRAIN's global p -- to
`inverse_indegree`, where the numpy engine passes the FIBER's p via
`self._p_for(src, tgt)`. That is the fiber-p fix (79fba4f) never reaching the
mirror, the same shape as the finalizer itself. It cannot explain THESE numbers
(the test brain is homogeneous, so the two p's coincide), but it is a live
defect on any heterogeneous brain and should be fixed with whatever else is
found here.

## Resolved: it was a DEAD FIBER, not pricing

The three failures were a real defect, and the test's own diagnosis ("sealed --
candidates over-divided") was the wrong mechanism. Measured WITHIN one brain,
which is what the docstring actually describes:

    engine         n_src  n_tgt |  w@conv  +5 same   stab | after switch  delta
    numpy_sparse   10000   1000 |     109      109  1.000 |         216   +107
    torch_sparse   10000   1000 |     239      274  0.930 |         274     +0

numpy converges (stability 1.000) then recruits +107 for the novel input; torch
never converges and recruits EXACTLY +0. Those two do not both follow from a
mispriced divisor -- churn means candidates win too easily, +0 means they
cannot win at all. Inspecting the fiber settled it:

    numpy_sparse  SRC2->TGT: ndarray shape=(514, 218) nnz=3837
    torch_sparse  SRC2->TGT: CSRConn nrows=0 ncols=0 nnz=0

`if csr.nnz == 0: continue` DEADLOCKS: an unmaterialised fiber delivers zero
drive, so nothing is recruited, so `_expand_connectomes` never runs, so the
fiber stays empty forever. The "Zero signal -- preserve current assembly"
branch then freezes the winners, and the whole thing presents as a sealed area
because a zero-drive projection still returns k winners
([[silent-no-op-dead-fibers]]). Same defect the fixed-assembly branch in the
same file already fixes -- that branch got it, the ordinary drive path did not.

### Two wrong fixes before the right one

Seeding empty fibers in the DRIVE LOOP made all 23 pricing tests pass and broke
six others: it duplicates `_expand_connectomes`, which owns growth once the
fiber exists, and areas stopped converging. Scoping to `nnz == 0` was not
enough either.

The discriminator is WHERE, not WHEN. Seed at the zero-signal branch, and only
for CROSS-AREA fibers:

* a SELF-fiber that is silent means the area has nothing to say to itself yet;
  it still has the stimulus driving recruitment, so it never reaches that
  branch, and seeding it mid-run replaces the assembly with a fresh random
  draw -- measured at stability 0.010, which IS chance (k/n = 100/10000);
* a CROSS-AREA fiber is the one nothing else will ever build.

Full non-slow suite after the fix: **1610 passed, 0 failed** (was 3 failed,
1606 passed).

### Still open

The fix does NOT close #98. Reciprocal recovery is unchanged at numpy 0.2320
vs torch 0.6500, so that divergence is a third, separate thing. The candidate
cause named above -- `_norm_scale_area` passing the brain's `self.p` where
numpy passes the FIBER's p -- remains unfixed and unverified; it cannot explain
a homogeneous-brain test but is live on any heterogeneous brain.
