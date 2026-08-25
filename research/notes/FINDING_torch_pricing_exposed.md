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

## Status

The three failures are LEFT FAILING on purpose. `test_engine_pricing.py`'s own
docstring says it "asserts the law directly instead of asserting a threshold a
degenerate state also passes"; loosening it to go green would discard the
signal it was written to produce.
