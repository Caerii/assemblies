# PREREG: drive semantics v2 — the decomposition becomes the definition

Registered before implementing. Follows `5732e15` (vectorized VirtualWeights,
byte-identical to dense, eval 1.15-1.45x dense).

## What changes and why

The drive of a projection is currently DEFINED by the dense implementation:
float32 pairwise `block[rows].sum(axis=0)`. That definition forbids the one
algorithmic win the virtual representation makes possible: the fiber's base
is IMMUTABLE (a pure hash), so

    drive(rows) = base_sum(rows)  +  delta(rows)

where `base_sum` is a pure function of the row-set — cacheable FOREVER, 33 KB
per row-set against 5.6 MB per block — and `delta` is a sparse sum over the
deviation store. Training presents the same row-sets ~15x and eval revisits
the same states hundreds of times, so the amortized drive cost drops from
O(k * n_cols) to O(deviations) + top-k. Dense storage structurally cannot do
this: plasticity mutates its matrix in place, so there is no immutable object
to memoize.

The catch: `base_sum + delta` is a different summation order than the dense
reduction, and an ulp flips k-WTA tie ORDER (measured, this session). So the
decomposition cannot be an optimization UNDER dense semantics; it must BE the
semantics.

## The new definition, fixed exactly

For a virtual fiber, `row_sum(rows, col_end)` is:

1. `base_sum` = float64 pairwise `raw_block[rows].sum(axis=0)` over the RAW
   hash base (no overrides), full fiber width, computed once per row-set and
   memoized keyed on the row-set bytes; sliced to `col_end` on read.
2. `delta` = float64 accumulation, via one unbuffered `np.add.at` over the
   concatenated deviation triples of the selected rows IN ROW ORDER, of
   `final(cell) - raw(cell)`, where `final = chain(override ? 1.0 : raw, n)`
   with the float32 event chain unchanged.
3. The result is `float32(base_sum[:cols] + delta)`.

Deterministic: every term is a pure function of (row-set, store state), and
the accumulation order is fixed. Memoized and cold paths are the SAME
computation, so self-consistency is exact by construction, not by luck.

Per-cell semantics (`cell`, `todense`, `column_nnz`) are UNCHANGED — the
event chain, the override clobber, the sparsity pattern all stay as pinned by
`test_virtual_weights`. Only the summation of a row-set moves, from f32
pairwise to f64 decomposition — a numerical STRENGTHENING at the only place
the two orders differ.

The DENSE path does not change at all. Its fingerprint golden is untouched.

## What guards it

* **Self-consistency**: cold-path and memoized `row_sum` bit-identical; two
  full runs bit-identical (extends the integration test).
* **A virtual-semantics fingerprint**: the five configurations captured
  gate-on into their OWN golden file, so v2 semantics is pinned against
  future drift exactly as dense is.
* **The dense-equality assertion is retired**, with the reason in the test:
  virtual winners may differ from dense by ulp-tie reordering, which is the
  cost this note exists to justify.

## Science invariance — the bar that legitimizes the change

Ulp-level tie reordering must not change FINDINGS. Registered check, run
before the semantics lands on any study path:

* **V-S5:** the registered S5 word-problem study's trained arms (4 groups x
  10 seeds, same builds, same words) re-run under gate-on v2 semantics.
  BAR: the four registered verdicts (S1 9/10, S2 FAIL, S3 PASS, S4 clean
  controls' spot check) are UNCHANGED. Per-seed exact@L flags are reported
  and MAY differ in isolated cells — ties are real — but any verdict flip
  stops the change.
  *Prediction: tables nearly identical, verdicts identical (~90%).*

## Committed in advance

1. Bars before implementation; no tuning after seeing results.
2. If V-S5 flips any verdict, the semantics change does NOT land on default
   paths; it stays a documented fast mode and the flip is investigated as a
   finding about tie sensitivity, not suppressed.
3. Speed is measured AFTER correctness closes, on the same cells as 5732e15,
   and reported whatever it says.

---

## Result: V-S5 FAILED, and the stop clause governs

    A4xZ5  L=50: 9->10   L=500: 9->7
    A5     L=10: 9->8    L=50: 9->8    L=100: 9->7    L=500: 7->6
    S5     L=50: 9->10   L=100: 8->10  L=500: 5->7
    16 cells moved.  S1 FLIPPED (A5 exact@100 9/10 -> 7/10).  S2, S3 unchanged.

Per commitment 2, drive semantics v2 does NOT land on any default or
registered-study path. It remains behind ASSEMBLIES_VIRTUAL_WEIGHTS as a
documented fast mode for exploratory work, pinned by its own fingerprint
golden and self-consistency tests.

**The finding the flip constitutes.** The movement has NO systematic
direction -- A5 lost 2 seeds at L=100 while S5 GAINED 2 -- which is what
tie-level arbitrariness looks like, not what degradation looks like. The
deeper point is about the INSTRUMENT: per-seed exact-trajectory tables carry
+/-1-2 cells of variance attributable to k-WTA ties whose resolution depends
on float summation order, i.e. on nothing physical. S1's registered PASS sat
exactly at its 9/10 bar, one arbitrary tie from either 10/10 or 7/10. This
sharpens the session's earlier conclusion twice over: exact@L was already
shown underpowered (Amendment 2) and cliff-shaped (hitting times); it is now
also shown TIE-FRAGILE. Robust sequence claims should rest on the
soft-transition census and hitting-time statistics, which are per-substrate
mechanical facts, not on exact-table cells.

**Speed, reported as committed** (noisy box, same cells as 5732e15):
build S5 128.8 -> ~110-130s, Z60 ~41-65s; eval did not cash out further
(memo hits were first defeated by winner-ORDER-sensitive cache keys -- fixed
by defining base_sum over SORTED rows, second amendment, golden regenerated
-- and are now bounded by engine-loop overhead, not drive). The memoization
architecture is sound and becomes decisive at CDS scale where n_cols grows;
at S5 scale the drive was no longer the bottleneck after 5732e15.
