# The k-WTA pricing law was implemented twice, and the copies diverged

*2026-07-30. Companion to `neural_assemblies/core/_pricing.py` and
`neural_assemblies/tests/test_engine_pricing.py`.*

## What was wrong

`project_into` selects winners by `top-k` over a concatenation of two
populations drawn on different scales — materialized incumbents (real summed
weights, divided per neuron by in-degree under `norm_init`) and sampled
candidates (order statistics of a binomial, divided by a single scalar). If the
two divisors are not commensurable, `top-k` is comparing different units.

Two fixes to that arithmetic landed in July 2026 — `c7ce506` (explicit-source
fibers were skipping `_norm_scale` entirely) and `54e6c00` (the candidate
divisor ignored per-fiber `n_pre`). Both landed in `numpy_engine/_sparse.py`
only. The torch engine carried a hand-written copy whose header comment read
*"this is the on-device mirror"*, and the mirror never received either fix.

A third and fourth divergence surfaced while unifying:

3. torch built `input_sizes` from `area.k` unconditionally, where numpy charges
   the actual `winners.size` under `norm_init`. These differ whenever an area's
   cap is not exactly `k` — a fixed assembly, an explicit source, a
   partially-converged area.
4. `CSRConn.column_indegree` counted present synapses over **all** stored rows
   while `_norm_scale_area` then charged the rows beyond `rows_known` a second
   time through its unknown-row term, double-counting the degree. (Latent in the
   configurations measured here — `_nrows <= rows_known` in all of them — but
   wrong, and it fires whenever the connectome is materialized ahead of the
   source's winner count.)

## What it cost

k=100, p=0.05, beta=0.05, 15 rounds, `norm_init=True`. Engine `w` (neurons ever
fired) and round-to-round assembly stability:

| n_src | n_tgt | numpy (fixed) | torch (mirror) |
|---|---|---|---|
| 1000 | 10000 | w=154, stab 1.000 | **w=100 SEALED**, stab 1.000 |
| 5000 | 5000 | w=117, stab 1.000 | w=114, stab 0.990 |
| 10000 | 1000 | w=111, stab 1.000 | **RuntimeError**, area exhausted |

Torch reproduced the *pre-fix numpy* table at both signs: candidates
over-divided when `n_src < n_tgt` seals the area at exactly `k`; under-divided
when `n_src > n_tgt` they always win and the area materializes toward `n` until
`_sample_truncated_normal_gpu` raised `effective_n=64, k=100`.

**Read the first row carefully.** The broken engine scores *better* on
stability — 1.000 — because a sealed area is trivially stable. Its cap is
frozen, so every cap-vs-assembly readout reads exactly 1.0000. That is the
perfect-score signature: the measurement is dead, not good.

## Why a parity suite did not catch it

`test_torch_parity.py` exists precisely to compare these two engines, and it was
green throughout. Two structural reasons, both of which generalize:

1. **A single module-level `N = 10000`.** Every area is the same size, so
   `tgt.n == n_pre` always and the per-fiber divisor bug cannot manifest *by
   construction*. Not under-tested — untestable.
2. **Qualitative thresholds.** Parity is asserted as "forms an assembly and
   stabilizes above a threshold". A sealed area satisfies that at 1.000.

The general lesson: a test whose true-negative case has never been constructed
has unmeasured power. For any qualitative-threshold assertion in this repo,
the question worth asking is whether a known-degenerate state also passes it.

## The fix

`neural_assemblies/core/_pricing.py` holds the law — `candidate_divisor`,
`inverse_indegree`, `area_fiber_activity` — and both engines call it. In-degree
*extraction* stays per-engine, because counting a column's realized synapses
genuinely depends on the storage format (numpy's maintained `_deg_counts`,
torch's CSR scatter, dense explicit connectomes). Everything that is arithmetic
is shared.

`test_engine_pricing.py` asserts the law directly rather than asserting a
threshold: the divisor as a pure function (including the *sign* of both
corrections), and then `k < w < n` on both engines at all three size ratios.
Seal and exhaust each violate one of those bounds, and no degenerate state
satisfies both.

Verification that the numpy path is behaviour-preserving: `w` reads 154 / 117 /
111 before and after the refactor, unchanged.

## Residual, still open

After unification the two engines agree on the *pricing* — divisor 50.0000 vs
50.0000 and 250.0000 vs 250.0000, `mean(1/d)` matching to 0.3% in the
homogeneous case — and both failure modes are gone. They do **not** agree on
recruitment rate:

| n_src | n_tgt | numpy | torch |
|---|---|---|---|
| 1000 | 10000 | w=154, stab 1.000 | w=506, stab 0.734 |
| 5000 | 5000 | w=117, stab 1.000 | w=157, stab 0.960 |
| 10000 | 1000 | w=111, stab 1.000 | w=182, stab 0.948 |

Ruled out: the candidate sampler. Forcing torch onto the numpy CPU sampler
(`_gpu_sampling=False`) leaves it at w=464 — the gap does not close.

The signal points at lazy materialization rather than pricing. At a matched
state, numpy's connectome is 632x200 backing `src.w=329, tgt.w=150`, while
torch's is 387x231 backing `src.w=387, tgt.w=231` — numpy over-allocates rows in
blocks, torch allocates exactly, so incumbent drive accumulates differently.
Incumbent median then reads 0.3049 (numpy) against 0.1549 (torch) with the same
candidate mean of 0.1000, which is self-reinforcing: more recruitment thins the
median, and a thinner median recruits more.

Torch is also **not reproducible across processes** at fixed `seed=1` — the same
cell read w=484, 514, 506 on three runs — so differences below ~10% cannot be
read finely. That is its own defect and compounds the above.

Until this closes, any quantitative result computed on `torch_sparse` with
unequal area sizes is engine-dependent and should be reported with the engine
named.
