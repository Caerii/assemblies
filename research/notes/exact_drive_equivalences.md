# Why the fast paths compute the same thing — proof sketches

`numpy_exact` replaced several direct computations with faster ones. Tests show
agreement on the cases someone thought to write; these sketches say **why the
agreement must hold in general**, which is what makes a change reviewable and a
result reproducible.

Notation. A fiber has initial weights `f(i,j)` — a fixed function of
`(row, col, seed)`, drawn at t=0 and never modified, since firing creates no
synapse. `c_ij` counts potentiation events on `(i,j)`. The weight is
`w_ij = f(i,j) * (1+beta)^{c_ij}`. `W` is the source assembly, `d_j` neuron
`j`'s in-degree, `k` the cap.

Throughout, "the top-k" means the k largest under the **total** order
*(drive descending, index ascending)* — a total order, so the answer is unique
and "equal outputs" is a meaningful claim.

---

## 1. Probabilistic pivot, exact result

**Claim.** Let `C = {i : d_i > pi}` for any threshold `pi`. If `|C| >= k` then
`T ⊆ C`, where `T` is the true top-k.

**Proof.** Let `tau` be the k-th largest value of `d`. Every element of `C`
exceeds `pi`, and there are at least `k` of them, so at least `k` values exceed
`pi`; hence `tau > pi`. Any `i ∈ T` has `d_i >= tau > pi`, so `i ∈ C`. ∎

**Consequences that matter.**

* Correctness does **not** depend on how `pi` was chosen — only on the checked
  condition `|C| >= k`. The sample can be biased, adversarial, or degenerate;
  a bad pivot changes only how much work the refinement does.
* When the check fails the code falls back to selecting over the whole array,
  so the result is exact unconditionally.
* The refinement inside `C` must use the same total order. `flatnonzero`
  returns ASCENDING indices and every subsequent operation is a mask or a
  prefix of `C`, so the index-ascending half of the tie-break is preserved.
  This is the step to be careful about: a refinement that sorted `C` by value
  with an unstable kind would break it.

The stride sample uses no RNG, so it cannot perturb reproducibility.

Checked adversarially (60/60 identical to full selection) on: all-ties,
all-identical, mass aligned with the sampling stride, heavy-tailed, and
mass at the far end of the index range.

---

## 2. Per-event application == dense exponent accumulation

**Claim.** Applying each stored outer product `(S_e, T_e, m_e)` in turn as an
in-place multiply by `(1+beta)^{m_e}` on the sub-block `S_e x T_e` produces
`f(i,j) * (1+beta)^{c_ij}` at every cell.

**Proof.** By construction `c_ij = sum_e m_e * 1[i in S_e] * 1[j in T_e]`.
Exponents add over multiplication, so

    (1+beta)^{c_ij} = prod_e (1+beta)^{m_e * 1[i in S_e] * 1[j in T_e]}

and each factor is `1` exactly when the cell lies outside `S_e x T_e`. So the
product of the per-event multiplies equals the total factor. ∎

**Caveat, and it is not cosmetic.** The proof holds in exact arithmetic.
Floating-point multiplication is **not associative**, so the ORDER of events
decides the last ulp of any cell touched by more than one event. The
implementation therefore iterates `sorted(touched)` rather than raw set order:
set iteration is deterministic for int keys only as an undocumented CPython
detail, and it depends on the table's growth history.

---

## 3. norm_init as a read-time scale

**Claim.** Dividing a neuron's SUMMED drive by `d_j` equals initialising every
incoming synapse of `j` to `1/d_j`.

**Proof.** Plasticity is purely multiplicative, so

    sum_i (f(i,j)/d_j) * (1+beta)^{c_ij} = (1/d_j) * sum_i f(i,j) * (1+beta)^{c_ij}

because `d_j` does not depend on `i`. ∎

This is what licenses folding the scale into the cached base drive: with
`base_j = sum_i f(i,j)` and `corr_j = sum_i (w_ij - f(i,j))`,

    stored_j = base_j / d_j     and     drive_j = stored_j + corr_j / d_j

sums to `(base_j + corr_j)/d_j = (sum_i w_ij)/d_j`, as required. The correction
must be scaled by the SAME `d_j`; that it is raw when computed is why the code
applies `scale[cols]` to it explicitly.

**Exactness of `d_j` here.** `inverse_indegree` charges rows that do not exist
yet at the ambient rate `p * (n_pre - rows_known)`. In this engine every row
exists, so `rows_known == n_pre`, that term is zero, and `d_j` is the true
in-degree. This engine removes the estimate rather than reproducing it.

---

## 4. Correction confined to touched columns

**Claim.** For `j ∉ TouchedCols(W)`, the correction is exactly zero.

**Proof.** `TouchedCols(W)` is the union of `T_e` over every stored event whose
`S_e` meets `W`. If `j` is outside it then for all `i ∈ W`, `c_ij = 0`, so
`w_ij = f(i,j)` and `sum_i (w_ij - f(i,j)) = 0`. ∎

So the base sum already carries those columns exactly, and only the touched
ones need a materialised sub-block — which is why the `(k, n)` block never has
to exist.

---

## 5. The base-drive cache is valid indefinitely

**Claim.** `base_j(S) = sum_{i in S} f(i,j,seed)` may be memoised forever.

**Proof.** `f` is a pure function of `(row, col, seed)`; `G(n,p)` is drawn at
t=0 and firing never creates or removes a synapse. So `base(S)` depends only on
`(S, fiber, seed)`, all immutable. Everything time-varying is potentiation,
which enters as a separate additive correction. ∎

This is exactly why caching the DRIVE is sound where caching a mutable weight
matrix would not be.

---

## 6. Edge-list count == dense scan

**Claim.** When `inhibitory_prob == 0`, accumulating the cached adjacency gives
the same vector as scanning the fiber.

**Proof.** With no inhibitory draw, `f(i,j) ∈ {0, 1}` and `f(i,j) = 1` exactly
when `(i,j) ∈ E`. So `sum_{i in W} f(i,j) = |{i in W : (i,j) ∈ E}|`, a count. ∎

The guard matters: with inhibitory synapses `f` takes a third value and the sum
is no longer a count, so that path keeps scanning. Verified equal at
p ∈ {0.01, 0.05, 0.1}.

---

## 7. Clamping only potentiated regions

**Claim.** `min(w_ij, w_max)` applied only on potentiated sub-blocks equals
applying it to the whole block, provided `w_max >= 1`.

**Proof.** An untouched cell has `c_ij = 0`, hence `w_ij = f(i,j) ∈
{0, 1, inhibitory_weight}`, all `<= 1 <= w_max`, so the clamp is the identity
there. ∎

The clamp must come after ALL events for a region, not between them: clamping a
partially-built product would cap a value still being multiplied.

---

## What is NOT proven here, and is therefore tested

* That `f` in Rust equals `f` in numpy. This is bit-identity of an integer hash
  and is asserted directly (`test_rust_kernels.py`), not argued.
* That float32 and float64 select the same winners. Precision changes the size
  of the tied band, which is a modelling question, not an algebraic identity —
  measured instead (1–2 tied at n=3e5; float32 and float16 both agreed 1.0000
  with float64 there, but that is a measurement at those parameters, not a
  guarantee).
