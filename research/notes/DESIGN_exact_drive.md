# Design: compute the drive, never store the substrate (task #85)

Written 2026-08-02, after benchmarking. This supersedes the "someday" framing:
the arithmetic is affordable by a wide margin and the blocker I recorded
dissolves under the right architecture.

## Why this is now the highest-value item

`research/notes/graded_similarity_and_sampler_load.md`: the exact substrate maps
similar inputs to similar assemblies (chance at f=0, monotone to 1.0). The
sampler flattens that to **0.906 for fully disjoint inputs** — 18× chance — at
low area load, with the error decaying and inverting as the area fills.

Every graded readout in this repo (next-token MRR, pattern completion,
association, category structure, every "shared area collapses" magnitude) is
scored on the property the sampler destroys. This is not a fidelity nicety.

## The identity that makes it work

    w_ij(t) = f(i, j, seed) * (1 + beta)^{c_ij}

`f` is the initial weight, already content-addressed by ABSOLUTE (row, col) via
`hash_area_weights` (`_sparse.py:592`), so it is recomputable and never needs
storing. `c_ij` is the number of potentiation events — nonzero only for pairs
that have co-fired, hence sparse. Drive decomposes:

    d_i = norm_i * [  SUM_{j in active} f(i,j)
                    + SUM_{(i,j) in pot, j in active} f(i,j) * ((1+beta)^{c_ij} - 1) ]

* **term 1 is a single dense block** `hash_area_weights(active_rows, 0, n)`
  summed over rows — no storage, and this is where the cost lives;
* **term 2 is sparse** and needs a batched hash lookup for the potentiated
  pairs only.

**The only stored state is potentiation.**

## Measured feasibility (Rust kernel present)

| | |
| --- | ---: |
| `hash_area_weights` throughput | **1.15–1.5 G cells/s** |
| exact drive, n=10⁴, k=200, total_k=400 | **3.43 ms / round** |
| one-time n² exact-in-degree pass | **0.09 s** |
| study-I scale (6360 rounds) | **0.4 min / seed** |

For comparison the sampled engine took ~40 min for study IV's 10 seeds × 3
arms, and `#77`/`#48` record connectome expansion at 41% of the training loop.
**The exact engine is plausibly FASTER than the one it replaces**, because it
does no growth, no realloc, and no scipy.

It was always memory, not arithmetic: n=10⁴ dense float32 is 400 MB *per
fiber*. Compute-don't-store needs O(n) per area plus the sparse potentiation.

## norm_init gets SIMPLER, not harder

`_norm_scale` (`_sparse.py:902`) is a read-time per-postsynaptic scale `1/d_j`,
justified by the algebra it documents:

    (w_0 / d_j) * prod_t (1 + beta_t)  ==  (w_0 * prod_t (1 + beta_t)) / d_j

Normalization commutes with multiplicative plasticity, so it stays a per-neuron
scalar applied at read time and storage stays on the unit scale (which is what
keeps `w_max`'s "multiples of the initial weight" semantics).

Crucially, `inverse_indegree(deg, n_pre, rows_known, p)` currently has to
ESTIMATE the contribution of rows that do not exist yet. **With every row
always addressable, `rows_known == n_pre` and `d_j` is exact.** The exact
engine removes an approximation rather than reproducing one. Expect numbers to
move; that is the fix working, not a regression.

## Architecture: a new engine, not a patch

Add `numpy_exact` alongside `numpy_sparse` / `explicit` / torch / cuda. Do NOT
mutate `numpy_sparse`.

* the repo's existing mechanism for "same model, different fidelity" IS the
  engine list, and `arbitrate()` already runs arms — this extends it rather
  than adding a parallel one;
* every standing result stays reproducible on the engine that produced it,
  which matters because a substrate change invalidates cross-change baselines
  ([[backbone-fingerprint-gap]]);
* migration becomes opt-in per study.

`arbitrate()` gains a fourth arm and `ARBITER_ARMS` becomes
`(explicit, materialized, sampled, exact)`.

**The blocker I recorded dissolves here.** "Recruitment assigns compact indices
sequentially, so materialized == prefix [0,w) must become a membership set" is
only a problem when patching the lazy engine. In an engine where every neuron
is addressable from t=0 there IS no recruitment and no compact space: compact
index == neuron id, and [[two-index-spaces-compact-vs-neuron-id]] stops being a
defect class. That is also the honest reading of the model — G(n,p) is fixed at
t=0 and firing never creates a synapse.

## Acceptance ladder

Ordered so each rung is falsifiable on its own. Given how many numbers here
have been fake-perfect, no rung may be skipped.

| rung | claim | test |
| --- | --- | --- |
| **L0** | same substrate | stimulus fiber + self-fiber rows elementwise identical to `materialized`, at norm_init **both** ways (the existing prototype did this only at False) |
| **L1** | same drive | pre-kWTA drive vector elementwise identical to `explicit` for one round, both norm_init settings |
| **L2** | same dynamics | full protocol winners identical, or differing ONLY in documented tie-breaks (`argsort` stable vs `argpartition`) |
| **L3** | **the point** | reproduces the graded-similarity curve: **0.0500 at f=0**, not 0.906. This is the rung that justifies the work |
| **L4** | affordable | wall-clock and peak RSS at n=10⁴ vs the sampled engine, measured not projected |
| **L5** | integrated | `arbitrate()` arm; then re-derive #90's magnitude list |

L3 is the acceptance test. L0–L2 can all pass on an engine that still merges.

## Risks, named

1. ~~**Potentiation state growth is the one unbounded quantity.**~~
   **MEASURED 2026-08-02, and it is not a risk.** Hebbian potentiation is
   `w *= 1+beta`, which leaves a zero at zero, so only pairs that co-fired AND
   have a synapse are ever stored. At n=2000, k=100, p=0.05, 40 stimuli:

   | | |
   | --- | ---: |
   | co-firing pairs (naive upper bound) | 400,000 |
   | synapses present (nnz) | 214,827 |
   | **potentiated pairs — the actual state** | **36,767** (9.2% of co-firing) |
   | as COO (int32×2 + f32) | **0.44 MB** |
   | dense n² float32 equivalent | 16.00 MB |

   36× smaller than dense, and it grows with *what was learned* rather than
   with n². `w_max = 20` clamps the multiplier, so storing the saturating float
   instead of a count is available if a pathological protocol ever needs it.
2. **Consumers that assume lazy semantics.** ~339 raw `areas[*].winners`
   accesses, plus everything reading `area.w`. A new engine where `w == n`
   always will expose any reader that means "how many have fired".
3. **Tie-breaking must be chosen deliberately**, not inherited. It decided
   borderline results before ([[global-rng-leak-nondeterminism]]).
4. **Door 6 and #81 re-land with this.** Door 6 (stim→area init drawn from the
   stream) was reverted pending #85; #81 (dense `Connectome` init is a stream)
   is the same defect class. In an engine that recomputes everything from
   `(row, col, seed)`, both are gone by construction rather than patched.

## What NOT to do

* **Do not tune the sampler offset.** Four policies were tried and measured at
  both ends; each trades one protocol against another because a scalar per
  fiber cannot carry a correlation. See `candidate_sampler_ground_truth.md`.
* **Do not mutate `numpy_sparse`.** Reproducibility of the back catalogue is
  the reason the arbiter exists.
* **Do not start with the language studies.** Start with L0–L3 on a two-area
  protocol where the answer is already known.
