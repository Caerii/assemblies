# DESIGN: stop storing the connectome

Written before implementing. The guard it depends on (`test_connectome_
representation_fingerprint`, commit `6e050a2`) is already in place and verified
to fail against a single perturbed cell out of 630000.

## The problem, in numbers

A materialised fiber is stored DENSE at `n_src x n_tgt`. The S5 word-problem
organ needs, per worker:

    arc <-> state, dense    40000 x 8400 x 4 bytes x 2 directions = 2.69 GB

At 14 workers that is 37.6 GB, and growth REALLOCATES (the new buffer is built
while the old is live), so the peak is about twice that. It ran out of 45 GB
free and the study died. The current fix is memory-tiered scheduling, which is
a workaround.

SPARSITY DOES NOT RESCUE IT. At the density the organs run (`organ_p=0.4`) the
fiber is 40% occupied: CSR stores 134M nonzeros at 8 bytes for 1.07 GB against
dense's 1.34 GB. A 20% saving does not change what is runnable.

## The observation

`hash_area_weights(row, col, pair_seed, p)` is a PURE FUNCTION of the cell's
position -- that is what [[content-addressed-synapse-init]] bought, and it is
already relied on for reproducibility. So the base connectome is recomputable
and needs no storage at all.

What deviates from base is only Hebbian potentiation, and the rule is
MULTIPLICATIVE (`_apply_plasticity`: `w *= (1 + beta)`, then `clip`). Three
consequences:

1. The deviation is an INTEGER COUNT, not a float weight:
   `w[i,j] = clip(base(i,j) * (1+beta)^n[i,j], lo, hi)`.
2. Multiplicative updates never create a nonzero, so the sparsity PATTERN is
   also a pure function of the hash. `_deg_counts` -- per-column nonzero
   counts, which `norm_init` divides by -- becomes analytic rather than a scan.
3. Only pairs that actually co-fired have `n > 0`. For S5 that is at most
   240 transitions x 70 x 70 = 1.18M entries against 336M cells.

Estimated: ~2.69 GB -> ~14 MB for the S5 organ.

## Why the clamp does not break it, and what does

`w_max = 20.0` by DEFAULT, so clipping is always on and had to be checked
rather than assumed. It is safe: for an upper bound with a growth factor > 1,
iterated clip-then-multiply equals a single clip of the closed form
(`clip(clip(x*a),a) == clip(x*a^2)` in both the saturated and unsaturated
cases), and the bounds `_weight_bounds()` are constant across the run.

HOMEOSTATIC SCALING IS THE ONE THAT BREAKS IT. `_normalize_area_columns`
rescales a whole column to a setpoint AFTER each update, so the sequence is
multiply, clip, rescale, multiply, clip, rescale. A clip applied to an
already-rescaled value does not factor out, and the closed form is wrong.

It is OFF by default (`synaptic_scaling=False`) and off in every sequence-organ
run, so this is a boundary rather than a blocker -- but it must be a STRUCTURAL
boundary, not a comment. A fiber with scaling enabled falls back to dense
storage, and the selection happens where the fiber is created so the wrong
combination cannot be spelled ([[one-canonical-way]]).

## Interface

Nothing above the engine may notice. The drive path currently traverses the
stored block; it would instead regenerate the `k` winner ROWS and add the
sparse deviations -- 70 x 8400 = 588K cells per projection rather than a
336M-cell traversal, so this is expected to be faster as well as smaller, and
that prediction is recorded here to be checked rather than assumed.

## Acceptance

1. `test_connectome_representation_fingerprint` passes UNCHANGED, on all five
   configurations. The golden was captured from the dense implementation; if
   the new one cannot reproduce it byte-for-byte, the new one is wrong or the
   old one was, and either way the change does not land until that is resolved.
2. The full non-slow suite passes, cold.
3. The S5 study reproduces its result on the tiered scheduler and without it.
4. Memory and wall-clock are reported as measurements, not estimates, and the
   estimates above are compared against them in the same note. If the change
   is smaller but SLOWER, that is reported and the trade stated.

## What is deliberately NOT in scope

The torch and cuda engines. [[pricing-law-implemented-twice]] is the standing
warning that a fix applied to one sibling and not the other is how this
codebase produces a confident wrong answer, so either the change is confined to
numpy_sparse behind an explicit capability check, or it lands in both. It will
be the former, and the check will raise rather than silently no-op -- the same
shape as `set_refracted` on numpy_exact.
