# DESIGN: the dense cross fiber -- one launch per drive, one per write, no store walk

## The cost model after the quick wins (5 brains, U1 size)

    3.4 ms per area-round = 0.68 ms per brain-round; arithmetic ~2e5 flops.
    _rescale 20%  (walks a 2.3M-entry store to price k columns)
    _emit    15%  (per-step fold: GEMM, local index, nonzero)
    contribute 15%, append 8%, stimuli 6%, remainder ~35% (topk, glue)
    ~20 kernel launches per round.

The LSM store exists so a connectome that does not fit can still learn. The
aligner's cross fiber FITS: LEX x FEAT x B counts at study size is
4000 x 1000 x 5 x 4 bytes = 80 MB.

## The representation

`DenseAreaFiber`: per-brain int32 count matrix `C[b, i, j]`, per-column
`cmax[b, j]`, per-column float64 relative mass `M'[b, j]` (initialized to the
base in-degree, which is what the first rescale of a column would compute),
per-column scale `S[b, j]`. Max-relative pricing throughout
(DESIGN_hashed_aligner.md): a weight is `base * rel[cmax_j - C_ij] * S_j`.

    drive   one kernel: d[b, j] = S_j / d_j * SUM_{i in rows} present(i,j) * rel[cmax_j - C[b,i,j]]
    write   one kernel per (b, j in new): for i in prev with a synapse:
              c_old = C[i,j]; C[i,j] = c_old + 1; track the new column max;
              M' <- M' * rel[dcmax] + SUM (rel[cmax' - c_old - 1] - rel[cmax' - c_old]);
              S_j = setpoint / M'_j
    no episodes, no masks, no fold, no append.

Exactness: the same numbers as the store path (which recomputes M' from
scratch each round); M' is accumulated in float64 so incremental drift stays
far below the 5e-6 gate.

## Anchored assemblies are constants

A non-learning stimulus gives a constant drive; with deterministic ties the
area's winners for that stimulus set never change. LEX(word) and FEAT's
stimulus-only assembly for a bundle are therefore computed ONCE and cached,
which removes every LEX round and every round-0 FEAT round: a (word, bundle)
step is exactly its cross rounds.

## Gate and target

* the drive-replay parity test parametrized over store kinds: the dense
  fiber must match `numpy_sparse` at < 5e-6 like the store fiber does;
* dense equals store on the same writes (test);
* U1 = 1.000 on five brains, unchanged;
* target: <= 0.1 ms per brain-round, U1 (five brains) under 5 s.

## Result (2026-09-03)

Gate: dense equals the store fiber on the same writes (test), and the
aligner's drive-replay parity against `numpy_sparse` passes at < 5e-6 with
the dense fiber as it does with the store. U1 on five brains: 1.000 on every
brain, shuffled at chance -- unchanged.

    numpy learner             15-90 s per seed
    store fiber, 5 rounds     103 s / 5 brains
    + quick wins               73 s
    + 2 rounds per step        31 s
    dense fiber, 2 rounds       4 s / 5 brains        ~0.2 ms per brain-round
    dense fiber, 1 round        4 s   (0.97-1.00)
    dense fiber, 3 rounds       7 s   (1.000)

Target was <= 0.1 ms per brain-round and U1 under 5 s; the second is met,
the first is within 2x and what remains is `topk_select`, the area glue and
launch count -- a fused "drive + select + write" round would be the next
step, not more representation work.

Two bookkeeping facts worth knowing: the dense kernel counts only cells that
have a synapse (the store counts every co-fired pair and checks presence at
apply time), so `nnz` differs while drives agree; and `cmax` agrees exactly
because both take the max over present cells.
