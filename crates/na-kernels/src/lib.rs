//! Rust accelerators for the content-addressed connectome initialiser.
//!
//! WHY THIS CRATE EXISTS, AND WHY IT IS THIS FUNCTION AND NOT ANOTHER.
//!
//! `neural_assemblies/core/numpy_engine/_seeding.py` decides a synapse's
//! initial weight as a pure function of its (row, col) position -- that is what
//! makes materialisation order-free, and it is load-bearing for probe
//! isolation. It is also the single biggest one-time cost in the engine: 56% of
//! `materialize_area`, which is what bounds the finite-size ladder (n=32,000
//! currently takes ~30s to build).
//!
//! It is a good Rust target for a reason that most numeric ports do NOT have:
//! **bit-identity is guaranteed by construction, not hoped for.** The whole
//! computation is exact `u32` arithmetic with wrapping multiplies, so
//! `wrapping_mul` reproduces numpy's `uint32` overflow semantics exactly, and
//! the only float step is `m / 2^24` for an integer `m < 2^24`, which is
//! exactly representable in f32 and therefore rounds identically. Contrast the
//! CUDA port, where numerical parity had to be litigated repeatedly.
//!
//! WHAT NUMPY CANNOT DO HERE. The vectorised spelling must materialise the
//! whole block for every step of the chain -- two multiplies, an xor, then
//! fmix32's six shift/xor/multiply passes -- so it moves `O(n^2)` floats
//! through memory roughly seven times and is bandwidth-bound. This does it in
//! ONE pass with zero temporaries, and rows are independent so rayon spreads it
//! across cores. `area_weights_csr_rows` goes further and never forms the dense
//! block at all, which matters because the block is only ~p occupied (5%).
//!
//! THIS IS AN ACCELERATOR, NOT A REPLACEMENT. The numpy implementation stays
//! the executable specification; `_seeding.py` imports this in a try/except and
//! `tests/test_rust_kernels.py` asserts the two agree byte-for-byte.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Multipliers and the inhibitory salt, mirroring `_seeding.py`. Duplicated
/// rather than passed in so a drift between the two shows up as a test failure
/// in `test_rust_kernels.py` rather than as silently different weights.
const MUL_ROW: u32 = 2654435761;
const MUL_COL: u32 = 2246822519;
const INHIBITORY_SALT: u32 = 0x9E37_79B9;
const MANTISSA: u32 = 0x00FF_FFFF;
const MANTISSA_SCALE: f32 = 16_777_216.0;

/// Below this many cells, spinning rayon up costs more than the work. See the
/// comment in `area_weights_block` for the measurement.
const PARALLEL_MIN_CELLS: usize = 32_768;

/// Murmur3's `fmix32` avalanche finaliser.
///
/// `(r*A) ^ (c*B)` has low bits that are nearly a function of the low bits of
/// r and c alone, and the Bernoulli test reads exactly those 24 bits. This
/// scrambles the high bits back down into them.
#[inline(always)]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// The uniform [0, 1) draw for one cell.
///
/// `(h & 0xFFFFFF) as f32 / 2^24` is exact: the numerator is an integer below
/// 2^24 so it has at most 24 significant bits, and the divisor is a power of
/// two. numpy computes the identical expression in float32.
///
/// The COMPARISON is where bit-identity is easy to lose, so every threshold
/// here is derived the way numpy derives it. `u < p` in numpy is a float32
/// array against a weak Python float, which NEP 50 casts down to float32 --
/// hence `p as f32`, computed from the f64 the caller passed rather than from
/// an already-narrowed argument. `hash_stim_counts` compares the other way
/// round (`m < p * 2^24`, in f64 then narrowed), and that asymmetry is
/// preserved in `stim_counts` rather than tidied away.
#[inline(always)]
fn cell_uniform(row: u32, col: u32, seed: u32, finalize: bool) -> f32 {
    let mut h = row.wrapping_mul(MUL_ROW) ^ col.wrapping_mul(MUL_COL) ^ seed;
    if finalize {
        h = fmix32(h);
    }
    ((h & MANTISSA) as f32) / MANTISSA_SCALE
}

/// The weight of one cell: 0 (absent), 1 (excitatory) or `inhibitory_weight`.
#[inline(always)]
fn cell_weight(
    row: u32,
    col: u32,
    seed: u32,
    p: f32,
    inhibitory_prob: f32,
    inhibitory_weight: f32,
    finalize: bool,
) -> f32 {
    if cell_uniform(row, col, seed, finalize) >= p {
        return 0.0;
    }
    if inhibitory_prob <= 0.0 {
        return 1.0;
    }
    // The salt is what stops the inhibitory draw from being identical to the
    // presence draw -- without it every present synapse would be inhibitory
    // whenever inhibitory_prob >= p.
    let u = cell_uniform(row, col, seed ^ INHIBITORY_SALT, finalize);
    if u < inhibitory_prob {
        inhibitory_weight
    } else {
        1.0
    }
}

fn check_block(
    row_start: i64,
    row_end: i64,
    col_start: i64,
    col_end: i64,
) -> PyResult<(usize, usize)> {
    if row_start < 0 || col_start < 0 {
        return Err(PyValueError::new_err("block bounds must be non-negative"));
    }
    if row_end > u32::MAX as i64 || col_end > u32::MAX as i64 {
        return Err(PyValueError::new_err(
            "block bounds exceed the u32 index space",
        ));
    }
    Ok((
        (row_end - row_start).max(0) as usize,
        (col_end - col_start).max(0) as usize,
    ))
}

/// Dense `hash_area_weights` for absolute rows [row_start, row_end) x
/// cols [col_start, col_end). Drop-in for the numpy function of that name.
#[pyfunction]
#[pyo3(signature = (row_start, row_end, col_start, col_end, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_weights_block<'py>(
    py: Python<'py>,
    row_start: i64,
    row_end: i64,
    col_start: i64,
    col_end: i64,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (nr, nc) = check_block(row_start, row_end, col_start, col_end)?;
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);
    // Releasing the GIL is what lets rayon actually use the other cores; the
    // closure touches no Python objects.
    let data = py.allow_threads(|| {
        let mut out = vec![0.0f32; nr * nc];
        // Rayon's fan-out costs ~9us, which is most of the work for the small
        // blocks `_expand_connectomes` asks for one row at a time during
        // training. Measured: a (8, 120) block was 2.8x over numpy in parallel
        // against 20x+ for shapes that skip the pool. Serial below the
        // threshold, parallel above it -- identical output either way, since
        // rows never interact.
        let body = |i: usize, row_out: &mut [f32]| {
            let row = (row_start as u32).wrapping_add(i as u32);
            let rh = row.wrapping_mul(MUL_ROW) ^ pair_seed;
            for (j, cell) in row_out.iter_mut().enumerate() {
                let col = (col_start as u32).wrapping_add(j as u32);
                // Inlined so the common (absent) case costs one hash and one
                // compare, with no call into the salted branch.
                let mut h = rh ^ col.wrapping_mul(MUL_COL);
                if finalize {
                    h = fmix32(h);
                }
                if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                    continue;
                }
                *cell = if inhibitory_prob <= 0.0 {
                    1.0
                } else {
                    cell_weight(
                        row,
                        col,
                        pair_seed,
                        p,
                        inhibitory_prob,
                        inhibitory_weight,
                        finalize,
                    )
                };
            }
        };
        if nr * nc < PARALLEL_MIN_CELLS {
            out.chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        } else {
            out.par_chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        }
        out
    });
    data.into_pyarray_bound(py).reshape([nr, nc])
}

/// The same block, emitted directly as CSR triplets -- the dense form is never
/// allocated.
///
/// This is the memory half of the win. At n=32,000 a dense row-chunk of 1024
/// rows is 131 MB of which ~95% is zero, and the ladder is bounded by exactly
/// that transient. Returns `(indptr, indices, data)` for rows
/// [row_start, row_end) over `n_cols` columns, with `indptr` relative to
/// `row_start` (so `indptr[0] == 0`).
#[pyfunction]
#[pyo3(signature = (row_start, row_end, n_cols, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_weights_csr_rows<'py>(
    py: Python<'py>,
    row_start: i64,
    row_end: i64,
    n_cols: i64,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let (nr, nc) = check_block(row_start, row_end, 0, n_cols)?;
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);
    let (indptr, indices, data) = py.allow_threads(|| {
        // Per-row Vecs collected IN ORDER, so the result does not depend on
        // which thread finished first. Parallelism must not reach the output.
        let rows: Vec<(Vec<i32>, Vec<f32>)> = (0..nr)
            .into_par_iter()
            .map(|i| {
                let row = (row_start as u32).wrapping_add(i as u32);
                let rh = row.wrapping_mul(MUL_ROW) ^ pair_seed;
                // Sized from the expected count so the common case never
                // reallocates; p is a probability so this is a tight estimate.
                let mut idx = Vec::with_capacity(((nc as f32) * p * 1.25) as usize + 8);
                let mut val = Vec::with_capacity(((nc as f32) * p * 1.25) as usize + 8);
                for j in 0..nc {
                    let col = j as u32;
                    let mut h = rh ^ col.wrapping_mul(MUL_COL);
                    if finalize {
                        h = fmix32(h);
                    }
                    if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                        continue;
                    }
                    idx.push(j as i32);
                    val.push(if inhibitory_prob <= 0.0 {
                        1.0
                    } else {
                        cell_weight(
                            row,
                            col,
                            pair_seed,
                            p,
                            inhibitory_prob,
                            inhibitory_weight,
                            finalize,
                        )
                    });
                }
                (idx, val)
            })
            .collect();

        let total: usize = rows.iter().map(|(i, _)| i.len()).sum();
        let mut indptr = Vec::with_capacity(nr + 1);
        let mut indices = Vec::with_capacity(total);
        let mut data = Vec::with_capacity(total);
        indptr.push(0i64);
        for (idx, val) in rows {
            indices.extend_from_slice(&idx);
            data.extend_from_slice(&val);
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    });
    Ok((
        indptr.into_pyarray_bound(py),
        indices.into_pyarray_bound(py),
        data.into_pyarray_bound(py),
    ))
}

/// Per-neuron count of connected stimulus fibers -- the content-addressed
/// replacement for `rng.binomial(stim_size, p, size)`.
///
/// numpy has to build a `stim_size x n` block and reduce it; here the reduction
/// is fused into the hash loop, so nothing `O(stim_size * n)` is ever
/// allocated. Parallelised over NEURONS (the output axis) so no accumulator is
/// shared between threads.
#[pyfunction]
#[pyo3(signature = (stim_size, neuron_start, neuron_end, pair_seed, p, finalize=true))]
fn stim_counts<'py>(
    py: Python<'py>,
    stim_size: i64,
    neuron_start: i64,
    neuron_end: i64,
    pair_seed: u32,
    p: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (n, _) = check_block(neuron_start, neuron_end, 0, 0)?;
    if stim_size < 0 {
        return Err(PyValueError::new_err("stim_size must be non-negative"));
    }
    // `hash_stim_counts` compares `m < p * 2^24` (f64 product, narrowed by the
    // comparison) rather than `m / 2^24 < p`. Same predicate mathematically,
    // different rounding at the boundary -- so it is reproduced, not unified.
    let threshold = (p * MANTISSA_SCALE as f64) as f32;
    let out = py.allow_threads(|| {
        let mut out = vec![0.0f32; n];
        out.par_iter_mut().enumerate().for_each(|(j, slot)| {
            let neuron = (neuron_start as u32).wrapping_add(j as u32);
            let ch = neuron.wrapping_mul(MUL_COL) ^ pair_seed;
            let mut count = 0u32;
            for s in 0..stim_size as u32 {
                let mut h = s.wrapping_mul(MUL_ROW) ^ ch;
                if finalize {
                    h = fmix32(h);
                }
                if ((h & MANTISSA) as f32) < threshold {
                    count += 1;
                }
            }
            *slot = count as f32;
        });
        out
    });
    Ok(out.into_pyarray_bound(py))
}

/// Check an arbitrary row-index set, returning it as u32.
fn check_rows(rows: &[i64]) -> PyResult<Vec<u32>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        if r < 0 || r > u32::MAX as i64 {
            return Err(PyValueError::new_err(
                "row indices must be within the u32 index space",
            ));
        }
        out.push(r as u32);
    }
    Ok(out)
}

/// Column sums over an ARBITRARY set of rows, fused -- the block is never
/// materialised.
///
/// This is the exact-drive inner loop. An assembly's winners are a scattered
/// index set, not a contiguous range, so the Python side had to call
/// `area_weights_block` once per row: at n=1e4, k=200 that is 200 calls and
/// 200 allocations per projection round for ~1.7 ms of actual hashing, and it
/// measured 8 ms. The un-potentiated drive only ever needs the SUM, so nothing
/// of size k*n has to exist at all.
///
/// Parallelised over COLUMNS (the output axis), so no accumulator is shared
/// between threads and the result is independent of thread count -- the same
/// discipline `stim_counts` uses. Summation order within a column is fixed by
/// the row order given, so this is reproducible, not merely deterministic.
#[pyfunction]
#[pyo3(signature = (rows, col_start, col_end, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_rows_sum<'py>(
    py: Python<'py>,
    rows: numpy::PyReadonlyArray1<'py, i64>,
    col_start: i64,
    col_end: i64,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (_, nc) = check_block(0, 0, col_start, col_end)?;
    let rows = check_rows(rows.as_slice()?)?;
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);

    let out = py.allow_threads(|| {
        // Precompute each row's half-hash once; it is reused for every column.
        let rh: Vec<u32> = rows
            .iter()
            .map(|r| r.wrapping_mul(MUL_ROW) ^ pair_seed)
            .collect();
        let mut out = vec![0.0f32; nc];
        let body = |j: usize, slot: &mut f32| {
            let col = (col_start as u32).wrapping_add(j as u32);
            let ch = col.wrapping_mul(MUL_COL);
            let mut acc = 0.0f32;
            for (i, &h0) in rh.iter().enumerate() {
                let mut h = h0 ^ ch;
                if finalize {
                    h = fmix32(h);
                }
                if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                    continue;
                }
                acc += if inhibitory_prob <= 0.0 {
                    1.0
                } else {
                    cell_weight(
                        rows[i],
                        col,
                        pair_seed,
                        p,
                        inhibitory_prob,
                        inhibitory_weight,
                        finalize,
                    )
                };
            }
            *slot = acc;
        };
        if rows.len() * nc >= PARALLEL_MIN_CELLS {
            out.par_iter_mut().enumerate().for_each(|(j, s)| body(j, s));
        } else {
            out.iter_mut().enumerate().for_each(|(j, s)| body(j, s));
        }
        out
    });
    Ok(out.into_pyarray_bound(py))
}

/// The `(len(rows), n_cols)` block for an ARBITRARY row set, in one call.
///
/// Needed only where the fiber carries potentiation, since the learned factor
/// applies per (row, col). Same fused hashing as `area_rows_sum`, but it has
/// to materialise, so it is the fallback rather than the default path.
#[pyfunction]
#[pyo3(signature = (rows, col_start, col_end, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_rows_block<'py>(
    py: Python<'py>,
    rows: numpy::PyReadonlyArray1<'py, i64>,
    col_start: i64,
    col_end: i64,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (_, nc) = check_block(0, 0, col_start, col_end)?;
    let rows = check_rows(rows.as_slice()?)?;
    let nr = rows.len();
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);

    let data = py.allow_threads(|| {
        let mut out = vec![0.0f32; nr * nc];
        let body = |i: usize, row_out: &mut [f32]| {
            let row = rows[i];
            let rh = row.wrapping_mul(MUL_ROW) ^ pair_seed;
            for (j, cell) in row_out.iter_mut().enumerate() {
                let col = (col_start as u32).wrapping_add(j as u32);
                let mut h = rh ^ col.wrapping_mul(MUL_COL);
                if finalize {
                    h = fmix32(h);
                }
                if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                    continue;
                }
                *cell = if inhibitory_prob <= 0.0 {
                    1.0
                } else {
                    cell_weight(
                        row,
                        col,
                        pair_seed,
                        p,
                        inhibitory_prob,
                        inhibitory_weight,
                        finalize,
                    )
                };
            }
        };
        if nr * nc >= PARALLEL_MIN_CELLS {
            out.par_chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        } else {
            out.chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        }
        out
    });
    let arr = data.into_pyarray_bound(py);
    Ok(arr.reshape([nr, nc])?)
}

/// Accumulate a cached CSR adjacency into a drive vector, in one pass.
///
/// The companion to `area_rows_csr`. Given the fiber's edge list, the drive
/// for an assembly is a scatter-add over its rows' neighbours -- about
/// `k * n * p` increments into an `n`-vector, which at n=1e4 is 100k updates
/// into 40 KB and stays in L2.
///
/// This exists because doing it from numpy does NOT pay: the accumulation
/// itself is cheap (`np.bincount` of pre-gathered indices, 0.21 ms) but
/// GATHERING the rows out of the CSR costs more than the dense rescan it was
/// meant to replace (`np.repeat` + fancy-index, 1.81 ms against 0.74 ms).
/// Fusing gather and accumulate removes the intermediate entirely.
///
/// Serial on purpose: splitting rows across threads needs a private
/// accumulator each and a reduction over `n`, which costs more than the
/// scatter for the sizes this is called at.
#[pyfunction]
#[pyo3(signature = (indptr, indices, rows, n_cols))]
fn csr_row_counts<'py>(
    py: Python<'py>,
    indptr: numpy::PyReadonlyArray1<'py, i64>,
    indices: numpy::PyReadonlyArray1<'py, i32>,
    rows: numpy::PyReadonlyArray1<'py, i64>,
    n_cols: i64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (_, nc) = check_block(0, 0, 0, n_cols)?;
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let rows = rows.as_slice()?;
    let out = py.allow_threads(|| {
        let mut out = vec![0.0f32; nc];
        for &r in rows {
            let r = r as usize;
            if r + 1 >= indptr.len() {
                continue;
            }
            let (a, b) = (indptr[r] as usize, indptr[r + 1] as usize);
            for &j in &indices[a..b] {
                out[j as usize] += 1.0;
            }
        }
        out
    });
    Ok(out.into_pyarray_bound(py))
}

/// Adjacency (CSR) of an ARBITRARY row set -- the present synapses only.
///
/// The drive is a sum over PRESENT synapses, of which a row has about
/// `n * p`: at n=1e4, p=0.05 that is 500 out of 10,000. Computing it by
/// scanning the row evaluates 10,000 hashes to find 500 edges, so ~95% of
/// every round is spent rediscovering that the other 9,500 are absent -- and
/// absence is a FIXED FACT of G(n,p), which the model says is drawn once at
/// t=0 and never changes.
///
/// Returning the edge list lets the caller cache it per row and reduce the
/// drive to a scatter-add over `k * n * p` entries instead of `k * n` hash
/// evaluations. Same weights, same order, no approximation -- the graph is
/// simply enumerated instead of rescanned.
#[pyfunction]
#[pyo3(signature = (rows, n_cols, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_rows_csr<'py>(
    py: Python<'py>,
    rows: numpy::PyReadonlyArray1<'py, i64>,
    n_cols: i64,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let (_, nc) = check_block(0, 0, 0, n_cols)?;
    let rows = check_rows(rows.as_slice()?)?;
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);

    let (indptr, indices, data) = py.allow_threads(|| {
        // Per-row vectors collected IN ORDER, so the result cannot depend on
        // how rayon scheduled the work.
        let per_row: Vec<(Vec<i32>, Vec<f32>)> = {
            let build = |&row: &u32| {
                let rh = row.wrapping_mul(MUL_ROW) ^ pair_seed;
                // n*p with headroom, so the common case never reallocates.
                let cap = ((nc as f32) * p * 1.3) as usize + 16;
                let (mut idx, mut val) = (Vec::with_capacity(cap), Vec::with_capacity(cap));
                for j in 0..nc as u32 {
                    let mut h = rh ^ j.wrapping_mul(MUL_COL);
                    if finalize {
                        h = fmix32(h);
                    }
                    if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                        continue;
                    }
                    idx.push(j as i32);
                    val.push(if inhibitory_prob <= 0.0 {
                        1.0
                    } else {
                        cell_weight(
                            row,
                            j,
                            pair_seed,
                            p,
                            inhibitory_prob,
                            inhibitory_weight,
                            finalize,
                        )
                    });
                }
                (idx, val)
            };
            if rows.len() * nc >= PARALLEL_MIN_CELLS {
                rows.par_iter().map(build).collect()
            } else {
                rows.iter().map(build).collect()
            }
        };
        let total: usize = per_row.iter().map(|(i, _)| i.len()).sum();
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        let mut indices = Vec::with_capacity(total);
        let mut data = Vec::with_capacity(total);
        indptr.push(0i64);
        for (idx, val) in per_row {
            indices.extend_from_slice(&idx);
            data.extend_from_slice(&val);
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    });
    Ok((
        indptr.into_pyarray_bound(py),
        indices.into_pyarray_bound(py),
        data.into_pyarray_bound(py),
    ))
}

/// Gather block on an arbitrary row set AND an arbitrary column set.
///
/// The potentiated part of a fiber is confined to the columns the stored
/// outer products actually touch, which is a scattered subset -- so a
/// contiguous column range would span almost all of `n` and defeat the point.
/// With this, the plastic path materialises only `len(rows) x len(cols)` and
/// the untouched bulk stays inside `area_rows_sum`, never allocated.
#[pyfunction]
#[pyo3(signature = (rows, cols, pair_seed, p,
                    inhibitory_prob=0.0, inhibitory_weight=-1.0, finalize=true))]
#[allow(clippy::too_many_arguments)]
fn area_cells_block<'py>(
    py: Python<'py>,
    rows: numpy::PyReadonlyArray1<'py, i64>,
    cols: numpy::PyReadonlyArray1<'py, i64>,
    pair_seed: u32,
    p: f64,
    inhibitory_prob: f64,
    inhibitory_weight: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows = check_rows(rows.as_slice()?)?;
    let cols = check_rows(cols.as_slice()?)?;
    let (nr, nc) = (rows.len(), cols.len());
    let (p, inhibitory_prob, inhibitory_weight) =
        (p as f32, inhibitory_prob as f32, inhibitory_weight as f32);

    let data = py.allow_threads(|| {
        let mut out = vec![0.0f32; nr * nc];
        let body = |i: usize, row_out: &mut [f32]| {
            let row = rows[i];
            let rh = row.wrapping_mul(MUL_ROW) ^ pair_seed;
            for (j, cell) in row_out.iter_mut().enumerate() {
                let col = cols[j];
                let mut h = rh ^ col.wrapping_mul(MUL_COL);
                if finalize {
                    h = fmix32(h);
                }
                if ((h & MANTISSA) as f32) / MANTISSA_SCALE >= p {
                    continue;
                }
                *cell = if inhibitory_prob <= 0.0 {
                    1.0
                } else {
                    cell_weight(
                        row,
                        col,
                        pair_seed,
                        p,
                        inhibitory_prob,
                        inhibitory_weight,
                        finalize,
                    )
                };
            }
        };
        if nr * nc >= PARALLEL_MIN_CELLS {
            out.par_chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        } else {
            out.chunks_mut(nc.max(1))
                .enumerate()
                .for_each(|(i, r)| body(i, r));
        }
        out
    });
    let arr = data.into_pyarray_bound(py);
    Ok(arr.reshape([nr, nc])?)
}

/// Exact per-column IN-DEGREE over rows [0, n_rows) -- the `norm_init` divisor.
///
/// `stim_counts` computes the same shape of quantity and is deliberately NOT
/// reused: it compares `m < p * 2^24` where `cell_weight` compares
/// `m / 2^24 >= p`. Those are the same predicate mathematically but are
/// evaluated differently, so they are not guaranteed to agree at the boundary
/// -- and a degree that disagreed with the weights actually drawn for the
/// fiber would corrupt the norm_init divisor silently.
///
/// Measured, they agree everywhere tested: p in {0.01, 0.05, 0.1, 0.2, 0.3,
/// 0.5, 0.0499999, 0.333333} over 1500 columns, ZERO differing columns. So
/// this duplication buys safety, not correctness, and could be collapsed if
/// the equivalence were ever proven rather than sampled.
///
/// This replaces a chunked numpy pass that allocated the block in slabs and
/// measured 0.8-2.4 s at n=1e4 -- one-time and cached, but paid per fiber.
#[pyfunction]
#[pyo3(signature = (n_rows, col_start, col_end, pair_seed, p, finalize=true))]
fn area_indegree<'py>(
    py: Python<'py>,
    n_rows: i64,
    col_start: i64,
    col_end: i64,
    pair_seed: u32,
    p: f64,
    finalize: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (_, nc) = check_block(0, 0, col_start, col_end)?;
    if n_rows < 0 || n_rows > u32::MAX as i64 {
        return Err(PyValueError::new_err(
            "n_rows must be within the u32 index space",
        ));
    }
    let p = p as f32;
    let out = py.allow_threads(|| {
        let mut out = vec![0.0f32; nc];
        let body = |j: usize, slot: &mut f32| {
            let col = (col_start as u32).wrapping_add(j as u32);
            let ch = col.wrapping_mul(MUL_COL) ^ pair_seed;
            let mut count = 0u32;
            for r in 0..n_rows as u32 {
                let mut h = r.wrapping_mul(MUL_ROW) ^ ch;
                if finalize {
                    h = fmix32(h);
                }
                // The `cell_weight` predicate, not the `stim_counts` one.
                if ((h & MANTISSA) as f32) / MANTISSA_SCALE < p {
                    count += 1;
                }
            }
            *slot = count as f32;
        };
        if (n_rows as usize) * nc >= PARALLEL_MIN_CELLS {
            out.par_iter_mut().enumerate().for_each(|(j, s)| body(j, s));
        } else {
            out.iter_mut().enumerate().for_each(|(j, s)| body(j, s));
        }
        out
    });
    Ok(out.into_pyarray_bound(py))
}

/// Replay `VirtualWeights._chain`: `n` float32 multiply-then-clip rounds.
///
/// The numpy spelling walks ROUNDS on the outside -- one masked multiply and
/// one whole-array clip per round -- so a cell touched `n` times costs `n`
/// numpy calls on an array of `len(values)`. The arrays are tiny (the
/// deviation cells of one row), so that is dispatch cost, not arithmetic:
/// measured 8,848 calls and 47,864 clips inside a single 32-assembly virtual
/// build, together about a quarter of it.
///
/// EXACT, NOT EQUIVALENT. Two properties of the numpy form are semantics and
/// are reproduced rather than optimised away:
///
///   * the multiply is repeated, never folded into `g^count`. Plasticity is
///     float32 `w *= 1 + beta` applied one event at a time, and `powi` would
///     round differently -- the dense engine's arithmetic is the contract.
///   * the clip runs on EVERY element every round, including elements whose
///     own count has already run out. That only matters for a value that
///     starts outside the bounds, but it is what the dense engine does.
///
/// Walking cells on the outside and rounds on the inside is the same sequence
/// of operations per element, so the result is bit-identical while the whole
/// thing becomes one call over contiguous memory.
#[pyfunction]
#[pyo3(signature = (values, counts, g, lo, hi))]
fn chain_clip<'py>(
    py: Python<'py>,
    values: numpy::PyReadonlyArray1<'py, f32>,
    counts: numpy::PyReadonlyArray1<'py, i64>,
    g: f32,
    lo: Option<f32>,
    hi: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let v = values.as_slice()?;
    let c = counts.as_slice()?;
    if v.len() != c.len() {
        return Err(PyValueError::new_err(
            "values and counts must have the same length",
        ));
    }
    let out = py.allow_threads(|| {
        let rounds = c.iter().copied().max().unwrap_or(0);
        let mut out = vec![0.0f32; v.len()];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut x = v[i];
            let n = c[i];
            for r in 1..=rounds {
                if n >= r {
                    x *= g;
                }
                // Gated on `hi` exactly as the python is: no upper bound
                // means no clip at all, not a one-sided one.
                if let Some(h) = hi {
                    if let Some(l) = lo {
                        if x < l {
                            x = l;
                        }
                    }
                    if x > h {
                        x = h;
                    }
                }
            }
            *slot = x;
        }
        out
    });
    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn na_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(area_weights_block, m)?)?;
    m.add_function(wrap_pyfunction!(area_weights_csr_rows, m)?)?;
    m.add_function(wrap_pyfunction!(stim_counts, m)?)?;
    m.add_function(wrap_pyfunction!(area_rows_sum, m)?)?;
    m.add_function(wrap_pyfunction!(area_rows_block, m)?)?;
    m.add_function(wrap_pyfunction!(area_cells_block, m)?)?;
    m.add_function(wrap_pyfunction!(area_rows_csr, m)?)?;
    m.add_function(wrap_pyfunction!(csr_row_counts, m)?)?;
    m.add_function(wrap_pyfunction!(area_indegree, m)?)?;
    m.add_function(wrap_pyfunction!(chain_clip, m)?)?;
    Ok(())
}
