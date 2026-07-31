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

#[pymodule]
fn na_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(area_weights_block, m)?)?;
    m.add_function(wrap_pyfunction!(area_weights_csr_rows, m)?)?;
    m.add_function(wrap_pyfunction!(stim_counts, m)?)?;
    Ok(())
}
