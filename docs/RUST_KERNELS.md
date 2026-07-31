# The optional Rust kernels (`crates/na-kernels`)

An accelerator for the content-addressed connectome initialiser. **Everything
works without it** — absent, unbuilt, or disabled, the engine falls back to the
numpy implementation in `neural_assemblies/core/numpy_engine/_seeding.py` and
produces the same numbers, just slower.

## Build

```bash
cargo build --release --manifest-path crates/Cargo.toml -p na-kernels
```

Then put the library where Python can import it — on Windows the extension
suffix is `.pyd`, on Linux/macOS `.so`:

```bash
python scripts/install_rust_kernels.py
```

Verify:

```bash
python -c "from neural_assemblies.core.numpy_engine._seeding import rust_kernels; print(rust_kernels())"
```

Set `NEURAL_ASSEMBLIES_NO_RUST=1` to force the numpy path in a process — that
is how `tests/test_rust_kernels.py` A/Bs the two.

## What it does, and what it measured

`_seeding.py` decides a synapse's initial weight as a pure function of its
`(row, col)` position. That is load-bearing, not an optimisation: it is what
makes materialisation order-free, and order-freedom is what makes read-only
probing sound. It is also the biggest one-time cost in the engine.

The numpy spelling is vectorised but must materialise the whole block for each
step of the chain — two multiplies, an xor, then `fmix32`'s six shift / xor /
multiply passes — so it moves `O(n²)` values through memory about seven times.
It is bandwidth-bound, not compute-bound, which is why the earlier attempts to
speed it up in numpy (chunking to stay cache-resident, `np.take(out=)`,
in-place `mix32`) bought little or nothing. The fix had to change the
*representation*, not the operation.

The Rust kernel does it in one pass with zero temporaries, and rows are
independent so rayon spreads it across cores. `area_weights_csr_rows` goes
further and emits CSR triplets directly, so the dense block never exists at
all — which matters because it is only ~5% occupied.

`materialize_area`, measured end to end:

| n | numpy | Rust | |
|---|---|---|---|
| 8,000 | 1.39 s | 0.065 s | **21×** |
| 16,000 | 5.58 s | 0.160 s | **35×** |
| 32,000 | 23.07 s | 0.544 s | **42×** |

Identical `nnz` (51,204,367 at n=32,000) and identical settled winners in every
case. On the block kernel alone the range is 8–32× depending on shape.

The training loop is **not** materially faster (1.02× on a 24-word lexicon) —
its blocks are small and its cost is elsewhere, in connectome-expansion
bookkeeping. This accelerates materialisation, which is what bounded the
finite-size ladder.

## Why bit-identity is a guarantee here, not a hope

Unlike the CUDA port — where numerical parity had to be litigated repeatedly —
this one is exact by construction:

- the hash is `u32` arithmetic with wrapping multiplies, which
  `wrapping_mul` reproduces exactly;
- the only float step is `m / 2²⁴` for an integer `m < 2²⁴`, which has at most
  24 significant bits and a power-of-two divisor, so it is exactly
  representable in `f32` and rounds identically.

The one place drift is possible is the **comparisons**, where numpy's NEP 50
narrowing decides the type. Two conventions are in play and the kernel
reproduces both rather than unifying them:

- `hash_uniform_2d` compares `m / 2²⁴ < p` (narrow `p` to `f32`);
- `hash_stim_counts` compares `m < p * 2²⁴` (product in `f64`, then narrowed).

Unifying those would move a handful of synapses per block, and nothing would
fail loudly. `tests/test_rust_kernels.py` asserts both head-on, along with the
salted inhibitory draw, `finalize=False`, absolute row addressing, and a whole
materialised brain through the real `materialize_area` wiring.

## Adding a kernel

The bar is the one above: **only port something whose bit-identity you can
argue from the arithmetic**, and assert it in `test_rust_kernels.py` against
the numpy implementation rather than against a recorded golden. The numpy code
stays the executable specification; if the two disagree, the numpy one is right.
