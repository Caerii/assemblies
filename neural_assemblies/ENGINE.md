# What the engines are, and what the sparse one approximates

Most of the time lost in this repo has gone to one question: *is this number
the substrate, or the machinery?* That question is hard mainly because nothing
wrote down what the sparse engine promises. This does.

---

## Two engines, one model

| | `explicit` | `numpy_sparse` (default) |
| --- | --- | --- |
| neurons | all `n` exist from the start | only the `w` that have won exist |
| weights | dense `n x n` per fiber | blocks sized to `w`, grown on recruitment |
| drive of a neuron | computed exactly | exact for the `w`; **invented** for the rest |
| memory | `O(n^2)` -- the binding limit | `O(w * total_k)` |
| exact? | yes | **only where it does not sample** |

Both implement the same model: an area's `k` highest-drive neurons fire, drive
is `d_i(x) = sum over j in x of w_ij`, and firing potentiates by `1+beta`.

## The one place they differ

`sample_new_winner_inputs` (`compute/sparse_simulation.py`). Deciding whether a
never-fired neuron outbids a settled one needs that neuron's drive, and the
sparse engine has not drawn its synapses. So it *invents* one, as an order
statistic of the unmaterialized pool.

**This is the entire accuracy gap.** Measured 2026-07-31 on the merge protocol
(50 rounds, support = distinct winners, units of `k`, areas A/B/C):

| n, k | explicit | materialized sparse | sampled sparse |
| ---: | ---: | ---: | ---: |
| 1000, 32 | 6.1 / 6.0 / 8.6 | 5.9 / 5.9 / 8.3 | 8.9 / 9.5 / 10.0 |
| 2000, 45 | 6.1 / 6.8 / 10.0 | 7.2 / 5.2 / 8.7 | 8.2 / 8.3 / 8.5 |
| 4000, 63 | 6.9 / 5.7 / 8.6 | 7.2 / 7.5 / 11.1 | 10.8 / 10.1 / 10.8 |

Bypass the sampler (`materialize_area`, so `w == n` and it offers zero
candidates) and the sparse engine **reproduces the explicit engine**. Not
pricing, not `norm_init`, not the tie-break, not `beta` -- the sampler.

### Why it cannot be fixed by tuning

The model gives correlation between similar inputs for free:

    d_i(x)  = d_i(S) + d_i(U)
    d_i(x') = d_i(S) + d_i(U')

For inputs sharing `S` the shared term is *the same random variable* -- but
only if a neuron's synapses are a fixed fact. An invented drive has discarded
that. Four offset policies were tried (per-key, correlation-discounted,
derived-by-rank, and none); each traded one protocol against another, because
none had the information. Do not re-derive them -- see
`research/notes/candidate_sampler_ground_truth.md`.

Measured directly on `pnas2020_scaling` (n=10000, k=100, p=0.01), where the
trade is visible in one table -- `chance = k/n = 0.0100`:

| arm | persistence | separate_overlap |
| --- | ---: | ---: |
| **materialized (samples nothing)** | **1.0000** | **0.0100** |
| sampled, offset on | 1.0000 | 0.0900 |
| sampled, offset 0 | 0.9800 | 0.0100 |

The exact model gets both. No offset setting does: it buys persistence and
pays in separation. The same lever swings the coin's fairness (attractor
0.825 with the offset, 0.475 without). A scalar per fiber cannot carry a
correlation, which is why tuning it keeps failing.

**Keying fixes the diagonal, not the off-diagonal.** Making a repeated
projection draw the same candidates fixes "same input, same drive". It cannot
give "similar inputs, similar drives" -- that correlation is gone at the draw.
So protocols scored on identity or distinctness (`separate`, the coin) are
repaired by keying alone, while ones scored on PARTIAL overlap are not:
next-token MRR reads 0.1159 exact, 0.0744 keyed-with-offset, 0.0901 keyed
without it. Expect the same for pattern completion from a cue, association,
and graded category structure.

The real fix is to *compute* the drive rather than sample it: `_init_area_block`
is addressed by absolute `(row, col)` via `hash_area_weights`, so any neuron's
drive is computable in `O(|active| * n)` time and `O(n)` memory, no weight
storage. It is memory, not arithmetic, that makes `explicit` unusable at scale.

### Known artifacts of the sampler

* merge support runs ~1.4x high
* pattern completion from a 50% cue reads **1.000** where the exact model reads
  0.67-0.89
* `project_persistence` reads **1.000** where the exact model reads 0.75-0.86

The last two are baked into self-recorded goldens. A `1.000` here is a claim
about the sampler, not the substrate.

---

## Reading a number safely

**0. Measure an ENSEMBLE, and prove your arms differ.** The calculus is a
claim about ensembles: `G(n,p)` is one draw and no result may depend on which
draw you got. A single-seed before/after is not a measurement.

    from neural_assemblies.diagnostics import compare_arms, paired_delta
    arms = compare_arms({"treatment": run_a, "control": run_b}, seeds=range(42, 52))
    print(paired_delta(arms["treatment"], arms["control"]))

`compare_arms` RAISES if two arms return identical values on every seed --
that is a dead pathway or a flag that never reached the engine, not a negative
result, and it is the most common way this repo produces a confident wrong
answer. `Ensemble.beats(x)` tests the confidence bound, never the mean.

**1. Arbitrate.** `diagnostics.arbitrate(build, measure)` runs a protocol on
`explicit` / `materialized` / `sampled` through **one** extractor and returns
all three. Use it before attributing an anomaly to the model.

**2. Do not compare quantities that merely share a name.** Four bugs so far:

| name | meaning A | meaning B |
| --- | --- | --- |
| `firing_stimuli` | stimuli firing now (tail left at zero) | stimuli connected |
| `Area.w` | num-ever-fired | `len(winners)`, after any winners assignment |
| `winners` | `Area`: compact indices `0..w-1` | `Assembly`: neuron IDs `0..n-1` |
| "support" | distinct winners over a run | `area.w` |
| per-fiber beta | `Area.beta_by_area` (dense path) | `AreaState.beta_by_source` (engine) |

Use `get_num_ever_fired()` / `active_count`, and `Assembly.neuron_ids`.

For beta, use **`Brain.update_plasticity(from, to, beta)`** -- it writes both.
`Area.update_beta_by_area` wrote only the first and so changed nothing on the
sparse/torch/cuda engines; it now raises. An `Area` cannot forward to the engine
because it holds no engine handle by design (it is pickled and deep-copied
constantly -- the same reason `_xp` stores a name, not a module).

**3. Check the floor and the degenerate case.** A metric is evidence only if a
plausible broken state fails it. A sealed area scores 1.000 on stability; a
churning one scores chance-level "separation". For any distinctness claim,
assert the assembly is *settled* too.

**4. Read a golden's tier.** `provenance.golden(value, tier, source)` puts it
in the failure message. Tier A is from a paper; tier C was recorded by running
this repo and may be pinning an artifact. As of 2026-07-31, 1 of 23 is tier A.

**5. Scale down by `k*p`, not `p`.** The expected afferent count governs the
dynamics. Copying the paper's `p` to a smaller `n` gives `k*p ~ 0.1`, and
k-WTA falls through to its index tie-break -- every measurement then reports
allocation order.

---

## Sharp edges

* **Zero drive returns early**, preserving the previous assembly. The symptom
  of a dead fiber is a *stale* assembly, not an empty one.
  `ASSEMBLIES_STRICT_DRIVE=1` warns.
* **Materialization is lazy**, so a protocol that drives an area from an
  arbitrary `k`-subset of `n` mostly names neurons that do not exist.
  `materialize_area` first.
* **Physical capacity is amortised and is NOT the logical size.** Buffers grow
  by doubling; every consumer slices by the logical `w` (`min(src.w,
  w.shape[0])`). Reading `weights.shape` as "how big is this fiber" is another
  instance of the same-name trap. Growth is clamped to `n` -- it was not, and
  an `n=10000` area held a (15682, 15682) matrix.
* **Dense `Connectome` init is draw-order dependent** (task #81): declaring an
  extra stimulus rewires every connectome built after it. Affects the explicit
  engine. Build comparison arms by `deepcopy`, not by rebuilding.
* **`read_only()`** blocks recruitment and RNG advance -- use it for probes, or
  measuring changes what it measures.
* **Gating cannot switch off self-recurrence.** `InhibitionState.project_map`
  derives `X -> X` from the state of the fibers INTO `X`, never from
  `fiber_states[X][X]` -- faithful to `parser.py:413`. So an area that is
  receiving is also self-sustaining, and the only way to stop that is to close
  every fiber into it. Since self-recurrence during training is this repo's
  measured collapse channel for shared areas, "live but non-plastic" is
  expressible **only** as a per-fiber beta, not as a gate.
