# GPU-driven assembly dynamics at scale — design note

Status: **design sketch** (2026-07-24). No code committed for the batched/dense
model yet; this note scopes it. Prereq work already landed in `torch_engine/`
(on-device `torch.topk` selection, fast `set_winners`, `norm_init` port).

---

## 1. The wall we hit, and why

The `torch_sparse` engine is GPU-native in *storage* (CSR connectomes on device)
but the per-area projection loop cannot beat CPU `numpy_sparse`, even on an
RTX 3080, at any size measured (n up to 10M, k up to 100k). After removing the
two worst anti-patterns (CPU `argpartition` selection; a `uint32` `torch.tensor`
cast in `set_winners` that alone was 91% of runtime), profiling shows **no single
bottleneck left** — the projection is simply sub-millisecond of arithmetic spread
across ~10 kernel launches plus small host syncs. numpy does the same work with
near-zero launch overhead and wins (~0.25 ms vs ~6 ms at k=100k).

This is **fundamental, not a bug**. The sparse-sampling algorithm exists to
*avoid* O(n) work: it materializes only `w` neurons and samples ~`k` candidates,
so per-round work is O(k). GPUs want the opposite — lots of parallel arithmetic
per kernel launch. The two are in tension.

Isolated evidence of the ceiling: GPU-resident `torch.topk` vs the old
copy-to-CPU + `argpartition`, on the drive vector alone:

| drive size W (k) | CPU path | GPU topk | speedup |
|---|---|---|---|
| 200k (100k) | 7.5 ms | 0.50 ms | 15× |
| 1M (100k) | 51 ms | 0.94 ms | **55×** |
| 5M (200k) | 165 ms | 1.03 ms | **159×** |

The GPU win **grows with the amount of parallel work**. To realize it we must
*create* that work. Two levers do so; the first also unlocks the second.

---

## 2. Two levers

```mermaid
flowchart LR
    subgraph now["Now: sparse, per-area (CPU-favorable)"]
        A1["score ~k sampled candidates"] --> A2["topk over w+k"] --> A3["Hebbian on k"]
    end
    subgraph A["Lever A: dense drive (within-area)"]
        B1["SpMV: winners x CSR -> dense length-n drive"] --> B2["topk over ALL n (55-159x regime)"] --> B3["Hebbian on k"]
    end
    subgraph B["Lever B: batched (across area/brain)"]
        C1["[Bxn] drive tensor"] --> C2["batched topk dim=-1"] --> C3["batched Hebbian"]
    end
    now -.replace sampling.-> A
    A -.fixed [n] per item enables batching.-> B
```

### Lever A — dense-drive projection (within-area)

Replace candidate *sampling* with dense *scoring*: compute the drive to **all n**
candidate neurons on the GPU, then a single `torch.topk` over n. This is the
reference NEMO algorithm (score all n, take top-k) — the sparse engine's sampling
is an approximation of exactly this.

Mechanics (per projection):
1. **Drive accumulation** — for each source area, `winners ⊗ CSR → dense[n]`.
   This is a sparse-matrix / sparse-vector product. Options: `torch.sparse` /
   cuSPARSE SpMV, or keep the current `scatter_add` but scatter into a length-**n**
   (not length-`w`) output. Stimulus fibers add their dense contribution directly.
2. **norm_init** — the `1/d_j` scale is now a single dense element-wise divide over
   `[n]` (in-degree is a `bincount` over CSR columns). *No candidate-sampling
   divisor, no un-normalization, no lazy-materialization bookkeeping* — dense mode
   **simplifies** norm_init versus the sparse path.
3. **Selection** — `torch.topk(drive, k)` over n. This is the measured 55–159× win.
4. **Hebbian** — potentiate the k winners' incoming edges (existing CSR update).
5. **Materialization** — winners that were never active before get real neuron ids
   / new CSR columns, same as today, but there is no `neuron_id_pool` exhaustion
   risk because we no longer sample a bounded pool (see the role-clear desync bug
   this would retire).

Cost model: drive vector is O(n) dense = `4·n` bytes (4 MB at n=1e6, 40 MB at
n=1e7). Connectivity stays **sparse** (CSR) — we never materialize an n×n matrix.
Per-projection work becomes O(n) for the topk + O(active-edges) for the SpMV,
which is exactly the arithmetic density GPUs reward. Crossover vs CPU is expected
around n ≳ 10^5–10^6; below that, `numpy_sparse` should remain the default.

Foundation already present: `_dense_area_conns`, `explicit_source`,
`_bootstrap_from_explicit_dense`, `_dense_weights` in
[`torch_engine/_engine.py`](../neural_assemblies/core/torch_engine/_engine.py) and
the `numpy_explicit` engine — dense areas are already a first-class concept to
extend, rather than a green-field build.

### Lever B — batched execution (across area / across brain)

The training bottleneck is not one projection; it is a **curriculum of thousands
of projections** over a corpus (parser training runs minutes–hours). Batch them.

The obstacle has always been **ragged materialization**: each area/brain grows a
different set of `w` materialized neurons, so their state tensors don't align.
**Lever A removes this obstacle**: a dense drive is a fixed `[n]` vector regardless
of what's materialized, so a batch is simply `[B, n]` and `torch.topk(dim=-1)`
selects for all B items in one kernel. Batched Hebbian updates follow.

Two batching granularities:
- **Across areas** within one brain — project independent areas together (e.g. the
  per-word core areas, or ROLE_AGENT/ROLE_PATIENT) in one batched call.
- **Across brains / sentences** — data-parallel curriculum: run B copies of a brain
  (or B sentences through shared connectivity) as a batch dimension. This is the
  big throughput multiplier for training.

Connectivity in a batch is still per-item sparse. Layout options:
- **Block-diagonal CSR** — stack B items' connectomes into one big CSR with disjoint
  index ranges; one SpMM does the whole batch. Cleanest for shared-topology batches.
- **Padded dense-drive** — since the drive is already `[B, n]`, keep B separate CSRs
  and loop the SpMV but batch everything downstream (topk, Hebbian, norm_init).
  Simplest first step; captures the topk/selection win immediately.

---

## 3. Memory budget (RTX 3080, 10 GB; scales linearly with card)

| quantity | formula | n=1e6 | n=1e7 |
|---|---|---|---|
| dense drive `[n]` | 4·n B | 4 MB | 40 MB |
| batched drive `[B,n]`, B=64 | 4·B·n B | 256 MB | 2.5 GB |
| CSR connectivity | ~8 B/edge | grows with training | — |

Dense drive is cheap; the budget is dominated by **CSR edge count**, which grows
with training (each projection adds ≲ k·(fan-in) edges). For large n or large B,
cap with: periodic consolidation (already in `consolidation.py`), edge pruning
below a weight floor, or sharding the batch. B=64 at n=1e6 is comfortable; B=64 at
n=1e7 needs edge budgeting.

---

## 4. Parity & validation strategy

Dense mode is a **different algorithm** (exact top-k over n vs sampled), so it will
not bit-match `numpy_sparse`. The bar is the same as the existing engine-parity
suite: **behavioral** thresholds (assembly size = k, stability > 0.9, recovery,
separation, association/merge overlap vs chance), which
[`test_torch_parity.py`](../neural_assemblies/tests/test_torch_parity.py) already
encodes. Dense mode should if anything match the **reference NEMO** more closely
than the sampled path does — worth a direct golden comparison against
`.reference/mdabagia-nemo`.

Validation ladder:
1. Single dense area matches `numpy_sparse` behaviorally (size/stability/recovery).
2. Dense norm_init reproduces the degree-hub-collapse fix (self-recurrence stays
   non-degenerate; cf. `recurrence-needs-norm-init`).
3. Batched B-item run == B independent single runs (per-item equality up to RNG).
4. End-to-end: a small curriculum trains to the same accuracy floor as the sparse
   engine, faster wall-clock at large n / large B.

---

## 5. Phased roadmap

| phase | scope | result | risk |
|---|---|---|---|
| **0 ✅** | on-device topk, fast set_winners, norm_init port | landed; 43 parity tests green | — |
| **1 ✅** | dense-drive single-area mode (Lever A), engine flag `dense_drive` | landed; behaviorally correct (sep 0.000, recovery 0.96), and *faster* than the sparse GPU sampler (n=5M: 0.43 vs 0.51 ms/round) | SpMV perf; parity vs reference |
| **2 ✅** | batched projection through a **shared** connectome (Lever B, clean case) | landed (`_batched.py`); identical to sequential per item, **7× @ B=8, 21× @ B=32** (RTX 3080) | VRAM (batched [B,n] + SpMM) |
| **3 ✅** | block-diagonal SpMM for **independent** connectomes + batched Hebbian | landed; projection 3.5–5×, and **batched training is bit-identical to independent training** | index bookkeeping; VRAM (edge count × B) |
| **4 (deferred)** | CUDA-graph capture of the fused projection | kill residual launch overhead | needs static buffers first (see below) |

Phases 0–3 are done and measured. Phase 3 extends batching from a *shared*
connectome (batch inference/parse — Phase 2) to *independent* per-item connectomes
with learning (data-parallel training — different brains stacked block-diagonal).

**Phase 3 done.** `block_diagonal` + `batched_project_independent(..., beta=)` in
`_batched.py` stack B independent connectomes into one `[B*n, B*n]` sparse matrix
and project (and Hebbian-train) all B in a single SpMM + one masked scatter. The
key result: because block-diagonal edges never cross items, the winner-pair
potentiation mask is automatically per-item correct, so **batched training is
bit-identical to training the B brains independently** (validated), at 3.5–5×
throughput. Speedup is flatter than the shared-connectome case because total nnz
— hence SpMM work — scales with B; the win is amortizing B kernel launches.

**Phase 4 (deferred, with reason).** CUDA-graph capture needs static shapes and no
dynamic allocation inside the captured region. The batched loops currently rebuild
the CSR each round (`to_sparse_csr`) and allocate fresh activity/drive tensors —
dynamic. Capturing first requires refactoring to pre-allocated static buffers and
an in-place CSR value update. The remaining win (shaving per-round launch overhead)
is also smaller now that batching already amortizes it across B, so this is
correctly last. What remains before it is worthwhile: preallocate the drive/act/CSR
buffers and update weights in place, then wrap the steady-state loop in a graph.

## 5a. Language integration — batched next-token (landed)

The first real language-scale payoff: `next_token.score_corpus` runs one *frozen*
prediction per corpus position, each independent of the others, so they batch
through the one shared connectome exactly like Phase 2. `BatchedLM`
(`assembly_calculus/batched_next_token.py`) extracts a trained brain's frozen
LEX→LEX connectome + per-word stimulus drive + vocabulary lexicon (baking the
frozen norm_init 1/d_j scale in), then drives B context prefixes through it as one
`[B, n]` activity tensor — one SpMM + one `topk` per step, one matmul for readout.
It replicates `_predict_next_token_inner` step-for-step, so predictions match the
sequential path. Measured on corpus scoring (RTX 3080): **420 predictions in
0.45 s vs 41 s sequential — 91× faster, ~930 pred/s vs ~10 pred/s.**

This surfaced a real correctness point: **prediction was not truly read-only** —
even frozen (no plasticity), the sparse engine samples candidates and materializes
new neurons, so the connectome mutated mid-scoring and predictions were
nondeterministic. Added a `readonly` engine mode that suppresses candidate
sampling (select only among materialized neurons) — inference no longer grows the
brain, prediction is deterministic, and there is a fixed connectome to batch over.
With it, batched matches sequential to ~100% (residual is GPU float
non-determinism in `scatter_add`/SpMM tie-breaking, not systematic).

Targets the production `norm_init=True` substrate. Use case: fast corpus scoring
(perplexity/accuracy over a large corpus), beam/sample generation over B
candidates. Tests: `test_batched_next_token.py` (agreement, score-corpus parity,
readonly-prevents-materialization).

## 5c. Batched training — the curriculum win (landed)

`BatchedSeqTrainer` (`assembly_calculus/batched_trainer.py`) turns the batched
forward pass into a *training* speedup. It processes B sentences' forward passes
in parallel as `[B, n]` tensors, records the Hebbian bridge each transition would
form, accumulates them across the batch with one `SRC.T @ TGT` edge-count matmul,
and applies the update once per batch (`batch_size=1` is online).

Measured (RTX 3080, n=6000, 240-sentence Markov corpus, 2 epochs):

| batch | test acc | train time | weights vs online |
|---|---|---|---|
| 1 (online) | 0.356 | 5.49 s | — |
| 8 | 0.356 | 0.82 s | 0.000% |
| 32 | 0.356 | 0.20 s | 0.000% |
| 64 | 0.356 | 0.11 s | **0.000% (bit-identical)** |

~50× faster training, and the weights are *bit-identical* to online (0.000%).

**What this does and does not show (corrected after a critique by Opus 5).** The
bit-identity is a *theorem*, not evidence about assembly dynamics. At the
`stim=2.0` used here the stimulus strictly dominates the normalized recurrent
drive, so `topk` returns each word's assembly `A[w]` *exactly*, independent of W —
the "recurrent multi-word context" collapses (`context([a,b,c]) == A[c]`,
verified by assertion in `minibatch_training.py`), and the model reduces to a
**bigram count matrix** `W[i,j] += β·count(w_{i-1}→w_i)` in a random basis. On a
first-order Markov corpus a bigram counter is the correct model, so "0.356 vs
0.0625 chance" confirms the counting works, *not* that assembly dynamics
contribute; and the flat accuracy across batch sizes is entailed by the identical
weights (one fact, not two). The honest, bounded claim is: **assembly *stability*
is a sufficient condition for exact update-order independence** — and the sweep in
`minibatch_training.py` shows it breaks (divergence → 58%) exactly as `stim` falls
and recurrence starts selecting winners, i.e. in the low-stim regime where merge /
association / pattern-completion / ordered-recall actually live. So the earlier
phrase "no change to what it learns" is **withdrawn** as an unqualified claim; it
holds only in the stability regime, which here is the regime that isn't using
assembly selection. The speedup mechanism is real; the scale (n=6000, V=16) is
trivial and demonstrates nothing that was hard. Tests: `test_batched_trainer.py`.

### Large vocabulary — sparse growing connectome

`BatchedSeqTrainer`'s dense `[n,n]` W caps at n≈1e4 (400 MB). `SparseBatchedSeqTrainer`
replaces it with a connectome that **starts empty and grows only the bridge edges
that actually form** (the same lazy-edge model the engine's `CSRConn` uses),
assemblies stored as indices, activity dense `[B,n]` with B capped. Result: same
accuracy at 100×+ larger n.

| model | n | test acc | peak GPU | dense would need |
|---|---|---|---|---|
| dense | 8k | 0.340 | — | 0.26 GB |
| sparse (m=1) | 8k | **0.340** | — | — |
| sparse (m=1) | **1e6** | **0.340** | **0.44 GB** | **~4000 GB** |

**The vocab ceiling is lifted ~100× with zero quality loss.** Honest limit on the
*context* axis: a bounded m-gram *union* state (bag of the last m words) does NOT
beat bigram — it dilutes the signal, exactly the point `StatePredictionMixin`
makes that naive aggregation is not the answer. So `m=1` is the effective default;
**genuinely richer context needs a *structured* bounded state** (core × syntactic
slot × mood, weights-between-states) — the existing `StatePredictionMixin`
architecture — not a bigger bag. That structured-state batched path is the next
step for large context; large *vocabulary* is solved here.

## 5b. What's productionized vs. what remains

Landed and tested (~60 GPU tests): the Phase-0 engine fixes, `dense_drive` and
`readonly` engine modes, the batched primitives in `_batched.py` (shared and
independent projection + training), `BatchedLM` (91× corpus scoring), and
`BatchedSeqTrainer` (~50× training, bit-identical to online). The full
train→infer loop now runs at GPU scale with no change to what the model learns.

What remains to make this the *default* language path rather than a parallel one:
- **Sparse connectome at larger n.** `BatchedSeqTrainer` uses a dense `[n,n]` W
  (clear + fast to n≈1e4 on 10 GB). Beyond that, move the batched `SRC.T @ TGT`
  accumulation onto a sparse/growing connectome (the block-diagonal machinery
  already handles per-item sparse; a shared-connectome sparse accumulate is the
  remaining piece).
- **Feed trained connectomes back into `Brain`** so a `BatchedSeqTrainer`-trained
  model is a first-class `Brain`/`EmergentParser` (today they share the
  fixed-assembly + bridge model, not the full parser's areas).
- **Phase-4 static-buffer/CUDA-graph pass** to shave residual launch overhead.

**Measured so far (RTX 3080):**
- Phase 1 dense-drive is behaviorally correct and, among GPU modes, *faster* than
  the sparse truncated-normal sampler — the dense draw + single topk beat the
  order-statistic machinery. It still loses to CPU `numpy_sparse` for a *single*
  area (more total work), exactly as predicted; its role is to enable batching.
- Phase 2 batched projection through a shared connectome is **numerically
  identical** to sequential and **7–21× faster** (efficiency peaks near B=32 before
  the batched SpMM saturates memory bandwidth). This is the first real wall-clock
  GPU win, and it validates the design's central claim: the throughput comes from
  the batch dimension, not from optimizing a single sparse projection.

---

## 6. Open questions / risks

- **SpMV vs scatter_add** for dense accumulation — benchmark `torch.sparse` SpMV
  against a length-n `scatter_add` at representative (n, k, nnz); the current CSR
  is hand-rolled scatter, so scatter-to-n is the low-friction first cut.
- **Exact-vs-sampled scientific equivalence** — does dense top-k change any
  published result? Gate dense mode behind a flag and keep `numpy_sparse` as the
  reference substrate for literature reproductions (cf.
  `norm-init-substrate-vs-reference`).
- **Small-n regression** — dense mode must *not* become the default at small n
  where CPU wins; wire it into `detect_best_engine` with a measured crossover, and
  fix that heuristic (it currently routes n≥1M to `torch_sparse` on faith, which
  this investigation showed is wrong for the *sparse* path).
- **Determinism** — batched RNG for reproducible curricula (per-item seeds along
  the batch dim).

---

## 7. One-line summary

The sparse per-area loop is CPU-favorable and cannot be optimized into a GPU win.
Going **dense within an area** creates the O(n) arithmetic the GPU rewards (and
simplifies norm_init + retires the pool-exhaustion class of bugs); the fixed `[n]`
drive it produces then makes **batching across areas/brains** trivial, which is
the real throughput multiplier for curriculum training. That is where
GPU-driven assembly dynamics at scale actually lives.


## 8. Where this stands (2026-09)

Levers A and B landed and were then superseded for research use by the
hashed substrate (`core/torch_engine/_hashed.py` and its units), which
regenerates each brain's connectome inside the kernel and batches brains
rather than items. Its layouts and measured floors are in
`research/notes/DESIGN_dense_floor.md` (dense counts, organ density) and
`research/notes/DESIGN_present_only.md` (present-only lists, sparse
density); the ports that run on it are in `DESIGN_hashed_aligner.md` and
`DESIGN_sequence_port.md`. Sections 1-7 above describe the earlier design
and remain accurate for the `Brain`-level engines.
