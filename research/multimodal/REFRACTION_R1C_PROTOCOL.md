# R1c: Paired masked-vs-net refraction protocol

**Status:** Registered before native execution. The implementation and tests
are committed, but results are not yet measured on this branch.
**Date:** 2026-11-23.
**Owner:** `research/multimodal/refraction_paired.py`.

## Prior evidence and why this experiment exists

`research/notes/memory/PREREG_refraction_memory.md` documents earlier
capacity studies: the measured effect is ANTI-MERGING, not orthogonalization.
The existing `research/experiments/refraction_memory_numpy.py` uses ten
graded stimulus components to avoid a zero-or-size input degeneracy.
It also explicitly distinguishes **masked** and **net** recall.

Our R1b native experiment had one cue per item and unmasked readout. It is
NOT an admissible test of that refraction result. R1c fixes these two
confounds by adopting the existing experiment's graded input and reading the
same trained assembly store both ways.

## Experimental design

For each seed: construct the same n=2000,k=60,p=.5,beta=.1
`numpy_sparse` Brain, materialize its memory area, disable column scaling,
enable norm_init, use T=8 joint stimulus+recurrent rounds per item.
Create 10 stimulus components each of size 6. Train control with s=0;
train refracted arm with s=0.5*beta. Store complete winner snapshots.

At M = 8,16,32, select the *same* sample of stored indices for masked
and net probes. For each sampled item: inject a half-size cue from the
original winner set, run 8 frozen recurrent rounds, rank its recovered
winners against all stored winner sets. Score top-1 with deterministic
earliest-index ties; also record self-overlap. Both probes run inside
`brain.read_only(seed=...)` plus `brain.frozen()`; restore the masked
flag in `finally`. This is a paired readout contrast, not a training-arm
comparison or arbitrary composition test.

## Registered metrics and decision gates

1. **Mask-vs-net effect:** per-seed, within-trained-brain top-1 difference
   `masked.rank1 - net.rank1`; keep both raw values and input indices.
2. **Control:** compute the same mask/net difference on un-refracted
   brains. A large difference in this control would indicate a
   readout-switch implementation artifact, not a refraction benefit.
3. **Isolation:** mask state resets even on exceptions; observation must
   not mutate prior synaptic weights, activity or learning history. Unit
   tests check API scoping, but native state invariance is still required.
4. **Reproduction:** compare R1c numbers to the earlier native
   `refraction_memory_numpy.py` at matched settings before interpreting.
5. **Negative cases:** report net > masked, ties, and insufficient
   convergence. No suppression of unfavorable seeds.

The small grid tests functionality, NOT the full capacity law. No
significance or threshold is preregistered for three-seed smoke
results; expand to at least 20 seeds and 8..512 items only after
correctness and matched-protocol gates pass.

## Commands

```sh
uv sync --no-group gpu
uv run python -m pytest -q research/multimodal/test_refraction_paired.py
uv run python research/multimodal/refraction_paired.py \
  --n 2000 --k 60 --loads 8 16 32 --seeds 42 43 44 \
  --output /tmp/refraction-r1c.json

# Existing repository experiment, comparison protocol:
uv run python -m research.runner refraction-memory-numpy \
  --tag R1C-COMPARATOR-UNIQUE --seeds 42 43 44 --arm both
```

IMPORTANT differences to audit before comparison:
the new R1c uses `brain.read_only` and `brain.frozen` for each
probe while the existing experiment directly calls the engine with
`plasticity_enabled=False`; R1c sorts each stored assembly before
taking the half cue, while the existing experiment takes the first
half of the originally returned winner order. These choices can shift
scores even when underlying weights match. **No equivalence is claimed.**
Do not label a discrepancy a model failure until these are reconciled.

## Next experiment (R1d)

After R1b/R1c native execution, compare refraction and explicit
binding schemes on unseen *role-object* combinations, with
same-budget episodic memory and symbolic key-value baselines.
Do not use input-overlap alone as proof of compositional transfer.
