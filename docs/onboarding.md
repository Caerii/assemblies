# Onboarding: how to work on this repository

For a new collaborator, human or model, who has to produce work here that
survives review. It is a reading order, the rules that every result has
to meet, the open problems worth a week, and the process constraints that
have cost us results when broken.

## Read in this order

First run [the CPU investigation](../examples/01_basic_assembly_calculus.py):
`uv run python examples/01_basic_assembly_calculus.py`. Follow the fixed graph,
explicit recurrent training, partial-cue readout, and learning-disabled control.
Explain why the probe uses `read_only()` and why a demonstration is not adopted
evidence. Then follow the reading order below.

1. [../README.md](../README.md). The model in four terms (area, assembly,
   Hebbian plasticity, refraction) and the headline results with their
   figures. Everything else assumes these terms.
2. [register.md](register.md). Every adopted result, with an ID, a
   status (PROVED, MEASURED, EXTENSION), its evidence and its caveat.
   Cite results by ID; `neural_assemblies.theory.cite("REFRACTION-ANTI-MERGING")`
   resolves one and refuses an unknown ID.
3. [../research/notes/README.md](../research/notes/README.md). The map of
   the registered lines: the symbols and terms, one Result and Evidence
   block per line, the figures with how to read them, and the working
   rules at the end.
4. Two registrations, read end to end, to see the discipline in use:
   [../research/notes/memory/PREREG_refraction_memory.md](../research/notes/memory/PREREG_refraction_memory.md)
   (six amendments, a scorecard, failed bars kept) and
   [../research/notes/sequence/PREREG_temporal_memory.md](../research/notes/sequence/PREREG_temporal_memory.md)
   (a prediction that failed, a window found, a mechanism half right).
5. [../research/notes/sequence/PREREG_sampler_audit.md](../research/notes/sequence/PREREG_sampler_audit.md).
   Why any sequence number measured on the lazily drawn numpy engine is
   void until re-run materialized or on the hashed substrate.
6. [gpu_scale_design.md](gpu_scale_design.md) and
   [../research/notes/sequence/DESIGN_sequence_port.md](../research/notes/sequence/DESIGN_sequence_port.md).
   The hashed substrate: a connectome regenerated from a hash, brains
   batched in one launch, and the parity gates every unit passes.
7. [../research/plans/PAPERS.md](../research/plans/PAPERS.md) and
   [../research/plans/SOUNDNESS_PROGRAM.md](../research/plans/SOUNDNESS_PROGRAM.md).
   What the results are for, in what order, and the standing rules for
   measurement.
8. [architecture.md](architecture.md), [api.md](api.md) and
   [supported_surfaces.md](supported_surfaces.md) when you need the code.

## What every result has to meet

- **Bars before data.** Hypotheses and pass conditions are written into a
  `PREREG_*.md` before the run. A change made before running is an
  amendment; a change made after seeing data is post hoc and is never
  adopted on its own.
- **At least three seeds; twenty on the hashed substrate.** Per-seed
  distributions with an interval, never a bare mean. The harness's
  `ensemble_from_values` refuses fewer than three.
- **Failed bars are committed** with their numbers. Four of six predictions
  registered in one week failed, and each failure named the mechanism.
- **A perfect score falsifies the measurement** until a sweep shows the
  number can move. Sealed areas, dead fibers and frozen winners all read
  1.000.
- **Name the engine.** The numpy engine draws connectomes lazily unless an
  area is materialized; the hashed substrate is exact. A number's engine
  is part of the number.
- **Smoke runs check the API, not the science.** A run with fewer seeds or
  a smaller grid than registered produces void numbers; tag it so.
- **Adopted results go into `neural_assemblies/theory.py`**, rendered to
  `docs/register.md`; a test fails if the rendering is stale.

## Open problems worth a collaborator's week

Each of these has data in hand and a registration to extend.

1. **The temporal carry's decay law.** The corrected position-specific run finds
   that plain copy-state conjunction carries subject number through the first
   distractor (contrast 0.048) but reaches zero at the second (0.004). Predicted-win
   preserves 0.203 then 0.167, while a state-blind negative is zero; all registered
   bars pass. The old pooled 0.11/0.22 values remain void. Extend the registered
   instrument to gaps 3 through 6 before fitting a decay law. A held-out readout at
   the first distractor is secondary: the gap-2 g=0 prediction site has no hidden
   representation for a readout to recover. See `PREREG_temporal_positions.md` and
   `SEQ-TEMPORAL-CARRY`.
2. **The clip window.** Every organ result holds between about 20 and 28
   presentations and collapses past 31, because the weight clip lets a
   synapse potentiated once per presentation relocate the arc. A weight
   rule that keeps the organ in its window under 200 presentations is the
   primitive that long sequences need. `PREREG_s5_cliff_anatomy.md`,
   Addenda 5 to 8; `PREREG_substrate_c_homeostasis.md` for what did not
   work.
3. **A derivation for the square capacity law.** The refracted ceiling is
   about 0.40 (n/k)^2 in regime, with exponents drifting 2.1 to 1.8 across
   the grid. The Willshaw and sparse-Hopfield arguments give the form;
   which of their assumptions the refracted area meets, and what fixes the
   constant, is open. `REFRACTION-ANTI-MERGING`, `PREREG_refraction_memory.md`.
4. **Baselines for the sequence line.** Bigram and oracle only. A trigram
   and a small recurrent network at matched step counts on the chain
   corpus, plus gaps 4 to 6 for the carry's decay law, decide whether the
   temporal memory is worth a paper or a paragraph.
5. **Engine provenance in the register.** Only four of the fifteen
   MEASURED entries name their engine. Add an `engine` field to `Result`
   and fill it from each registration.

## Process constraints that have cost us results

- **One working tree per session.** Two sessions editing the same
  checkout put reverted engine files into a commit and voided a run. If
  you must share a machine, use a worktree pinned at a commit and import
  from it (the editable install resolves to the main checkout, so drop
  that finder from `sys.meta_path` and put the worktree first).
- **One GPU job at a time.** Studies contend for the card and the memory
  limit; a smoke run scheduled beside a study slowed both and had to be
  killed by PID.
- **Commit by explicit path** when anything else may be staged;
  `git add -A` swept another session's files once.
- **Commits go to `dev`; `master` is a fast-forward of it.**
- **Tag every run** (`--tag`) so a results file is never overwritten by
  the next run of the same script; two were, and had to be restored from
  git.
- **CHILDES transcripts are cite-only.** Never committed, never fetched
  around an authorization wall.
- **The compiled CUDA extension cannot be rebuilt while a process holds
  it.** Finish or kill the GPU job first.
- **Do not edit engine files while a run imports them**, and do not scale
  or column-normalize a refracted area (`REFRACTION-CANCELS-CONVERGENCE`).

## Running things

The shared experiment entry and its current migration limits are documented in
[research_workflow.md](research_workflow.md). For the maintained CUDA extension,
use [cuda_toolchain.md](cuda_toolchain.md); it includes a checker that does not
compile or use the GPU.

```bash
uv sync --extra gpu
uv run pytest neural_assemblies/tests -q -m "not slow"
```

GPU tests and studies need the Visual Studio developer shell on Windows
(`vcvars64.bat`) so the fused extension can compile on first import; see
[../research/experiments/README.md](../research/experiments/README.md)
for the active scripts by line and their typical runs, and
[../research/results/README.md](../research/results/README.md) for where
their evidence lands.
