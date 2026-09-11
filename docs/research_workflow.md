# Running an attributable experiment

Start with the [CPU investigation](../examples/01_basic_assembly_calculus.py).
It forms a recurrent assembly on a fixed graph, probes a half cue without
learning, and compares against a learning-disabled control over paired seeds.
It is a demonstration, not adopted evidence.

## The migrated execution path

```powershell
uv run python -m research.runner a1-horizon --help
```

The A1 horizon study is the first migrated experiment. A real run requires the
[GPU toolchain](cuda_toolchain.md) and exclusive use of the GPU:

```powershell
uv run python -m research.runner a1-horizon --tag horizon-20260909-01
```

Its default seeds are 1 through 20. `--seeds` takes explicit identities.
`--smoke --seeds 1 2 3` selects an API check whose observations are marked VOID.
Tags are required for both modes. The original script entry point uses the same
runner, so omitting the tag there is also an error.

The runner creates `research/results/runs/<protocol>/<tag>/` before computation.
An existing directory causes an error before the experiment runs. There is no
overwrite flag. Use a new tag for a rerun; preserve a failed attempt.

- `run.json` records the script, commit, source digest, registration and its
  digest, protocol version, resolved engine, parameters, seeds, input artifacts,
  and smoke/study status.
- `results.json` embeds that record alongside completed observations.
- `failure.json` records an exception or interruption. An abrupt process kill
  may leave only `run.json`; absence of completed results means incomplete.

The record remains UNJUDGED for a study and VOID for smoke. Scientific PASS/FAIL
belongs to evaluated protocol criteria, not to successful program execution.
A1's old sampled comparison is explicitly labeled historical; it is not newly
validated sequence evidence.

Validate an artifact's record and file edges:

```powershell
uv run python -m research.evidence validate research/results/runs/sequence.a1-horizon/horizon-20260909-01/results.json
```

This validates the runner record. It does not prove that the experiment's
mechanism, statistics, or scientific conclusion are sound. Operation contracts,
negative controls and adoption review remain required.

## Shared measurement behavior

`research.harness.study` orchestrates paired arms and counterbalances execution
order. `neural_assemblies.diagnostics` owns seed ensembles and Student-t intervals.
The older `base.summarize` and `_substrate.mean_ci` helpers delegate to that owner.

At least three unique seeds are required; migrated hashed studies require twenty.
Duplicate keys, mismatched key/value lengths, NaN and infinity raise. Paired
ensembles with recorded identities must have identical seed keys in identical
order. Unkeyed legacy ensembles can only be paired positionally with another
unkeyed ensemble; new research should always supply keys.

Missing registered metric names and empty result sets raise. Metrics with no
criteria are UNJUDGED, including in printed output. A study with no judged
metrics has `passed == False`; it is not reported as a failed experiment either.
Named criteria retain their original meaning: `above` checks the mean,
`on_every_seed` checks each seed, and `delta_excludes_zero` checks the paired
interval. Choose the registered criterion explicitly.

The parser's training-source fingerprint now hashes source contents, once per
process. Existing cached fingerprints change and can require retraining. The
training fingerprint is not a complete run record: it does not independently
identify datasets, all environment choices, or the measurement code. Use the
runner record as well, and use an immutable worktree for each process.

## Historical evidence and remaining migration

`uv run python -m research.evidence audit` inventories literal references across
tracked Python and Markdown. It reports candidate orphan results, unresolved
references and preregistrations without resolved result links. Dynamic consumers,
unrun preregistrations and descriptive example paths require human review.

Historical artifacts are not rewritten to manufacture missing provenance.
The full claim-to-registration-to-run graph, remaining experiment migrations,
runtime index wrappers, model configuration and operation contracts are tracked
in the [implementation plan](reviews/whole-codebase/REFACTOR_PLAN.md).
The legacy experiment tree is not yet protected by the new runner globally.
Capacity scaling is also migrated, using protocol version 2:

```powershell
uv run python -m research.runner capacity-scaling --help
```

It requires both `--tag` and `--registration` naming the registration/amendment
for the chosen grid. Defaults use twenty explicit seeds, 42 through 61.
`--nk 2000:40,2000:60` retains both cells. Configuration is immutable and a
second invocation cannot inherit the first invocation's settings.

Version 2 stores a list of cells with arm, n, k and seed identities. Historical
plotters that expect flat `arm/n` keys need an explicit version-2 reader.
GPU errors fail the run with a retained failure record; they are not silently
classified as scientifically unmeasurable cells. No study was run during this
migration. The numerical kernels were not changed.

The former automatic slope verdict is deliberately absent: it pooled varying-k
cells and used a normal approximation over very few grid points. A replacement
fit must specify the estimand, censoring and uncertainty protocol, including
between-brain uncertainty; program completion leaves the study UNJUDGED.

Golden MNIST verification requires the real CSV files. If they are absent,
pytest skips with a dataset-specific explanation and the verification CLI
reports `unavailable`. Synthetic teaching examples are separate from reproducing
a golden measured on MNIST.

Golden verification also fails on missing expected metrics, missing acceptance
criteria, unsupported threshold rules and nonfinite bound measurements. A
`metrics_match_tolerance` now checks the scalar golden measurements it declares.
This can expose historical failures that the old comparator silently skipped;
do not repair them by widening the golden after seeing the output.

The `Research contracts` pull-request workflow runs the fast CPU rejection,
provenance, registration and teaching-example tests. It is a narrow instrument
gate, separate from hardware parity and preregistered study execution.

## Explicit partial-source merge

`merge` accepts zero or two parent stimuli without another mode argument. A call
with exactly one stimulus must state what the current unstimulated parent means:

```python
merge(brain, "COMPOSED", "WORD", "NEXT", stim_b="word-7",
      unstimulated_source_mode="require-fixed")
```

Use `require-fixed` when an outer scope owns the clamp, `fix-current` when merge
should borrow a clamp around the current winners, and `evolving` when the current
unstimulated parent is deliberately allowed to move. The first and third modes
verify the current facade state before projection. Omitting this choice now raises
before mutation; the old call shape did not say which transition system it meant.
