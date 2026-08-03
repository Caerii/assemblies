# Where things actually live

Salvaged from six `# STATUS: planned` placeholder packages -- `area_management`,
`connectome_management`, `experiments`, `input_processing`, `interfaces`,
`learning` -- which existed only to hold this map. They had zero importers, and
every path they pointed at used a `src.` prefix that has not existed since the
package was renamed to `neural_assemblies`, so the map they carried was correct
in substance and wrong in every address.

Kept as a document because the information is worth having and a package is the
wrong container for it: an importable stub that nothing imports is indexed by
tooling, walked by the fingerprint, and read as a real module.

## Area lifecycle

| concern | lives in |
|---|---|
| `Area` descriptor | `core/area.py` |
| creation and orchestration | `core/brain.py` (`Brain.add_area`) |
| per-area compute state | `core/numpy_engine/_state.py`, `core/cuda_engine.py` |
| how many neurons exist | `ComputeEngine.materialized_count` -- NOT `area.w`, which means two different things ([[same-name-two-meanings]]) |

## Connectome

| concern | lives in |
|---|---|
| data structure | `core/connectome.py` |
| sparse / explicit logic | `core/numpy_engine/_sparse.py`, `_explicit.py` |
| exact (computed, not sampled) drive | `core/numpy_engine/_exact.py` |
| implicit hash-based connectivity | `core/cuda_engine.py` |
| initial weights, content-addressed | `core/numpy_engine/_seeding.py` |
| how wide a fiber is | `ComputeEngine.fiber_extent` |

## Input

| concern | lives in |
|---|---|
| `Stimulus` descriptor | `core/stimulus.py` |
| stimulus projection | `core/brain.py` (`Brain.add_stimulus`) |
| image activation | `compute/image_activation.py` |

## Learning

| concern | lives in |
|---|---|
| Hebbian rules | `compute/plasticity.py` |
| application during projection | `core/numpy_engine/_sparse.py`, `_exact.py` |
| CUDA update kernel | `core/cuda_engine.py` |
| k-WTA pricing law | `core/_pricing.py` -- ONE definition, deliberately; it was implemented twice and one copy missed two fixes |

## Interfaces

| concern | lives in |
|---|---|
| `ComputeEngine` ABC, `create_engine` | `core/engine.py` |
| backend selection, `get_xp` | `core/backend.py` |
| cross-language protocol export | `ir/protocol.py`, `parity/` |

## Experiments

| concern | lives in |
|---|---|
| standalone scripts | `research/experiments/` |
| shared statistical helpers | `diagnostics.py` -- `ensemble`, `paired_delta`, `compare_arms`, `separation` |
| parser training cache | `assembly_calculus/emergent/evaluation/sweep.py` |

## Measurement discipline

Not in the original stubs, and the part most often got wrong. Before trusting a
number, see `diagnostics.py`:

| question | tool |
|---|---|
| is this fiber carrying anything? | `fiber_census(driven=...)` |
| did my probe change what it measured? | `Brain.probe()`, `NEURAL_ASSEMBLIES_STRICT_PROBES=1` |
| how well do two conditions separate? | `separation` (AUC), NOT Cohen's d |
| is this constant still meaningful? | `core/metric.py` -- `Metric`, `Threshold.audit` |
| enough seeds? | `ensemble` refuses fewer than three |
