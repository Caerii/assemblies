# Scientific Validation Experiments

This directory contains systematic experiments to validate the Assembly Calculus framework.

## Quick Start

```bash
# Run all primitive tests (quick)
uv run python research/experiments/primitives/run_all.py --quick

# Run individual tests
uv run python research/experiments/primitives/test_projection.py --quick
uv run python research/experiments/primitives/test_association.py --quick
uv run python research/experiments/primitives/test_merge.py --quick

# Run stability tests
uv run python research/experiments/stability/test_phase_diagram.py --quick
```

## Experiment Categories

### 1. Primitives (`primitives/`)

Validates the three core Assembly Calculus operations:

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Projection** | Does projection reliably create stable assemblies? | ✅ Validated |
| **Association** | Does co-activation increase assembly overlap? | ✅ Validated |
| **Merge** | Does simultaneous projection create composite assemblies? | ✅ Validated |

### 2. Stability (`stability/`)

Investigates assembly stability and phase transitions:

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Phase Diagram** | Is there a critical sparsity for assembly stability? | ✅ Implemented |
| **Scaling Laws** | How does convergence time scale with network size? | 🔄 Planned |
| **Noise Robustness** | How much noise can assemblies tolerate? | 🔄 Planned |

### 3. Information Theory (`information_theory/`)

Analyzes coding properties of assembly representations:

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Coding Capacity** | How much information can assemblies encode? | 🔄 Planned |
| **Error Correction** | How robust is assembly coding to noise? | 🔄 Planned |

### 4. Biological Validation (`biological_validation/`)

Validates against real neural data:

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Sparsity Comparison** | Do simulated sparsity levels match biology? | 🔄 Planned |
| **Assembly Detection** | Can we find assembly-like structures in real data? | 🔄 Planned |

## Results

Results are saved to `research/results/` in JSON format with:
- Experiment parameters
- Summary metrics
- Raw trial data
- Timestamps for reproducibility

## Running Full Experiments

For publication-quality results, run full parameter sweeps:

```bash
# Full primitive validation (takes ~30 minutes)
uv run python research/experiments/primitives/run_all.py --full

# Full phase diagram (takes ~1 hour)
uv run python research/experiments/stability/test_phase_diagram.py
```

## Key Findings (Preliminary)

### Primitive Validation
- **Projection**: 100% convergence rate across all tested configurations
- **Association**: Overlap increases consistently with co-activation
- **Merge**: Composite assemblies capture both parent concepts (88% quality)

### Stability
- Assemblies are stable across wide range of sparsity levels (0.01 - 0.1)
- Convergence typically occurs within 5-10 projection rounds
- Higher beta (plasticity) leads to faster convergence

## Adding New Experiments

1. Create new file in appropriate category directory
2. Inherit from `ExperimentBase` in `base.py`
3. Implement `run()` method returning `ExperimentResult`
4. Add to category's `__init__.py`
5. Update this README

See existing experiments for examples.

