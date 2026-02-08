# Assemblies

A computational neuroscience framework implementing the Assembly Calculus — a model of brain computation through neural assemblies — with extensions for language learning, image classification, GPU acceleration, and Turing machine simulation.

Based on:
- Papadimitriou et al., "Brain Computation by Assemblies of Neurons", *PNAS* 2020
- Mitropolsky et al., "The Architecture of a Biologically Plausible Language Organ", 2023

## Quick Start

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# With GPU support (CUDA 12.x)
pip install -e ".[gpu]"

# Run tests
uv run python -m pytest src/tests/ -q
```

## Overview

Neural assemblies are groups of neurons that fire together to represent concepts. The Assembly Calculus provides three core operations on these assemblies:

- **Projection**: Copy an assembly into a downstream brain area
- **Association**: Increase overlap between assemblies to link concepts
- **Merge**: Combine two assemblies into a new one representing their conjunction

Brain delegates all computation to a **ComputeEngine** — swap between CPU and GPU
backends with a single parameter:

```
Brain(p=0.05, engine="auto")    # auto-detect best backend
Brain(p=0.05, engine="numpy_sparse")    # CPU, scales to large n
Brain(p=0.05, engine="cuda_implicit")   # GPU, 40x speedup at n=100k
```

## Usage

### Core API

```python
from src.core.brain import Brain

b = Brain(p=0.05, engine="numpy_sparse")
b.add_stimulus("stim", size=100)
b.add_area("A", n=10000, k=100, beta=0.05)

# Project stimulus into area A
b.project({"stim": ["A"]}, {})

# Recurrent projection to stabilize assembly
for _ in range(9):
    b.project({}, {"A": ["A"]})
```

### Classic API (backward-compatible)

```python
from brain import Brain  # thin shim re-exporting src.core.brain

b = Brain(p=0.05)
b.add_stimulus("stim", k=100)
b.add_area("A", n=10000, k=100, beta=0.05)
b.project({"stim": ["A"]}, {})
```

### Engine API (direct access)

```python
from src.core.engine import create_engine

engine = create_engine("numpy_sparse", p=0.05, seed=42)
engine.add_area("A", n=10000, k=100, beta=0.05)
engine.add_stimulus("s", size=100)
result = engine.project_into("A", from_stimuli=["s"], from_areas=[])
print(result.winners, result.num_ever_fired)
```

### NEMO Language Learning

```python
from src.nemo.core import Brain, BrainParams
from src.nemo.language import LanguageLearner, SentenceGenerator

learner = LanguageLearner()
learner.hear_sentence(["dog", "chases", "cat"])
learner.hear_sentence(["cat", "sees", "dog"])

generator = SentenceGenerator(learner)
sentence = generator.generate_sentence()
```

### Running Simulations

```python
from src.simulation.projection_simulator import project_sim
weights = project_sim(n=100000, k=317, p=0.01, beta=0.05, t=50)

from src.simulation.pattern_completion import pattern_com
weights, winners = pattern_com(alpha=0.5, comp_iter=5)

from src.simulation.merge_simulator import merge_sim
a_w, b_w, c_w = merge_sim(n=100000, k=317, p=0.01, beta=0.05)
```

### Parser (English / Russian)

```python
from parser import parse
parse("cats chase mice", language="English")
```

## Project Structure

```
assemblies/
|-- brain.py                # Backward-compatible shim (re-exports src.core.brain)
|-- brain_util.py           # Overlap computation, save/load utilities
|-- learner.py              # Word acquisition and syntax learning experiments
|-- parser.py               # Sentence parsing via assembly operations
|-- simulations.py          # Simulation runners and experiment harness
|-- image_learner.py        # CIFAR-10 classification via assemblies
|
|-- src/                    # Modular package (pip install -e .)
|   |-- core/               # Brain, Area, Stimulus, Connectome, ComputeEngine
|   |   |-- engine.py       # ComputeEngine ABC + registry
|   |   |-- numpy_engine.py # NumpySparse + NumpyExplicit engines
|   |   |-- cuda_engine.py  # CudaImplicit engine (GPU)
|   |   |-- backend.py      # NumPy/CuPy switching layer
|   |   `-- kernels/        # CUDA kernels (implicit, batched, v2)
|   |-- compute/            # Statistics, plasticity, winner selection, projections
|   |-- simulation/         # Projection, merge, pattern completion, association sims
|   |-- language/           # Grammar rules, language areas, parser, readout
|   |-- lexicon/            # 5000+ word lexicon, curriculum, GPU learners
|   |-- nemo/               # NEMO v2: learned grammar, language generation (GPU)
|   |-- constants/          # Default parameters
|   |-- utils/              # Math utilities
|   `-- tests/              # 17 unit and integration test files
|
|-- research/               # Experiments, NEMO runners, results, plans
|-- cpp/                    # Custom CUDA kernels (.cu) and build scripts
`-- pyproject.toml          # Package configuration
```

## Tests

```bash
# Fast core tests
uv run python -m pytest src/tests/test_brain.py src/tests/test_engine_parity.py -v

# All unit tests (excludes slow simulation integration)
uv run python -m pytest src/tests/ -q --ignore=src/tests/test_simulation_integration.py

# Full suite
uv run python -m pytest src/tests/ -q
```

## Dependencies

**Required**: numpy, scipy, matplotlib, pptree, tqdm, pandas, seaborn

**Optional GPU**: cupy-cuda12x, torch

**Dev**: pytest, pytest-cov, ruff

## References

- Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., & Maass, W. (2020). Brain Computation by Assemblies of Neurons. *PNAS*, 117(25), 14464-14472. [doi:10.1073/pnas.2001893117](https://www.pnas.org/doi/full/10.1073/pnas.2001893117)
- Mitropolsky, D., & Papadimitriou, C. H. (2023). The Architecture of a Biologically Plausible Language Organ. [arXiv:2306.15364](https://arxiv.org/abs/2306.15364)

## License

MIT License. See [LICENSE](LICENSE).
