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
python -m pytest tests/ src/tests/ -q
```

## Overview

Neural assemblies are groups of neurons that fire together to represent concepts. The Assembly Calculus provides three core operations on these assemblies:

- **Projection**: Copy an assembly into a downstream brain area
- **Association**: Increase overlap between assemblies to link concepts
- **Merge**: Combine two assemblies into a new one representing their conjunction

This framework provides two Brain implementations (classic and modular), simulation tools for studying assembly dynamics, and higher-level systems for language learning and image classification built on top of the assembly primitives.

## Usage

### Classic API (root-level, matches original paper)

```python
from brain import Brain

b = Brain(p=0.05)
b.add_stimulus("stim", k=100)
b.add_area("A", n=10000, k=100, beta=0.05)

# Project stimulus into area A
b.project({"stim": ["A"]}, {})

# Recurrent projection to stabilize assembly
for _ in range(9):
    b.project({}, {"A": ["A"]})
```

### Modular API (src package)

```python
from src.core.brain import Brain, Area

b = Brain(p=0.01)
b.add_area("V1", n=100000, k=317, beta=0.05)
b.add_area("V2", n=100000, k=317, beta=0.05)
b.add_stimulus("visual", k=317)

b.project({"visual": ["V1"]}, {})
b.project({}, {"V1": ["V2"]})
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

### Lexicon System

```python
from src.lexicon import LexiconManager

lm = LexiconManager()
lm.load_curriculum("basic")
words = lm.get_words_by_category("nouns")
```

### Running Simulations

```python
# Projection dynamics
from src.simulation.projection_simulator import project_sim
weights = project_sim(n=100000, k=317, p=0.01, beta=0.05, t=50)

# Pattern completion
from src.simulation.pattern_completion import pattern_com
weights, winners = pattern_com(alpha=0.5, comp_iter=5)

# Merge simulation
from src.simulation.merge_simulator import merge_sim
a_w, b_w, c_w = merge_sim(n=100000, k=317, p=0.01, beta=0.05)
```

### Parser (English / Russian)

```python
from parser import parse
parse("cats chase mice", language="English")
```

### Image Classification (CIFAR-10)

```bash
python image_learner.py
```

## Project Structure

```
assemblies/
|-- brain.py                # Classic Brain/Area implementation (authoritative)
|-- brain_util.py           # Overlap computation, save/load utilities
|-- learner.py              # Word acquisition and syntax learning experiments
|-- parser.py               # Sentence parsing via assembly operations
|-- recursive_parser.py     # Recursive descent parser variant
|-- simulations.py          # Simulation runners and experiment harness
|-- image_learner.py        # CIFAR-10 classification via assemblies
|
|-- src/                    # Modular package (pip install -e .)
|   |-- core/               # Refactored Brain, Area, Stimulus, Connectome
|   |-- math_primitives/    # Statistics, plasticity, winner selection, projections
|   |-- simulation/         # Projection, merge, pattern completion, association sims
|   |-- language/           # Grammar rules, language areas, parser, readout
|   |-- gpu/                # CuPy and PyTorch accelerated brain implementations
|   |-- lexicon/            # 5000+ word lexicon, curriculum, NEMO full/hierarchical/ultra
|   |-- nemo/               # NEMO v2: learned grammar, language generation
|   |-- text_generation/    # Assembly-based text generation
|   |-- constants/          # Default parameters
|   |-- utils/              # Math utilities
|   `-- tests/              # Unit and integration tests
|
|-- tests/                  # Root-level test suite
|   |-- test_brain_core.py
|   |-- test_simulations.py
|   `-- performance/        # GPU and CUDA performance benchmarks
|
|-- cpp/
|   |-- cuda_kernels/       # Custom CUDA kernels for assembly operations
|   |-- python_implementations/
|   `-- build_scripts/
|
|-- scripts/                # Utility scripts (animator, overlap sim, turing sim)
|-- research/               # Experiments, results, papers, open questions
`-- pyproject.toml          # Package configuration
```

## Tests

```bash
# Core tests
python -m pytest tests/ src/tests/ -q

# With coverage
python -m pytest tests/ src/tests/ --cov=src --cov-report=term-missing

# Performance benchmarks (requires GPU)
python -m pytest tests/performance/ -q
```

## Dependencies

**Required**: numpy, scipy, matplotlib, pptree, tqdm, pandas, seaborn

**Optional GPU**: cupy-cuda12x, torch

**Dev**: pytest, pytest-cov, ruff

```bash
# Install all optional deps
pip install -e ".[all]"
```

## References

- Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., & Maass, W. (2020). Brain Computation by Assemblies of Neurons. *PNAS*, 117(25), 14464-14472. [doi:10.1073/pnas.2001893117](https://www.pnas.org/doi/full/10.1073/pnas.2001893117)
- Mitropolsky, D., & Papadimitriou, C. H. (2023). The Architecture of a Biologically Plausible Language Organ. [arXiv:2306.15364](https://arxiv.org/abs/2306.15364)

## License

MIT License. See [LICENSE](LICENSE).
