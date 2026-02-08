# Assemblies

**Neural assembly calculus:** projection, association, merge — with language learning (NEMO), GPU acceleration, and Turing-style simulation. Python 3.10+.

---

I'm Alif Jakir, I like to tinker! Some of my research work involves me extending the neural assemblies framework toward omnimodality, scaling, and general language learning grounded in neurobiological and neurolinguistic facts, in collaboration with Daniel Mitropolsky (MIT Poggio Lab; [dmitropolsky.github.io](https://dmitropolsky.github.io/)). As you can see I am quite excited by neural alternatives beyond deep learning... which is much in vogue today.

I started extending Daniel's PhD thesis project in *2024* in MIT's [Projects in the Science of Intelligence](https://poggio-lab.mit.edu/9-58/) class. I asked whether the neural assembly calculus and NEMO-style models could do visual discrimination — they could. I didn't get stable enough assemblies for every CIFAR-10 category consistently, but the experiments showed it works in principle. I also rewrote the codebase several times to make it easier to extend and understand; this repo is the result.

This codebase is a garden I've co-built with AI, exploring assemblies as a foundational computational paradigm.

**What this is:** A computational neuroscience framework that implements the **Assembly Calculus** — a model of brain computation via neural ensembles (assemblies) — with extensions for language learning, image classification, GPU acceleration, and Turing-machine simulation. It is a concrete, discrete, efficiently optimized spiking-neural implementation that stays within neurobiologically plausible dynamics. Assemblies are the intermediate primitive between raw neural activity and high-level cognition. The framework is Turing-complete (Dabagia et al., 2023, via sequences of assemblies and long-range interneurons) and is treated here as a family of highly energy-efficient dynamical systems.

**What this is not:** Not a deep-learning library (no backprop, no attention). Not a full biological brain model — it's a minimal calculus (projection, association, merge, Hebbian, top-k) that we scale and compose. Not claiming to beat transformers at everything; the bet is that this *alternative* — sparse, interpretable, neurobiologically grounded — can support language and reasoning in a way that's efficient and comprehensible.

**Also not:** Not a biophysical spiking simulator (no ion channels, no ms-level dynamics) — we use discrete rounds of projection and competition. "Neurobiologically plausible" means the *operations* are plausible (Hebbian, sparse, local), not that we're modeling a specific brain region or species in detail. Not differentiable end-to-end — you can't plug it into autograd; assemblies are the primitive, not an interpretation layer on top of a big net. Not a chatbot or API — it's a research codebase and simulation framework; no hosted service.

This is also not the only implementation of the assembly calculus, it is merely the most complete artifact — *just one* implementation with specific extensions (NEMO, lexicon, GPU). "Co-built with AI" means I direct and AI assists; the science and design are mine. We're not claiming we've solved language or AGI, or that assemblies are *the* way the brain does it — we're exploring one formalization. Turing-completeness is from Dabagia et al. (2023), not our result.

**Why it matters (to us):** One substrate (assemblies) for perception, language, and structure; interpretable primitives (assemblies = concepts); and a path to scaling that stays close to how cortex might compute. We're exploring whether this can be a foundation for a different kind of model.

**Who it's for:** Researchers and students in computational neuroscience, neuro-inspired ML, or alternative approaches to language and reasoning — and anyone curious about assembly calculus, NEMO-style language learning, or scaling sparse neural systems without backprop.

**Where things stand:** Core operations (projection, association, merge) are implemented and validated; NEMO does learned grammar and word order; scaling and CUDA work has succeeded. We now implement **sequence memorization**, **ordered recall**, and **Long-Range Inhibition (LRI)** from Dabagia et al. (2023/2025) — enabling ordered sequence learning and cue-driven recall via refractory suppression. FSM-learning is the remaining piece from that paper. Full foundation-model scale and omnimodality are goals, not yet achieved. CIFAR-10 showed feasibility in principle; stable per-category assemblies at scale are still in progress.

My main goal in summer *2025* was to scale the system using custom CUDA and algorithmic improvements toward a new kind of foundational model. Success was found in the optimization project.

**Based on:**
- Papadimitriou et al., "Brain Computation by Assemblies of Neurons", *PNAS* 2020
- Mitropolsky & Papadimitriou, "The Architecture of a Biologically Plausible Language Organ", 2023
- Mitropolsky & Papadimitriou, "Simulated Language Acquisition in a Biologically Realistic Model of the Brain", 2025
- Hebb, "The Organization of Behavior", 1949 (Hebbian plasticity)
- Hopfield, "Neural networks and physical systems with emergent collective computational abilities", *PNAS* 1982 (attractor dynamics)
- Olshausen & Field, "Sparse coding with an overcomplete basis set: A strategy employed by V1?", *Vision Research* 1997 (sparse coding)

## Contents

- [Documentation](#documentation) · [Requirements](#requirements) · [Quick Start](#quick-start) · [Overview](#overview) · [Usage](#usage) · [Project Structure](#project-structure) · [Tests](#tests) · [Citation](#citation)

## Documentation

**By section (each has a README):**

| Section | Path | Description |
|--------|------|-------------|
| **Core** | [src/core/README.md](src/core/README.md) | Brain, Area, Stimulus, Connectome, ComputeEngine (CPU/GPU) |
| **Compute** | [src/compute/README.md](src/compute/README.md) | Statistics, plasticity, winner selection, projection primitives |
| **Simulation** | [src/simulation/README.md](src/simulation/README.md) | Projection, association, merge, pattern completion, Turing-style sims |
| **Language** | [src/language/README.md](src/language/README.md) | Rule-based parsing (English/Russian), grammar, readout |
| **Lexicon** | [src/lexicon/README.md](src/lexicon/README.md) | Word lists, curriculum, assembly/GPU learners |
| **NEMO** | [src/nemo/README.md](src/nemo/README.md) | Learned grammar, language acquisition (GPU) |
| **GPU** | [src/gpu/README.md](src/gpu/README.md) | CuPy/PyTorch acceleration (stubs and roadmap) |

**Project-wide:**

- [**DOCUMENTATION.md**](DOCUMENTATION.md) — Full API and module guide.
- [**ARCHITECTURE.md**](ARCHITECTURE.md) — High-level design and engine layout.
- [**research/**](research/README.md) — Experiments, results, plans, open questions.

## Requirements

- **Python 3.10+**
- For GPU: CUDA-capable device and [cupy-cuda13x](https://pypi.org/project/cupy-cuda13x/) (or cupy-cuda12x); see [pyproject.toml](pyproject.toml) optional deps.

## Quick Start

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# With GPU support (CUDA 12.x / 13.x)
pip install -e ".[gpu]"

# Run tests
uv run python -m pytest src/tests/ -q

# Quick check: project a stimulus into an area (from repo root)
uv run python -c "
from src.core.brain import Brain
b = Brain(p=0.05, engine='numpy_sparse')
b.add_stimulus('s', size=50); b.add_area('A', n=2000, k=50, beta=0.05)
b.project({'s': ['A']}, {}); b.project({}, {'A': ['A']})
print('Winners (first 10):', b.area_by_name['A'].winners[:10])
"
```

**Note:** No GPU? Use `engine="numpy_sparse"` (default when CuPy is missing). Run from the repo root or after `pip install -e .` so `src` and root scripts resolve.

## Overview

Neural assemblies are groups of neurons that fire together to represent concepts. The Assembly Calculus provides core operations on these assemblies:

- **Projection**: Copy an assembly into a downstream brain area
- **Association**: Increase overlap between assemblies to link concepts
- **Merge**: Combine two assemblies into a new one representing their conjunction
- **Sequence Memorize**: Learn an ordered sequence of assemblies via Hebbian bridges
- **Ordered Recall**: Replay a memorized sequence from a cue using Long-Range Inhibition (LRI)

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

### Assembly Calculus Operations

```python
from src.core.brain import Brain
from src.assembly_calculus import (
    project, associate, merge, pattern_complete, separate,
    sequence_memorize, ordered_recall,
    Assembly, Sequence, overlap,
)

b = Brain(p=0.05, save_winners=True, seed=42, engine="numpy_sparse")
for i in range(3):
    b.add_stimulus(f"s{i}", 100)
b.add_area("A", n=10000, k=100, beta=0.1)

# Memorize a sequence of stimuli
seq = sequence_memorize(b, ["s0", "s1", "s2"], "A", rounds_per_step=10)
print(f"Memorized {len(seq)} assemblies, consecutive overlap: {seq.mean_consecutive_overlap():.3f}")

# Enable LRI and recall the sequence from a cue
b.set_lri("A", refractory_period=3, inhibition_strength=100.0)
recalled = ordered_recall(b, "A", "s0", max_steps=10, known_assemblies=list(seq))
print(f"Recalled {len(recalled)} assemblies")
print(f"First recalled overlaps first memorized: {overlap(recalled[0], seq[0]):.3f}")
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
|-- src/                    # Modular package (pip install -e .); see section READMEs
|   |-- assembly_calculus/  # Named ops (project, merge, sequence_memorize, ordered_recall)
|   |-- core/               # Brain, Area, Stimulus, Connectome, ComputeEngine → README
|   |-- compute/            # Statistics, plasticity, winner selection, projections → README
|   |-- simulation/         # Projection, merge, pattern completion, association sims → README
|   |-- language/           # Grammar rules, language areas, parser, readout → README
|   |-- lexicon/            # 5000+ word lexicon, curriculum, GPU learners → README
|   |-- nemo/               # NEMO v2: learned grammar, language generation (GPU) → README
|   |-- gpu/                # CuPy/PyTorch acceleration (stubs, roadmap) → README
|   |-- constants/          # Default parameters
|   |-- utils/               # Math utilities
|   `-- tests/               # Unit and integration tests
|
|-- research/               # Experiments, results, plans → research/README.md
|-- cpp/                    # Custom CUDA kernels (.cu) and build scripts → cpp/README.md
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

**Optional GPU**: cupy-cuda12x or cupy-cuda13x (see [pyproject.toml](pyproject.toml)), torch

**Dev**: pytest, pytest-cov, ruff

## Citation

If you use this code or build on the assembly calculus, please cite the foundational work:

- **Assembly calculus:** Papadimitriou et al. (2020), *Brain Computation by Assemblies of Neurons*, PNAS.
- **Sequences, FSM, Turing-completeness:** Dabagia et al. (2023), *Computation with Sequences of Assemblies in a Model of the Brain*, ALT 2024 / arXiv:2306.03812.
- **Language organ / NEMO:** Mitropolsky & Papadimitriou (2023), *The Architecture of a Biologically Plausible Language Organ*; Mitropolsky & Papadimitriou (2025), *Simulated Language Acquisition in a Biologically Realistic Model of the Brain*.

Full references below.

## References

- Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., & Maass, W. (2020). Brain Computation by Assemblies of Neurons. *PNAS*, 117(25), 14464–14472. [doi:10.1073/pnas.2001893117](https://www.pnas.org/doi/full/10.1073/pnas.2001893117)
- Dabagia, M., Papadimitriou, C. H., & Vempala, S. S. (2023). Computation with Sequences of Assemblies in a Model of the Brain. *Proceedings of the 35th Conference on Algorithmic Learning Theory (ALT)* 2024. [arXiv:2306.03812](https://arxiv.org/abs/2306.03812)
- Mitropolsky, D., & Papadimitriou, C. H. (2023). The Architecture of a Biologically Plausible Language Organ. [arXiv:2306.15364](https://arxiv.org/abs/2306.15364)
- Mitropolsky, D., & Papadimitriou, C. H. (2025). Simulated Language Acquisition in a Biologically Realistic Model of the Brain. [arXiv:2507.11788](https://arxiv.org/abs/2507.11788). [doi:10.48550/arXiv.2507.11788](https://doi.org/10.48550/arXiv.2507.11788)
- Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.
- Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by V1? *Vision Research*, 37(23), 3311–3325.

## Acknowledgments

This project grew out of MIT's [Projects in the Science of Intelligence](https://poggio-lab.mit.edu/9-58/) (9.58). Thanks to the Poggio Lab and Daniel Mitropolsky for the collaboration on the assembly calculus and language organ.

## License

MIT License. See [LICENSE](LICENSE).
