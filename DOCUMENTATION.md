# Brain Simulation Repository Documentation

This documentation provides an overview of the brain simulation repository, detailing each module, their functionalities, and how they interact within the simulation framework. The project implements simulations of neural assemblies based on the concepts introduced in the following papers:

1. **"Brain Computation by Assemblies of Neurons"** by Papadimitriou et al., *Proceedings of the National Academy of Sciences*, 2020.
2. **"The Architecture of a Biologically Plausible Language Organ"** by Mitropolsky et al., 2023.
3. **"Computation with Sequences of Assemblies in a Model of the Brain"** by Dabagia, Papadimitriou, Vempala, *Neural Computation*, 2025. arXiv:2306.03812.

The simulations model how assemblies of neurons can perform computational tasks, including language processing and acquisition, illustrating the computational power of neural circuits in processing complex cognitive functions.

### Documentation by section

Each major part of the codebase has a short README with purpose, main entry points, and links:

| Section | Path | Description |
|--------|------|-------------|
| **Core** | [src/core/README.md](src/core/README.md) | Brain, Area, Stimulus, Connectome, ComputeEngine (CPU/GPU) |
| **Compute** | [src/compute/README.md](src/compute/README.md) | Statistics, plasticity, winner selection, projection primitives |
| **Simulation** | [src/simulation/README.md](src/simulation/README.md) | Projection, association, merge, pattern completion, Turing-style sims |
| **Language** | [src/language/README.md](src/language/README.md) | Rule-based parsing (English/Russian), grammar, readout |
| **Lexicon** | [src/lexicon/README.md](src/lexicon/README.md) | Word lists, curriculum, assembly/GPU learners |
| **NEMO** | [src/nemo/README.md](src/nemo/README.md) | Learned grammar, language acquisition (GPU) |
| **GPU** | [src/gpu/README.md](src/gpu/README.md) | CuPy/PyTorch acceleration (stubs and roadmap) |

See also [ARCHITECTURE.md](ARCHITECTURE.md) for high-level design and [README.md](README.md) for quick start and project overview.

---

## Assembly Calculus Operations (`src/assembly_calculus/`)

The `assembly_calculus` package provides named, first-class operations for neural assembly computation. These wrap the low-level `Brain.project()` calls into clean functional APIs.

### Data Types

- **`Assembly`** — Immutable snapshot of k winner neurons in a brain area. Supports `overlap()` comparison and `len()`.
- **`Sequence`** — Frozen ordered list of `Assembly` snapshots. Supports indexing, iteration, `pairwise_overlaps()`, `overlap_matrix()`, and `mean_consecutive_overlap()`.

### Core Operations (Papadimitriou et al. 2020)

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `project` | `(brain, stimulus, target, rounds=10) → Assembly` | Project stimulus into target area, forming a stable assembly |
| `reciprocal_project` | `(brain, source, target, rounds=10) → Assembly` | Copy an assembly from one area to another |
| `associate` | `(brain, src_a, src_b, target, ...) → Assembly` | Link two assemblies through a shared target area |
| `merge` | `(brain, src_a, src_b, target, ...) → Assembly` | Combine two assemblies into a conjunctive representation |
| `pattern_complete` | `(brain, area, fraction=0.5, ...) → (Assembly, float)` | Recover full assembly from partial activation |
| `separate` | `(brain, stim_a, stim_b, target, ...) → (Assembly, Assembly, float)` | Verify two stimuli create distinct assemblies |

### Sequence Operations (Dabagia et al. 2025)

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `sequence_memorize` | `(brain, stimuli, target, rounds_per_step=10, repetitions=1) → Sequence` | Memorize an ordered sequence of stimuli via Hebbian bridges between consecutive assemblies |
| `ordered_recall` | `(brain, area, cue, max_steps=20, ...) → Sequence` | Recall a memorized sequence from a cue using Long-Range Inhibition (LRI) |

### Long-Range Inhibition (LRI)

LRI enables sequence recall by suppressing recently-fired neurons so the network advances to the next assembly instead of oscillating. Key APIs:

```python
# Add area with LRI (or enable later with set_lri)
brain.add_area("A", n=10000, k=100, beta=0.1,
               refractory_period=3, inhibition_strength=100.0)

# Dynamic toggling: disable during memorization, enable for recall
brain.set_lri("A", refractory_period=3, inhibition_strength=100.0)

# Clear refractory history between phases
brain.clear_refractory("A")
```

LRI parameters:
- **`refractory_period`**: Number of past steps to track (0 = disabled). Recently-fired neurons receive a penalty that decays linearly over this window.
- **`inhibition_strength`**: Penalty magnitude. Use ~5.0 for soft suppression (partial shift), ~100-1000 for hard suppression (force next assembly).

### Control

- **`FiberCircuit`** — Declarative inhibition/disinhibition of projection channels between areas.

---

## Table of Contents

- [Introduction](#introduction)
  - [Neural Assemblies and the Assembly Calculus](#neural-assemblies-and-the-assembly-calculus)
  - [The NEMO Model](#the-nemo-model)
- [Project Overview](#project-overview)
  - [Relation to the Papers](#relation-to-the-papers)
- [Project Structure](#project-structure)
  - [File Descriptions](#file-descriptions)
- [Module Descriptions](#module-descriptions)
  - [Module: Brain Simulation (`brain.py`)](#module-brain-simulation-brainpy)
  - [Module: Brain Utilities (`brain_util.py`)](#module-brain-utilities-brain_utilpy)
  - [Module: Learning Experiments (`learner.py`)](#module-learning-experiments-learnerpy)
  - [Module: Overlap Simulations (`overlap_sim.py`)](#module-overlap-simulations-overlap_simpy)
  - [Module: Parser Simulation (`parser.py`)](#module-parser-simulation-parserpy)
  - [Other Modules](#other-modules)
- [Simulation Details](#simulation-details)
  - [Implementing the Assembly Calculus](#implementing-the-assembly-calculus)
  - [Parser Simulation](#parser-simulation)
  - [Turing Machine Simulation](#turing-machine-simulation)
- [Usage](#usage)
  - [Running the Parser Simulation](#running-the-parser-simulation)
  - [Running Learning Experiments](#running-learning-experiments)
  - [Running Overlap Simulations](#running-overlap-simulations)
  - [Running Other Simulations](#running-other-simulations)
  - [Running Tests](#running-tests)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

### Neural Assemblies and the Assembly Calculus

Neural assemblies are groups of neurons that fire together and represent specific concepts or cognitive functions. According to the Assembly Calculus proposed by Papadimitriou et al., these assemblies can perform complex computations through operations like projection, association, and merge.

The Assembly Calculus provides a computational framework that models how assemblies of neurons can carry out computations. It introduces operations that manipulate these assemblies, allowing for the representation and processing of hierarchical structures, such as those found in language syntax.

Key concepts from the Assembly Calculus:

- **Projection**: Creating a new assembly in a downstream area that represents a copy of an existing assembly.
- **Association**: Increasing the overlap between two assemblies to represent an association between their respective concepts.
- **Merge**: Combining two assemblies to form a new assembly that represents the combination of their respective concepts.

### The NEMO Model

The NEMO (NEuronal MOdels) model, introduced by Mitropolsky et al., extends the Assembly Calculus by providing a biologically plausible model of neural computation. It incorporates additional biological realism, including:

- **Distinct Neuron Types**: Modeling excitatory and inhibitory neurons.
- **Plasticity Mechanisms**: Implementing Hebbian plasticity and synaptic weight adjustments.
- **Brain Areas and Connectivity**: Simulating multiple brain areas with specific connectivity patterns.

The NEMO model aims to bridge the gap between neural activity and cognitive functions, providing a framework to simulate complex tasks like language acquisition and processing.

---

## Project Overview

This project implements simulations that demonstrate how neural assemblies can perform computational tasks, specifically focusing on language processing and Turing machine concepts.

### Relation to the Papers

#### "Brain Computation by Assemblies of Neurons"

The project embodies the concepts introduced by Papadimitriou et al. by:

- **Implementing Assembly Operations**: The simulation uses projection, association, and merge operations to manipulate assemblies, as described in the paper.
- **Modeling Syntax Processing**: It demonstrates how merge operations can represent syntactic structures in language, aligning with the paper's discussion on language processing.
- **Simulating Neural Computation**: The brain model reflects the theoretical underpinnings of neural computation through assemblies.

#### "The Architecture of a Biologically Plausible Language Organ"

The project aligns with the work of Mitropolsky et al. by:

- **Implementing the NEMO Model**: The simulations incorporate the NEMO model to enhance the biological realism of neural computations.
- **Modeling Language Acquisition**: It simulates how assemblies can learn representations of nouns and verbs from grounded input, demonstrating early stages of language acquisition.
- **Exploring Computational Power**: By enhancing the neural model with additional biological realism, the project investigates whether this yields new computational capabilities.

---

## Project Structure

The repository is organized into several modules, each responsible for different aspects of the brain simulation and experiments:

- **`brain.py`**: Core classes for simulating brain areas and their interactions, including `Area` and `Brain`.
- **`brain_util.py`**: Utility functions for saving/loading simulations and computing overlaps between assemblies.
- **`learner.py`**: Implements learning experiments, including word acquisition and syntax learning.
- **`overlap_sim.py`**: Simulations for examining overlap preservation in assemblies and the effect of assembly associations.
- **`parser.py`**: Code for parsing sentences using the brain simulation framework, including classes for different languages.
- **`simulations.py`**: Various simulations to test and illustrate the behavior of neural assemblies, including projection, association, and merge operations.
- **`simulations_test.py`**: Unit tests for the simulations, ensuring the correctness and stability of the implemented algorithms and neural models.
- **`turing_sim.py`**: Simulations related to Turing machine concepts using neural assemblies, exploring the computational limits of the model.
- **`tests.py`**: Additional tests for the brain model.
- **`README.md`**: This documentation file.

### File Descriptions

- **`brain.py`**: Contains the core implementation of the brain model, including neuron assemblies, synaptic connections, and the fundamental operations of the Assembly Calculus and the NEMO model.
- **`brain_util.py`**: Provides utility functions for simulations, including saving and loading brain models, and computing overlaps between neuron assemblies.
- **`learner.py`**: Implements learning experiments using the brain simulation framework. It includes classes and functions for simulating word acquisition, syntax learning, and the effects of various parameters on learning efficiency.
- **`overlap_sim.py`**: Provides simulations for examining overlap preservation in assemblies and the effect of assembly associations.
- **`parser.py`**: Contains code for parsing sentences using the brain simulation framework. It includes classes for different languages (English and Russian) and demonstrates how assemblies can represent grammatical structures.
- **`simulations.py`**: Runs various simulations to test and illustrate the behavior of neural assemblies, including projection, association, merge operations, and language-related tasks.
- **`simulations_test.py`**: Unit tests for the simulations.
- **`turing_sim.py`**: Explores how neural assemblies can simulate Turing machine concepts, investigating the computational limits of the neural model.
- **`tests.py`**: Additional tests for the brain model.

---

## Module Descriptions

### Module: Brain Simulation (`brain.py`)

#### Description

This module provides a configurable assembly model for simulating neural activity within brain areas. It allows for the creation of brain areas (`Area` class) and the brain itself (`Brain` class), facilitating simulations of neural firing patterns, synaptic plasticity, and inter-area connectivity.

#### Relation to the Papers

- Implements the core concepts of the Assembly Calculus and the NEMO model, providing the foundation for simulating neural assemblies and their interactions.
- Models the projection, association, and merge operations as per Papadimitriou et al.'s work.
- Incorporates biologically plausible features such as Hebbian plasticity and random synaptic connectivity.

---

#### Class: `Area`

##### Description

Represents a brain area containing a population of neurons. It tracks neuron firing, synaptic strengths, and supports both explicit and implicit simulations.

##### Attributes

- **name** (`str`): Unique identifier for the area.
- **n** (`int`): Total number of neurons in the area.
- **k** (`int`): Number of neurons that fire during activation (the assembly size).
- **beta** (`float`): Default synaptic plasticity parameter for the area.
- **beta_by_stimulus** (`dict`): Maps stimulus names to their specific `beta` values for synapses into this area.
- **beta_by_area** (`dict`): Maps area names to their specific `beta` values for synapses into this area.
- **w** (`int`): Number of neurons that have ever fired in this area.
- **saved_w** (`list`): Stores the size of the active assembly after each simulation round.
- **winners** (`list`): Indices of neurons that fired in the previous activation.
- **saved_winners** (`list`): Stores the list of `winners` after each simulation round.
- **num_first_winners** (`int`): Number of neurons that fired for the first time in the last activation.
- **fixed_assembly** (`bool`): If `True`, the assembly is fixed and does not change during simulation.
- **explicit** (`bool`): If `True`, the area is fully simulated; otherwise, a sparse simulation is used.

##### Methods

- `__init__(name, n, k, beta=0.05, w=0, explicit=False)`: Initializes an `Area` instance.
- `_update_winners()`: Updates the `winners` list after a projection step.
- `update_beta_by_stimulus(name, new_beta)`: Sets a new `beta` value for synapses from a specific stimulus.
- `update_area_beta(name, new_beta)`: Sets a new `beta` value for synapses from a specific area.
- `fix_assembly()`: Freezes the current assembly, preventing it from changing in future simulations.
- `unfix_assembly()`: Allows the assembly to change in future simulations.
- `get_num_ever_fired()`: Returns the total number of neurons that have ever fired in the area.

---

#### Class: `Brain`

##### Description

Represents the entire brain model, managing multiple areas and stimuli. It handles the creation of areas, stimuli, and the simulation of neural activity and synaptic plasticity across areas.

##### Attributes

- **area_by_name** (`dict`): Maps area names to `Area` instances.
- **stimulus_size_by_name** (`dict`): Maps stimulus names to their sizes (number of firing neurons).
- **connectomes_by_stimulus** (`dict`): Stores the synaptic connections from stimuli to areas.
- **connectomes** (`dict`): Stores the synaptic connections between areas.
- **p** (`float`): Default probability of connection between neurons.
- **save_size** (`bool`): If `True`, saves the assembly sizes after each simulation round.
- **save_winners** (`bool`): If `True`, saves the `winners` after each simulation round.
- **disable_plasticity** (`bool`): If `True`, synaptic plasticity is disabled.
- **_rng** (`numpy.random.Generator`): Random number generator for reproducibility.

##### Methods

- `__init__(p, save_size=True, save_winners=False, seed=0)`: Initializes a `Brain` instance.
- `add_stimulus(stimulus_name, size)`: Adds a stimulus to the brain model.
- `add_area(area_name, n, k, beta)`: Adds a new area to the brain model.
- `add_explicit_area(area_name, n, k, beta, custom_inner_p=None, custom_out_p=None, custom_in_p=None)`: Adds an explicit (fully simulated) area to the brain model.
- `update_plasticity(from_area, to_area, new_beta)`: Updates the synaptic plasticity parameter between two areas.
- `update_plasticities(area_update_map={}, stim_update_map={})`: Updates synaptic plasticity parameters for multiple areas and stimuli.
- `activate(area_name, index)`: Activates a fixed assembly within an area.
- `project(areas_by_stim, dst_areas_by_src_area, verbose=0)`: Simulates neural projections from stimuli and areas into target areas.
- `project_into(target_area, from_stimuli, from_areas, verbose=0)`: Internal method to handle the projection into a specific target area.

---

#### Usage Example

```python
from brain import Brain

# Initialize the brain with a connection probability of 0.05
brain_model = Brain(p=0.05)

# Add stimuli
brain_model.add_stimulus('visual_stimulus', size=100)
brain_model.add_stimulus('auditory_stimulus', size=80)

# Add areas
brain_model.add_area('VisualCortex', n=1000, k=100, beta=0.05)
brain_model.add_area('AuditoryCortex', n=800, k=80, beta=0.05)

# Update plasticity between areas
brain_model.update_plasticity('VisualCortex', 'AuditoryCortex', new_beta=0.1)

# Activate a fixed assembly in the visual cortex
brain_model.activate('VisualCortex', index=0)

# Simulate projections
brain_model.project(
    areas_by_stim={'visual_stimulus': ['VisualCortex']},
    dst_areas_by_src_area={'VisualCortex': ['AuditoryCortex']}
)
```

---

### Module: Brain Utilities (`brain_util.py`)

#### Description

Provides utility functions for simulations, including saving and loading brain models, and computing overlaps between neuron assemblies.

#### Relation to the Papers

- Supports the analysis and validation of simulation results related to assembly overlap and convergence, as discussed in the papers.
- Assists in demonstrating the preservation of overlap during projection and association operations.

---

#### Functions

- `sim_save(file_name, obj)`: Saves a Python object (e.g., a `Brain` instance) to a file using pickle.
  - **Parameters**:
    - `file_name` (`str`): The name of the file to save the object to.
    - `obj` (`object`): The object to save.
- `sim_load(file_name)`: Loads a Python object from a file.
  - **Parameters**:
    - `file_name` (`str`): The name of the file to load the object from.
  - **Returns**: The loaded object.
- `overlap(a, b, percentage=False)`: Computes the overlap between two lists, treating them as sets.
  - **Parameters**:
    - `a` (`list`): First list.
    - `b` (`list`): Second list.
    - `percentage` (`bool`): If `True`, returns the overlap as a percentage of the length of `b`.
  - **Returns**: The number of overlapping elements or the percentage overlap.
- `get_overlaps(winners_list, base, percentage=False)`: Computes the overlap of each list in `winners_list` with a base list.
  - **Parameters**:
    - `winners_list` (`list of lists`): List containing multiple winners lists.
    - `base` (`int`): Index of the base winners list in `winners_list`.
    - `percentage` (`bool`): If `True`, returns overlaps as percentages.
  - **Returns**: A list of overlaps.

---

#### Usage Example

```python
import brain_util as bu

# Save the brain model
bu.sim_save('brain_model.pkl', brain_model)

# Load the brain model
brain_model = bu.sim_load('brain_model.pkl')

# Compute overlap between two assemblies
assembly_a = [1, 2, 3, 4, 5]
assembly_b = [4, 5, 6, 7, 8]
overlap_count = bu.overlap(assembly_a, assembly_b)

# Compute overlap percentage
overlap_percentage = bu.overlap(assembly_a, assembly_b, percentage=True)
```

---

### Module: Learning Experiments (`learner.py`)

#### Description

Implements learning experiments using the brain simulation framework. It includes classes and functions for simulating word acquisition, syntax learning, and the effects of various parameters on learning efficiency.

#### Relation to the Papers

- Demonstrates how the NEMO model can simulate early stages of language acquisition, as described in "The Architecture of a Biologically Plausible Language Organ".
- Models the learning of nouns and verbs from grounded input without explicit supervision.
- Explores the impact of parameters such as plasticity (`beta`) and network connectivity (`p`) on learning efficiency.

---

#### Constants

Defines constants for words, area names, and phoneme indices used in the experiments.

```python
# Words
DOG = "DOG"
CAT = "CAT"
JUMP = "JUMP"
RUN = "RUN"

# Bilingual words
PERRO = "PERRO"
GATO = "GATO"
SALTAR = "SALTAR"
CORRER = "CORRER"

# Area names
PHON = "PHON"
MOTOR = "MOTOR"
VISUAL = "VISUAL"
NOUN = "NOUN"  # Lexicon area for nouns
VERB = "VERB"  # Lexicon area for verbs
CORE = "CORE"  # Area for "cores" (LRI populations)
SEQ = "SEQ"
MOOD = "MOOD"

# Phoneme indices
PHON_INDICES = {
    DOG: 0,
    CAT: 1,
    JUMP: 2,
    RUN: 3,
    PERRO: 4,
    GATO: 5,
    SALTAR: 6,
    CORRER: 7
}
```

---

#### Functions

- `lexicon_sizes_experiment(...)`: Tests learning efficiency across different lexicon sizes.
- `betas_experiment(...)`: Tests the effect of varying the `beta` parameter on learning.
- `p_experiment(...)`: Tests the effect of varying the connection probability `p`.
- `single_word_tutoring_exp(...)`: Simulates learning with single-word tutoring.

---

#### Class: `LearnBrain`

##### Description

Extends the `Brain` class to simulate language acquisition experiments, including word learning and syntax.

##### Methods

- `__init__(...)`: Initializes the `LearnBrain` instance with specified parameters.
- `tutor_single_word(word)`: Tutors a single word to the brain.
- `tutor_random_word()`: Tutors a random word from the lexicon.
- `activate_context(word)`: Activates the contextual areas associated with a word.
- `activate_PHON(word)`: Activates the phonological representation of a word.
- `project_star(mutual_inhibition=False)`: Simulates projections with optional mutual inhibition between areas.
- `parse_sentence(sentence)`: Parses a sentence by activating contexts and phonemes.
- `train_simple(rounds)`: Trains the brain with a simple set of sentences.
- `train_random_sentence()`: Trains the brain with a random sentence.
- `train_experiment(max_rounds=100, use_extra_context=False)`: Runs a training experiment and tests word acquisition.
- `train_experiment_randomized(...)`: Runs a randomized training experiment.
- `test_all_words(use_extra_context=False)`: Tests whether all words have been correctly learned.
- `testIndexedWord(word_index, min_overlap=0.75, use_extra_context=False, no_print=False)`: Tests retrieval of a word by its index.

---

#### Class: `SimpleSyntaxBrain`

##### Description

Simulates syntax acquisition, focusing on word order and grammatical structures using sequence and core areas.

##### Methods

- `__init__(...)`: Initializes the `SimpleSyntaxBrain` instance.
- `add_cores(...)`: Adds core areas representing grammatical roles.
- `parse(sentence, mood_state=0)`: Parses a sentence, simulating syntax processing.
- `pre_train(proj_rounds=20)`: Pre-trains the model to establish initial synaptic strengths.
- `train(order, train_rounds=40, train_interrogative=False)`: Trains the brain with sentences in a specified word order.

---

#### Usage Example

```python
from learner import LearnBrain

# Initialize the brain for word acquisition
brain = LearnBrain(p=0.05, LEX_k=100)

# Train with simple sentences
brain.train_simple(30)

# Test word retrieval
retrieved_word = brain.testIndexedWord(PHON_INDICES["RUN"])
print(f"Retrieved word: {retrieved_word}")
```

---

#### Notes

- **Bilingual Support**: The `LearnBrain` class can simulate bilingual experiments by setting the `bilingual` parameter.
- **Extra Context Areas**: Supports additional context areas for more complex simulations.
- **Syntax Learning**: The `SimpleSyntaxBrain` class demonstrates how syntax and word order can be learned using assemblies and core areas.

---

### Module: Overlap Simulations (`overlap_sim.py`)

#### Description

Provides simulations for examining overlap preservation in assemblies and the effect of assembly associations. It demonstrates how assemblies preserve overlap and how associative learning impacts the overlap between projected assemblies.

#### Relation to the Papers

- Validates the analytical and simulation results regarding the preservation of overlap during projection and association, as discussed in "Brain Computation by Assemblies of Neurons".
- Explores the implications of overlap preservation for probabilistic computation through assemblies.

---

#### Functions

- `overlap_sim(n=100000, k=317, p=0.05, beta=0.1, project_iter=10)`: Simulates the preservation of overlap between assemblies.
- `overlap_grand_sim(n=100000, k=317, p=0.01, beta=0.05, min_iter=10, max_iter=30)`: Simulates overlap over multiple iterations.

---

#### Usage Example

```python
import overlap_sim

# Run a simple overlap simulation
assembly_overlap, proj_overlap = overlap_sim.overlap_sim()

print(f"Assembly Overlap: {assembly_overlap}")
print(f"Projection Overlap: {proj_overlap}")

# Run a grand simulation over multiple iterations
results = overlap_sim.overlap_grand_sim()

for assembly_overlap, proj_overlap in results.items():
    print(f"Assembly Overlap: {assembly_overlap} -> Projection Overlap: {proj_overlap}")
```

---

### Module: Parser Simulation (`parser.py`)

#### Description

Contains code for parsing sentences using the brain simulation framework. It includes classes for different languages (English and Russian) and demonstrates how assemblies can represent grammatical structures.

#### Relation to the Papers

- Implements the parser as described in "A Biologically Plausible Parser" by Mitropolsky et al.
- Demonstrates how merge operations can represent syntactic structures, aligning with the discussion in "Brain Computation by Assemblies of Neurons".
- Models language processing using assemblies, reflecting the computational power of neural circuits in handling language.

---

#### Classes

- `ParserBrain`: Extends the `Brain` class to handle parsing-specific functionality, such as applying grammar rules and managing fiber states.
- `EnglishParserBrain`: Extends `ParserBrain` for English grammar, implementing specific rules and areas.
- `RussianParserBrain`: Extends `ParserBrain` for Russian grammar, accommodating cases and free word order.

---

#### Key Components

- **Areas**: Defines brain areas corresponding to grammatical roles (e.g., `SUBJ`, `VERB`, `OBJ`).
- **Lexemes**: Dictionaries (`LEXEME_DICT` for English and `RUSSIAN_LEXEME_DICT` for Russian) mapping words to their grammatical roles and rules.
- **Rules**: Defines pre- and post-processing rules (`PRE_RULES` and `POST_RULES`) for each lexeme to simulate grammar.

---

#### Functions

- `parse(sentence, language="English", p=0.1, LEX_k=20, project_rounds=20, verbose=True, debug=False, readout_method=ReadoutMethod.FIBER_READOUT)`: Parses a sentence using the specified language model.

---

#### Usage Example

```python
from parser import parse

# Parse an English sentence
parse("cats chase mice", language="English")

# Parse a Russian sentence
parse("kot vidit sobaku", language="Russian")
```

---

#### Notes

- **Grammar Rules**: The parser uses pre-defined grammar rules to simulate the parsing process, applying activation and inhibition in various brain areas.
- **Debugging**: The `ParserDebugger` class allows step-by-step debugging of the parsing process.
- **Readout Methods**: Supports different methods to read out the parse tree, including fixed map readout and fiber activation readout.

---

### Other Modules

#### `simulations.py`

- **Description**: Runs various simulations to test and illustrate the behavior of neural assemblies, including projection, association, merge operations, and language-related tasks.
- **Relation to the Papers**: Validates the mathematical predictions of the Assembly Calculus and NEMO model through simulations.

#### `simulations_test.py`

- **Description**: Contains unit tests for the simulations, ensuring the correctness and stability of the implemented algorithms and neural models.

#### `turing_sim.py`

- **Description**: Explores how neural assemblies can simulate Turing machine concepts, investigating the computational limits of the neural model.
- **Relation to the Papers**: Demonstrates that the enhanced NEMO model can perform arbitrary computations, supporting the claims about the model's computational capabilities.

#### `tests.py`

- **Description**: Additional tests for the brain model, focusing on specific scenarios and edge cases to validate the neural computations.

---

## Simulation Details

### Implementing the Assembly Calculus

The project implements the operations of the Assembly Calculus within the `brain.py` and `brain_util.py` modules. These operations include:

- **Projection**: Implemented in the `project` method of the `Brain` class, where an assembly projects to a downstream area, forming a new assembly.
- **Association**: Implemented in the `associate` method (if present), where two assemblies increase their overlap through simultaneous activation.
- **Merge**: Implemented in the `merge` method (if present), combining two assemblies to form a new one with connections to both parent assemblies.

The simulations validate the mathematical predictions of the Assembly Calculus, such as the convergence of assemblies during projection and the preservation of overlaps.

### Parser Simulation

The `parser.py` script simulates a parser based on neural assemblies, demonstrating how assemblies can represent and process language structures.

#### Key Components

- **Brain Areas**: Represent different linguistic components, such as `LEX` (lexicon), `SUBJ` (subject), `VERB`, and `OBJ` (object).
- **Assemblies**: Groups of neurons representing specific words or grammatical structures.
- **Operations**: Utilize projection and association to simulate the activation and interaction of assemblies.
- **Rules**: Define how assemblies interact based on grammatical structures, implemented through `AreaRule` and `FiberRule` classes.

#### How It Works

1. **Word Activation**: As each word in a sentence is read, its corresponding assembly in the lexicon (`LEX`) is activated in the `PHON` (phonological) area.

2. **Rule Application**: Predefined rules determine how this activation projects to other brain areas, simulating grammatical processing. For example, a noun projects from `PHON` to `LEX_NOUN` and then to `SUBJ` or `OBJ` based on its role.

3. **Projection**: Assemblies project to other areas, forming new assemblies that represent higher-level structures, such as phrases or sentences.

4. **Readout**: After processing, the assemblies can be read out to reconstruct the parsed sentence or to extract grammatical relationships.

#### Relation to Language Acquisition

By simulating how assemblies form and interact during language processing, the project models early stages of language acquisition as described in "The Architecture of a Biologically Plausible Language Organ". It demonstrates how neural circuits can learn representations of nouns and verbs from grounded input without explicit supervision.

### Turing Machine Simulation

The `turing_sim.py` script explores how neural assemblies can simulate Turing machine concepts, investigating the computational limits of the neural model.

#### Key Concepts

- **Sequence Formation**: Simulating the storage and retrieval of sequences, analogous to the tape of a Turing machine.
- **State Transitions**: Modeling state changes using assemblies, where each state is represented by an assembly, and transitions are simulated through projections.
- **Computational Power**: Demonstrating that the enhanced NEMO model can perform arbitrary computations, supporting the claims in the papers about the model's computational capabilities.

---

## Usage

### Running the Parser Simulation

To run the parser simulation:

1. **Ensure Dependencies Are Installed**: See [Dependencies](#dependencies) below.

2. **Run the `parser.py` Script**:

   ```bash
   python parser.py
   ```

3. **Input Sentences**: Modify the `sentence` variable in the `parse()` function to test different inputs.

### Running Learning Experiments

To run learning experiments for word acquisition and syntax learning:

```bash
python learner.py
```

You can modify parameters within the script to test different scenarios, such as varying lexicon sizes, plasticity parameters, or connection probabilities.

### Running Overlap Simulations

To run overlap simulations:

```bash
python overlap_sim.py
```

This will execute the simulations that examine overlap preservation in assemblies and print the results.

### Running Other Simulations

- **Projection and Association Simulations**: Run `simulations.py` to execute various tests and observe the behavior of neural assemblies.

  ```bash
  python simulations.py
  ```

- **Turing Machine Simulation**: Run `turing_sim.py` to explore Turing machine concepts using neural assemblies.

  ```bash
  python turing_sim.py
  ```

### Running Tests

To run the unit tests:

```bash
python simulations_test.py
python tests.py
```

---

## Dependencies

- **Python 3.x**

- **Required Libraries**:

  - `numpy`: For numerical computations and array manipulations.
  - `matplotlib`: For plotting and visualizations.
  - `unittest`: For running tests.
  - `pickle`: For saving and loading simulation states.

Install dependencies using pip:

```bash
pip install numpy matplotlib
```

---

## Contributing

Contributions are welcome! Please submit issues or pull requests to help improve the project.

---

## License

This project is licensed under the MIT License.

---

This documentation provides an integrated and detailed overview of the brain simulation repository, combining all relevant content to aid in understanding and utilizing the simulation framework. The examples illustrate how to set up and run experiments, helping users to grasp how the classes and methods interact within the context of neural assemblies and the computational models described in the referenced papers.