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

# Run composed ERP experiments
uv run python research/experiments/primitives/test_composed_erp.py --quick
uv run python research/experiments/primitives/test_incremental_erp.py --quick

# Run variable-length sentence experiments
uv run python research/experiments/primitives/test_variable_length.py --quick
uv run python research/experiments/primitives/test_variable_incremental.py --quick
uv run python research/experiments/primitives/test_recursive_structure.py --quick

# Run psycholinguistic phenomena experiments
uv run python research/experiments/primitives/test_garden_path_erp.py --quick
uv run python research/experiments/primitives/test_rc_asymmetry.py --quick
uv run python research/experiments/primitives/test_agreement_attraction.py --quick

# Run robustness & depth experiments
uv run python research/experiments/primitives/test_generalization.py --quick
uv run python research/experiments/primitives/test_unified_phenomena.py --quick
uv run python research/experiments/primitives/test_cloze_probability.py --quick
uv run python research/experiments/primitives/test_parameter_robustness.py --quick
uv run python research/experiments/primitives/test_semantic_similarity.py --quick

# Run developmental and generation experiments
uv run python research/experiments/primitives/test_developmental_curriculum.py --quick
uv run python research/experiments/primitives/test_sentence_generation.py --quick

# Run stability tests
uv run python research/experiments/stability/test_phase_diagram.py --quick
```

## Experiment Categories

### 1. Primitives (`primitives/`)

Validates core Assembly Calculus operations and derived phenomena:

**Core Operations:**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Projection** | Does projection reliably create stable assemblies? | ✅ Validated |
| **Association** | Does co-activation increase assembly overlap? | ✅ Validated |
| **Merge** | Does simultaneous projection create composite assemblies? | ✅ Validated |

**Derived Primitives:**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Self-Organization** | Do category areas self-organize from distributional input? | ✅ Validated |
| **Role Discovery** | Do structural roles emerge from co-projection? | ✅ Validated |
| **Forward Prediction** | Can Hebbian bridges support context-driven prediction? | ✅ Validated |
| **Binding & Retrieval** | Can co-projection bind words to roles and retrieve them? | ✅ Validated |

**Composed Phenomena (ERP Signals):**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Prediction N400** | Does prediction error produce graded N400 signals? | ✅ Validated |
| **Binding P600** | Does anchored instability produce graded P600 signals? | ✅ Validated |
| **Composed ERP** | Do N400 and P600 show a double dissociation? | ✅ Validated |
| **Incremental ERP** | Does online learning produce graded, developmental ERP curves? | ✅ Validated |
| **ERP Diagnostics** | What are the representational and signal-flow dynamics? | ✅ Complete |

**Variable-Length Processing:**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Variable-Length** | Does prediction+binding generalize to SVO+PP? | ✅ Validated |
| **Variable Incremental** | Do CFG-generated sentences produce graded learning curves? | ✅ Validated |
| **Recursive Structure** | Does AC handle recursive PP and center-embedding? | ✅ Validated |
| **Garden-Path ERP** | Does structural ambiguity produce prediction violation + reanalysis? | ✅ Validated |
| **RC Asymmetry** | Does the SRC > ORC processing asymmetry emerge from binding? | ✅ Validated |
| **Agreement Attraction** | Do intervening PP nouns interfere with subject binding? | ✅ Validated |

**Robustness & Depth:**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Generalization** | Do phenomena emerge from composition, not memorization? | ✅ Validated |
| **Unified Phenomena** | Do all phenomena coexist in a single brain? | ✅ Validated |
| **Cloze Probability** | Does N400 scale continuously with training frequency? | ✅ Validated |
| **Parameter Robustness** | Are effects robust across parameter ranges? | ✅ Validated |
| **Semantic Similarity** | Does co-projection create semantic overlap affecting N400? | 🔄 Partial |

**Development & Generation:**

| Experiment | Scientific Question | Status |
|------------|---------------------|--------|
| **Developmental Curriculum** | Do phenomena emerge in child-like developmental order? | ✅ Validated |
| **Sentence Generation** | Can the system produce novel grammatical sequences? | ✅ Validated |

See `research/results/primitives/RESULTS_composed_erp.md` for detailed findings.

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

## Key Findings

### Primitive Validation
- **Projection**: 100% convergence rate across all tested configurations
- **Association**: Overlap increases consistently with co-activation
- **Merge**: Composite assemblies capture both parent concepts (88% quality)

### Composed ERP Signals
- **N400/P600 double dissociation** from a single prediction + binding mechanism
- Novel nouns show N400 effect (+0.89) but not P600 (~0); category violations show both
- **Anchored instability** (P600 metric, d=21-90) resolves the zero-signal problem
  where untrained pathways paradoxically showed zero Jaccard instability
- **Incremental learning** produces graded word-frequency effects (N400 decreases
  with exposure) and immediate category-level generalization

### Variable-Length Processing
- **Double dissociation preserved** at both object (word 3) and PP-object (word 5) positions
- **Prediction chain works across positions**: verb->object, object->prep, prep->PP-obj
  (N400 at preposition: expected=0.09, unexpected=0.39, d=5.60)
- **CFG-generated sentences** produce the same learning curves as hand-crafted templates
- PP bindings develop alongside SVO bindings with no special handling
- 8-area architecture (3 lexical + 1 prediction + 3 role + 1 prep) handles variable length

### Recursive Structure & Center-Embedding
- **Recursive PP** at depth 0 (P600 d=14.39) and depth 1 (d=15.65) — same double
  dissociation pattern at arbitrary recursive depth
- **Center-embedding** ("dog that chases cat sees bird") separates at both clause
  levels: main patient P600 d=12.07, relative patient P600 d=10.58
- **Dual binding** works: agent binds to both ROLE_AGENT (instability 0.214) and
  ROLE_REL_AGENT (0.101) simultaneously — one word, two structural roles
- **Prediction across embedding**: main verb N400 = 0.181 after intervening relative
  clause — the prediction chain persists across embedded material
- 11-area architecture (NOUN_CORE, VERB_CORE, PREP_CORE, COMP_CORE, PREDICTION,
  ROLE_AGENT, ROLE_PATIENT, ROLE_PP_OBJ, ROLE_PP_OBJ_1, ROLE_REL_AGENT, ROLE_REL_PATIENT)

### Psycholinguistic Phenomena
- **Garden-path effect**: Omitting the complementizer ("dog chases cat sees bird"
  vs "dog that chases cat sees bird") produces N400 at the second verb (d=0.88) —
  the system predicted PP or sentence-end, not another verb. Post-object predictions
  confirm: prep N400=0.09 < verb=0.17 < noun=0.43.
- **SRC/ORC asymmetry**: Object-relatives produce higher processing difficulty than
  subject-relatives. The key result is dual-binding P600: SRC (same-direction
  AGENT+REL_AGENT) = 0.043, ORC (conflicting AGENT+REL_PATIENT) = 0.181, d=0.81.
  The asymmetry emerges from binding direction conflict, not from any built-in SRC/ORC
  distinction.
- **Agreement attraction**: Intervening PP nouns increase AGENT binding P600 (no PP:
  0.119, short PP: 0.130, d=0.67). The effect plateaus rather than increasing with
  further PP depth, suggesting a threshold interference mechanism rather than linear
  distance scaling.

### Robustness & Depth
- **Generalization**: Phenomena emerge from composition, not memorization. ORC dual-binding
  produces elevated P600 even when trained only on SRC (never saw AGENT+REL_PATIENT binding).
  Depth-1 PP dissociation holds when only depth-0 was trained.
- **Unified phenomena**: A single brain trained on one mixed corpus simultaneously exhibits
  5/6 tested phenomena with d > 0.5, including object dissociation (d=11.58), PP dissociation
  (d=15.98), garden-path (d=0.58), agreement attraction (d=0.58), and recursive PP (d=28-94).
- **Cloze probability**: N400 scales continuously with training frequency. Monotonic gradient:
  high < med < low < zero. The quantitative frequency-N400 relationship matches the pattern
  needed for fitting human cloze probability data.
- **Parameter robustness**: P600 double dissociation is rock-solid across all tested parameter
  combinations (d=3.6 to 60.5). Subtler phenomena (ORC/SRC, garden-path) require sufficient
  network capacity and training to reliably detect.
- **Semantic similarity**: Co-projection creates measurable semantic structure in NOUN_CORE
  (related overlap 0.898 > unrelated 0.890, d=0.73). The N400 priming effect through
  prediction is minimal with equal training frequency — frequency dominates semantic
  relatedness in the current architecture.

### Development & Generation
- **Developmental curriculum**: Phenomena emerge in the correct developmental order from a
  single brain trained through six staged inputs. Forward prediction (N400 d=14.03) emerges
  exactly at the SVO stage. PP binding (P600 d=10.16) emerges exactly at the SVO+PP stage.
  Binding P600 is rock-solid once SVO training begins (d=7.34 → 64.92).
- **Sentence generation**: The bidirectional pathways created by role binding support reverse
  readout. Role accuracy reaches 1.0 by the SVO stage — ROLE_AGENT and ROLE_PATIENT reliably
  recover nouns. Prediction chain generation produces novel grammatical sequences (SVO
  rate=1.0 at stage 3). After PP training, chains follow the most-trained pathway
  ("dog in park in park"), showing that prediction strength drives generation.

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

