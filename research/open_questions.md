# Open Research Questions

A living document tracking all questions being explored in this project.
Questions are organized by theme and status.

Last Updated: 2026-04-22

---

## Fundamental Theory

### Q01: Assembly Stability
**Status:** Completed
**Hypothesis:** Sparse assemblies converge to stable attractors under specific connectivity and sparsity conditions.
**Result:** In the tested recurrent regimes, stim+self training produces autonomous persistence well above chance. The fixed-training projection study gives `0.709 +/- 0.032` persistence vs `0.113 +/- 0.007` for the stim-only control at `n=1000, k=100`, while heavier single-assembly training reaches `0.97 - 1.00` persistence in favorable settings.
**Location:** `research/core_questions/Q01_assembly_stability/`

### Q02: Critical Phenomena
**Status:** Not Started
**Hypothesis:** Assembly formation exhibits phase transitions at critical sparsity levels.
**Why It Matters:** Explains robustness and universality.
**Location:** `core_questions/Q02_critical_phenomena/`

### Q03: Scaling Laws
**Status:** Completed - needs theoretical derivation
**Hypothesis:** Assembly properties follow systematic scaling trends with network size.
**Result:** In the tested `k=sqrt(n)` regime, convergence time trends downward with network size. The current fit is `T = -17.49 * log10(n) + 78.68`, `R^2 = 0.601`, `p = 0.070`, so the evidence is consistent with logarithmic scaling but still noisy.
**Note:** Persistence stays moderate rather than monotonic because increasing `n` also decreases `k/n`.
**Location:** `research/core_questions/Q03_scaling_laws/`

---

## Algorithmic Contributions

### Q04: Complexity Bounds
**Status:** Not Started
**Hypothesis:** Top-K selection can be performed in `O(N log K)` time with appropriate data structures.
**Why It Matters:** Enables billion-scale simulation.
**Location:** `core_questions/Q04_complexity_bounds/`

### Q05: Memory Efficiency
**Status:** Not Started
**Hypothesis:** Sparse representations reduce memory by more than 80 percent without accuracy loss.
**Why It Matters:** Practical scalability.
**Location:** `core_questions/Q05_memory_efficiency/`

---

## Biological Validation

### Q06: Assembly Detection
**Status:** Not Started
**Hypothesis:** Computational assemblies can be detected in biological neural recordings.
**Why It Matters:** Validates theoretical predictions.
**Location:** `core_questions/Q06_assembly_detection/`

### Q07: Sparsity Measurements
**Status:** Completed
**Hypothesis:** Biological neural networks exhibit predicted sparsity levels in the 1-5 percent range.
**Result:** The repo's chosen cortical/hippocampal sparsity settings (2 percent) and local connection probability settings (10 percent) fall within the cited biological parameter ranges used by the validation script. This is parameter alignment, not a full neural-data comparison.
**Location:** `research/experiments/biological_validation/test_biological_parameters.py`

### Q08: Cross-Species Universality
**Status:** Not Started
**Hypothesis:** Assembly principles are universal across species and brain regions.
**Why It Matters:** Tests generality of theory.
**Location:** `core_questions/Q08_universality/`

---

## Information Theory

### Q09: Coding Capacity
**Status:** Completed (with important caveat)
**Hypothesis:** Sparse assembly coding is near-optimal for biological energy/information constraints.
**Result:** `0.282 bits/neuron` achieved.
**Critical Caveat:** Only valid when stimuli compete in the same brain.
**Location:** `research/experiments/information_theory/test_coding_capacity.py`

### Q10: Noise Robustness
**Status:** Test fixed but needs sparser parameters for meaningful validation
**Hypothesis:** Assemblies maintain information in the presence of realistic neural noise levels.
**Problem:** The original test kept providing stimulus, so recovery was trivial. The revised protocol uses autonomous recurrence, but at `k/n=0.10` it still passes too easily to be a good discriminator.
**Next Step:** Re-run with `N=10000`, `K=50`, `P=0.01` to test genuine noise resilience.
**Location:** `research/experiments/stability/test_noise_robustness_v2.py`

---

## Dynamics and Plasticity

### Q11: Convergence Time
**Status:** Completed
**Hypothesis:** Assemblies converge in `O(log N)` steps under typical conditions.
**Result:** Empirical convergence typically falls in the 2-6 step range in the faster tested conditions, with slower and noisier behavior in weaker regimes.
**Location:** `research/experiments/stability/test_scaling_laws.py`

### Q12: Learning Rules
**Status:** Completed - bounded retrieval result, broader theory still open
**Hypothesis:** Hebbian plasticity enables stable assembly modification without catastrophic forgetting.
**Result:** In the current quick retrieval test, mean retrieval accuracy is `1.000` with no early-vs-late degradation across sequentially trained stimuli.
**Caveat:** This supports a bounded "no catastrophic interference in this retrieval test" claim, but not yet a full learning-rules theory or capacity claim.
**Location:** `research/experiments/distinctiveness/test_capacity_limits.py`
**Results:** `research/results/distinctiveness/capacity_limits_20260208_171552_quick.json`

### Q13: Neural Oscillations
**Status:** Not Started
**Hypothesis:** Assembly dynamics generate emergent oscillations matching biological frequencies.
**Why It Matters:** Links computation to observed brain activity.
**Location:** `core_questions/Q13_neural_oscillations/`

---

## Critical Discoveries

### Q20: Competition and Distinctiveness
**Status:** Completed - strong phenomenon, mechanism still being decomposed
**Finding:** In the tested same-brain stim-only regime, assemblies remain near chance overlap while preserving perfect reactivation, indicating that shared-area competition can support distinct representations.
**Evidence:**
- Same brain, `n=1000, k=100`: pairwise overlap `0.104 - 0.121` vs chance `0.100`
- Reactivation fidelity: `1.000 +/- 0.000` across 2-8 stimuli
- Quick mechanism follow-up: low neuron reuse (about 5-7 percent), but not every simple explanatory hypothesis is supported yet
**Implication:** The distinctiveness phenomenon is strong; the full mechanism story still needs a sharper claim and decomposition.
**Location:** `research/core_questions/Q20_competition_distinctiveness/`

### Q21: Autonomous Recurrence
**Status:** Completed - resolved as a documentation gap
**Finding:** `brain.project({}, {area: [area]})` supports pure area-to-area self-recurrence. The confusion came from some higher-level helper paths excluding self-recurrence, not from the engine lacking the capability.
**Evidence:** LRI tests and sequence recall both use autonomous self-recurrence successfully.
**Location:** `neural_assemblies/tests/test_lri.py`, `neural_assemblies/tests/test_sequences.py`

### Q22: N400 = Global Pre-k-WTA Energy
**Status:** Completed - formalized claim plus supporting experiments
**Finding:** The N400 semantic priming effect in Assembly Calculus corresponds to `sum(all_inputs)` - total synaptic input before k-WTA selection - not to post-competition assembly overlap or neuron-specific activation.
**Evidence:**
- Global energy: `d = -25 to -31`, `p < 0.001` (correct N400 direction)
- Settling dynamics: `d = -16 to -18`, `p = 0.001` (correct direction)
- Thirteen prior post-k-WTA conditions showed reversed direction
- Neuron-specific pre-k-WTA metrics also reversed (competition effect)
**Implication:** Assembly Calculus can model aggregate neural signals (ERPs), not just single-cell recordings. The pre-/post-k-WTA boundary is fundamental.
**Location:** `research/core_questions/Q22_n400_global_energy/`, `research/claims/N400_GLOBAL_ENERGY.md`

---

## Implementation

### Q14: Multi-Backend Performance
**Status:** Not Started
**Hypothesis:** A unified API across CUDA, CuPy, and NumPy can achieve within 10 percent of optimal performance.
**Why It Matters:** Practical usability.
**Location:** `core_questions/Q14_multi_backend/`

### Q15: Quantization Effects
**Status:** Not Started
**Hypothesis:** FP16 quantization maintains accuracy while achieving 2-4x speedup.
**Why It Matters:** Further scaling potential.
**Location:** `core_questions/Q15_quantization/`

---

## Applications

### Q16: Language Processing
**Status:** In Progress
**Hypothesis:** Assembly Calculus can model syntactic structure formation.
**Why It Matters:** Tests computational expressiveness.
**Progress:** EmergentParser implements a 44-area parser with grounded vocabulary, role binding, incremental processing, and curriculum learning.
**Location:** `neural_assemblies/assembly_calculus/emergent/`, `research/experiments/applications/`

### Application Track: N400 Semantic Priming
**Status:** See Q22 above
**Note:** The N400 application result is now tracked under the single Q22 entry so the research tree does not duplicate the same question in multiple sections. Future application work should extend that question rather than restate it here.

### Q17: Memory Formation
**Status:** Not Started
**Hypothesis:** Transient assemblies can become persistent memories through plasticity.
**Why It Matters:** Links to real cognitive functions.
**Location:** `core_questions/Q17_memory/`

---

## Meta-Questions

### Q18: Validation Framework
**Status:** Partially Complete
**Question:** What constitutes valid evidence for each type of claim?
**Progress:** Critical analysis revealed several methodological flaws.
**Location:** `research/CRITICAL_ANALYSIS.md`

### Q19: Experimental Design
**Status:** In Progress
**Question:** What experiments can falsify our hypotheses?
**Progress:** Identified that the original noise-robustness test was unfalsifiable.
**Location:** `core_questions/Q19_experimental_design/`

---

## Question Status Legend

- **Not Started:** Question identified but not yet investigated
- **In Progress:** Active experimentation and analysis
- **Completed:** Bounded, evidence-backed conclusions reached
- **INVALID:** Test methodology was flawed
- **Abandoned:** Question did not pan out; document why

---

## Priority Questions (Updated)

### Tier 1 (Foundation) - Mostly Done
- Q01: Assembly Stability
- Q03: Scaling Laws
- Q11: Convergence Time
- Q20: Competition and Distinctiveness

### Tier 2 (Validation) - Next Priority
- Q02: Critical Phenomena
- Q12: Learning Rules
- Q06: Assembly Detection
- Q22: N400 = Global Pre-k-WTA Energy

### Tier 3 (Applications)
- Q16: Language Processing
- Q17: Memory Formation
- Q13: Neural Oscillations

---

## Summary Statistics

| Category | Total | Completed | In Progress | Issues |
|----------|-------|-----------|-------------|--------|
| Fundamental | 3 | 2 | 0 | 0 |
| Algorithmic | 2 | 0 | 0 | 0 |
| Biological | 3 | 1 | 0 | 0 |
| Information | 2 | 1 | 0 | 1 |
| Dynamics | 3 | 2 | 0 | 0 |
| Critical Discoveries | 3 | 3 | 0 | 0 |
| Implementation | 2 | 0 | 0 | 0 |
| Applications | 2 | 0 | 1 | 0 |
| Meta | 2 | 0 | 2 | 0 |

**Overall Progress:** 9 completed, 3 in progress, 1 issue, 9 not started

---

This list should evolve as the research matures. Update frequently, and keep the
strongest statements aligned with specific artifacts and limitations.
