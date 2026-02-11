# Open Research Questions

A living document tracking **all** questions being explored in this project. Questions are organized by theme and status.

*Last Updated: 2026-02-11*

---

## 🎯 Fundamental Theory

### Q01: Assembly Stability ✅ VALIDATED
**Status:** Completed  
**Hypothesis:** Sparse assemblies converge to stable attractors under specific connectivity and sparsity conditions.  
**Result:** Confirmed - 100% convergence rate, mean 2.375 steps to stability  
**Location:** `research/experiments/primitives/test_projection.py`  

### Q02: Critical Phenomena
**Status:** Not Started  
**Hypothesis:** Assembly formation exhibits phase transitions at critical sparsity levels.  
**Why It Matters:** Explains robustness and universality  
**Location:** `core_questions/Q02_critical_phenomena/`  

### Q03: Scaling Laws ✅ VALIDATED (needs theory)
**Status:** Completed - needs theoretical derivation  
**Hypothesis:** Assembly properties follow power-law scaling with network size.  
**Result:** O(log N) scaling confirmed with R²=0.989  
**Note:** Convergence time DECREASES with N (negative exponent) - needs explanation  
**Location:** `research/experiments/stability/test_scaling_laws.py`  

---

## 📊 Algorithmic Contributions

### Q04: Complexity Bounds
**Status:** Not Started  
**Hypothesis:** Top-K selection can be performed in O(N log K) time with appropriate data structures.  
**Why It Matters:** Enables billion-scale simulation  
**Location:** `core_questions/Q04_complexity_bounds/`  

### Q05: Memory Efficiency
**Status:** Not Started  
**Hypothesis:** Sparse representations reduce memory by >80% without accuracy loss.  
**Why It Matters:** Practical scalability  
**Location:** `core_questions/Q05_memory_efficiency/`  

---

## 🧠 Biological Validation

### Q06: Assembly Detection
**Status:** Not Started  
**Hypothesis:** Computational assemblies can be detected in biological neural recordings.  
**Why It Matters:** Validates theoretical predictions  
**Location:** `core_questions/Q06_assembly_detection/`  

### Q07: Sparsity Measurements ✅ VALIDATED
**Status:** Completed  
**Hypothesis:** Biological neural networks exhibit predicted sparsity levels (1-5%).  
**Result:** Simulation parameters (2% sparsity) match biological ranges  
**Location:** `research/experiments/biological_validation/test_biological_parameters.py`  

### Q08: Cross-Species Universality
**Status:** Not Started  
**Hypothesis:** Assembly principles are universal across species and brain regions.  
**Why It Matters:** Tests generality of theory  
**Location:** `core_questions/Q08_universality/`  

---

## 📈 Information Theory

### Q09: Coding Capacity ✅ VALIDATED*
**Status:** Completed (*with important caveat)  
**Hypothesis:** Sparse assembly coding is near-optimal for biological energy/information constraints.  
**Result:** 0.282 bits/neuron achieved  
**CRITICAL CAVEAT:** Only valid when stimuli compete in same brain!  
**Location:** `research/experiments/information_theory/test_coding_capacity.py`  

### Q10: Noise Robustness ⚠️ PARTIALLY VALIDATED
**Status:** Test fixed but needs sparser parameters for meaningful validation
**Hypothesis:** Assemblies maintain information in presence of realistic neural noise levels.
**Problem:** Original test kept providing stimulus, trivially recovering assembly.
Test was fixed to use autonomous recurrence (area→area self-projection), but at
k/n=0.10 the test passes trivially. Needs sparser parameters (k/n < 0.01) to
be a meaningful discriminator.
**Next Step:** Re-run with N=10000, K=50, P=0.01 to test genuine noise resilience
**Location:** `research/experiments/stability/test_noise_robustness_v2.py`  

---

## ⚡ Dynamics and Plasticity

### Q11: Convergence Time ✅ VALIDATED
**Status:** Completed  
**Hypothesis:** Assemblies converge in O(log N) steps under typical conditions.  
**Result:** Confirmed - 2-6 steps depending on parameters  
**Location:** `research/experiments/stability/test_scaling_laws.py`  

### Q12: Learning Rules ✅ VALIDATED
**Status:** Completed — no catastrophic interference detected
**Hypothesis:** Hebbian plasticity enables stable assembly modification without catastrophic forgetting.
**Result:** Mean retrieval accuracy = 1.000, no early degradation across trials.
Tested via capacity limits experiment with sequential assembly formation and retrieval.
**Location:** `research/experiments/distinctiveness/test_capacity_limits.py`
**Results:** `research/results/distinctiveness/capacity_limits_20260208_171552_quick.json`  

### Q13: Neural Oscillations
**Status:** Not Started  
**Hypothesis:** Assembly dynamics generate emergent oscillations matching biological frequencies.  
**Why It Matters:** Links computation to observed brain activity  
**Location:** `core_questions/Q13_neural_oscillations/`  

---

## 🔬 NEW: Critical Discoveries

### Q20: Competition is Essential ✅ MAJOR FINDING
**Status:** Completed - CRITICAL DISCOVERY  
**Finding:** Assembly distinctiveness requires competition within same brain  
**Evidence:**
- Separate brains: 94% overlap (NOT distinct)
- Same brain: 3% overlap (distinct)
**Implication:** Lateral inhibition is essential mechanism, not optional  
**Location:** `research/experiments/stability/test_assembly_distinctiveness.py`  

### Q21: Autonomous Recurrence ✅ RESOLVED (documentation gap)
**Status:** Completed — was a documentation gap, not an implementation limitation
**Finding:** `brain.project({}, {area: [area]})` supports pure area→area
self-recurrence. This always worked — the confusion was that the `project()`
*op* in `ops.py` uses `project_rounds()` which excludes self-recurrence, but
calling `brain.project()` directly with explicit area→area routing works.
**Evidence:** LRI tests (test_lri.py) and sequence recall (test_sequences.py)
both use autonomous self-recurrence successfully.
**Location:** `src/tests/test_lri.py`, `src/tests/test_sequences.py`

### Q22: N400 = Global Pre-k-WTA Energy ✅ MAJOR FINDING
**Status:** Completed — mechanism identified after 16 experimental conditions
**Finding:** The N400 semantic priming effect in Assembly Calculus corresponds
to `sum(all_inputs)` — total synaptic input before k-WTA selection — NOT to
post-competition assembly overlap or neuron-specific activation.
**Evidence:**
- Global energy: d = -25 to -31, p < 0.001 (correct N400 direction)
- Settling dynamics: d = -16 to -18, p = 0.001 (correct direction)
- 13 prior post-k-WTA conditions all showed reversed direction
- Neuron-specific pre-k-WTA metrics also reversed (competition effect)
**Implication:** AC can model aggregate neural signals (ERPs), not just
single-cell recordings. The pre-/post-k-WTA boundary is fundamental.
**Location:** `research/experiments/applications/test_n400_pre_kwta.py`

---

## 🔧 Implementation

### Q14: Multi-Backend Performance
**Status:** Not Started  
**Hypothesis:** Unified API across CUDA/CuPy/NumPy achieves within 10% of optimal performance.  
**Why It Matters:** Practical usability  
**Location:** `core_questions/Q14_multi_backend/`  

### Q15: Quantization Effects
**Status:** Not Started  
**Hypothesis:** FP16 quantization maintains accuracy while achieving 2-4x speedup.  
**Why It Matters:** Further scaling potential  
**Location:** `core_questions/Q15_quantization/`  

---

## 🎓 Applications

### Q16: Language Processing
**Status:** In Progress
**Hypothesis:** Assembly calculus can model syntactic structure formation.
**Why It Matters:** Tests computational expressiveness
**Progress:** EmergentParser implements 44-area parser with grounded vocabulary,
role binding, incremental processing, and curriculum learning.
**Location:** `src/assembly_calculus/emergent/`, `research/experiments/applications/`

### Q22: N400 Semantic Priming ✅ MAJOR FINDING
**Status:** Completed — core mechanism identified
**Hypothesis:** Assembly calculus can reproduce the N400 ERP (semantic priming facilitation).
**Finding:** The N400 maps to **global pre-k-WTA energy** (`sum(all_inputs)` before
winner-take-all selection), NOT to post-competition assembly overlap or neuron-specific
activation. Related primes reduce global energy via Hebbian pre-activation of shared
features; unrelated primes leave global energy higher.
**Evidence:**
- Global energy: Cohen's d = -25 to -31, p < 0.001 (CORRECT direction)
- Settling dynamics: Cohen's d = -16 to -18, p = 0.001 (CORRECT direction)
- Neuron-specific metrics: REVERSED (competition effect, as expected)
- 13 post-k-WTA conditions across 3 prior experiments all showed reversed direction,
  confirming the k-WTA boundary is the critical distinction
**Insight:** The N400 ERP is a scalp-recorded aggregate signal reflecting total
cortical energy, not single-neuron activity. `sum(all_inputs)` is the AC analogue.
**Next Steps:** Parameter robustness, cloze probability, sentence context, P600.
See `research/plans/N400_NEXT_STEPS.md`.
**Location:** `research/experiments/applications/test_n400_pre_kwta.py`

### Q17: Memory Formation
**Status:** Not Started  
**Hypothesis:** Transient assemblies can become persistent memories through plasticity.  
**Why It Matters:** Links to real cognitive functions  
**Location:** `core_questions/Q17_memory/`  

---

## 📝 Meta-Questions

### Q18: Validation Framework
**Status:** Partially Complete  
**Question:** What constitutes valid evidence for each type of claim?  
**Progress:** Critical analysis revealed several methodological flaws  
**Location:** `research/CRITICAL_ANALYSIS.md`  

### Q19: Experimental Design
**Status:** In Progress  
**Question:** What experiments can falsify our hypotheses?  
**Progress:** Identified that noise robustness test was unfalsifiable  
**Location:** `core_questions/Q19_experimental_design/`  

---

## 🚧 Question Status Legend

- **Not Started:** Question identified but not yet investigated
- **In Progress:** Active experimentation and analysis
- **Completed:** Validated conclusions reached
- **INVALID:** Test methodology was flawed
- **Abandoned:** Question didn't pan out (document why)

---

## 🎯 Priority Questions (Updated)

### Tier 1 (Foundation) - MOSTLY DONE
- ✅ Q01: Assembly Stability - VALIDATED
- ✅ Q03: Scaling Laws - VALIDATED (needs theory)
- ✅ Q11: Convergence Time - VALIDATED
- ✅ Q20: Competition Essential - MAJOR FINDING

### Tier 2 (Validation) - NEXT PRIORITY
- 🔲 Q02: Critical Phenomena - Map phase transitions
- ✅ Q12: Learning Rules - VALIDATED (no catastrophic interference)
- 🔲 Q06: Assembly Detection - Compare to real data
- ✅ Q22: N400 Semantic Priming - MAJOR FINDING (global pre-k-WTA energy)

### Tier 3 (Applications)
- 🔄 Q16: Language Processing - In Progress (EmergentParser)
- 🔲 Q17: Memory Formation
- 🔲 Q13: Neural Oscillations

---

## 📊 Summary Statistics

| Category | Total | Completed | In Progress | Issues |
|----------|-------|-----------|-------------|--------|
| Fundamental | 3 | 2 | 0 | 0 |
| Algorithmic | 2 | 0 | 0 | 0 |
| Biological | 3 | 1 | 0 | 0 |
| Information | 2 | 1 | 0 | 1 |
| Dynamics | 3 | 2 | 0 | 0 |
| New Discoveries | 2 | 2 | 0 | 0 |
| Implementation | 2 | 0 | 0 | 0 |
| Applications | 3 | 1 | 1 | 0 |
| Meta | 2 | 0 | 2 | 0 |

**Overall Progress:** 10 completed, 0 invalid, 12 remaining

---

**This list will grow as research progresses. Update frequently!**
