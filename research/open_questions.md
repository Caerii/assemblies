# Open Research Questions

A living document tracking **all** questions being explored in this project. Questions are organized by theme and status.

*Last Updated: 2025-11-28*

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

### Q10: Noise Robustness ❌ NEEDS REWORK
**Status:** Original test INVALID - needs redesign  
**Hypothesis:** Assemblies maintain information in presence of realistic neural noise levels.  
**Problem:** Original test kept providing stimulus, trivially recovering assembly  
**Finding:** True autonomous recovery not supported by current implementation  
**Location:** `research/experiments/stability/test_noise_robustness_v2.py`  

---

## ⚡ Dynamics and Plasticity

### Q11: Convergence Time ✅ VALIDATED
**Status:** Completed  
**Hypothesis:** Assemblies converge in O(log N) steps under typical conditions.  
**Result:** Confirmed - 2-6 steps depending on parameters  
**Location:** `research/experiments/stability/test_scaling_laws.py`  

### Q12: Learning Rules
**Status:** Not Started  
**Hypothesis:** Hebbian plasticity enables stable assembly modification without catastrophic forgetting.  
**Why It Matters:** Learning and memory  
**Location:** `core_questions/Q12_learning_rules/`  

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

### Q21: Autonomous Recurrence Not Supported
**Status:** Discovered - Implementation Limitation  
**Finding:** Current brain.py doesn't support pure recurrent dynamics  
**Implication:** Can't test true attractor behavior without stimulus  
**Action Needed:** Extend implementation or use different approach  

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
**Status:** Not Started  
**Hypothesis:** Assembly calculus can model syntactic structure formation.  
**Why It Matters:** Tests computational expressiveness  
**Location:** `core_questions/Q16_language/`  

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
- 🔲 Q12: Learning Rules - Test plasticity
- 🔲 Q06: Assembly Detection - Compare to real data

### Tier 3 (Applications)
- 🔲 Q16: Language Processing
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
| Dynamics | 3 | 1 | 0 | 0 |
| New Discoveries | 2 | 2 | 0 | 0 |
| Implementation | 2 | 0 | 0 | 0 |
| Applications | 2 | 0 | 0 | 0 |
| Meta | 2 | 0 | 2 | 0 |

**Overall Progress:** 7 completed, 1 invalid, 13 remaining

---

**This list will grow as research progresses. Update frequently!**
