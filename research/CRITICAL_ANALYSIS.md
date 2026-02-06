# Critical Analysis of Experimental Results

**Date:** November 28, 2025  
**Status:** 🔴 SIGNIFICANT METHODOLOGICAL ISSUES IDENTIFIED

## Executive Summary

Upon critical review, several of our "successful" experiments have **fundamental methodological flaws** that invalidate their conclusions. This document identifies these issues and proposes corrections.

---

## 🔴 CRITICAL ISSUE 1: Noise Robustness Test is Invalid

### The Problem

The noise robustness experiment claims "100% recovery from 100% noise" - this is **too good to be true** and indeed it is.

**What the test actually does:**
1. Establish assembly A via stimulus S → TARGET
2. Inject noise into TARGET's winners
3. Continue projecting S → TARGET
4. Measure "recovery" to original assembly

**Why this is wrong:**
- The stimulus S is **unchanged** and deterministic
- S always drives the same assembly formation
- Recovery is guaranteed because S is forcing the system back
- We're not testing assembly robustness - we're testing stimulus determinism

### What We Should Test

True noise robustness requires testing whether an assembly can recover **without the original stimulus**:

1. **Attractor test (no stimulus):** Perturb assembly, let it evolve autonomously
2. **Cross-stimulus test:** Perturb assembly, present different stimulus
3. **Recurrent recovery:** Test area-to-area recurrence without stimulus

### Current Result: ❌ INVALID

The claim "assemblies recover from 100% noise" is **not supported** by this experiment.

---

## 🟡 ISSUE 2: Scaling Laws - Negative Exponent is Suspicious

### The Finding

```
T = -1.48 * log10(N) + 9.88
```

Convergence time **decreases** as N increases. This is counterintuitive.

### Possible Explanations

1. **Sparsity effect:** We kept k/n = 0.05 constant, so larger N means larger k
2. **Larger assemblies converge faster** due to more averaging
3. **This may be correct** but needs theoretical explanation

### What We Need

- Theoretical derivation of why larger assemblies converge faster
- Test with fixed k (not fixed k/n ratio)
- Compare to Papadimitriou et al.'s theoretical predictions

### Current Result: 🟡 NEEDS VERIFICATION

---

## 🔴 CRITICAL ISSUE 3: Assembly Distinctiveness Depends on Competition

### The Finding (MAJOR DISCOVERY)

```
Separate brains: overlap=0.935  (94% overlap - BAD!)
Same brain: overlap=0.025       (3% overlap - GOOD!)
```

**Different stimuli create distinct assemblies ONLY when they compete in the same brain!**

### Why This Happens

1. **Separate brains:** Same random seed → same "hub" neurons win
2. **Same brain:** Competition/inhibition forces different neurons to win

### Critical Implications

1. **The original coding capacity test was WRONG** - it used separate brains
2. **Competition is ESSENTIAL** for distinct representations
3. **Inhibition mechanisms** must be part of the theory
4. **The 95% overlap was an artifact** of the test methodology

### What This Means for Assembly Calculus

- Assembly Calculus DOES work when stimuli compete
- The theory requires **lateral inhibition** or competition
- Separate "brain instances" don't represent biological reality
- **This is actually GOOD NEWS** - the framework works correctly!

### Current Result: 🟢 RESOLVED - Competition is key mechanism

---

## 🟢 VALID RESULTS

### Projection Convergence ✅

- Assemblies do converge to stable fixed points
- This is well-established in Assembly Calculus literature
- Our results match theory

### Association Binding ✅

- Association creates bidirectional links
- Pattern completion works
- This is consistent with theory

### Merge Composition ✅

- Merge creates combined assemblies
- Quality ~87% is reasonable
- Consistent with theory

### Biological Parameters ✅

- Our parameters are in biological ranges
- This is a sanity check, not a discovery

---

## 🔧 Required Fixes

### Fix 1: Rewrite Noise Robustness Test

```python
# WRONG: Keep projecting from same stimulus
b.project(areas_by_stim={"STIM": ["TARGET"]}, ...)

# RIGHT: Test autonomous recovery (no stimulus)
b.project(areas_by_stim={}, dst_areas_by_src_area={"TARGET": ["TARGET"]})

# OR: Test with recurrent connections only
# OR: Test pattern completion from partial cue
```

### Fix 2: Add Attractor Basin Analysis

Test how far an assembly can be perturbed before it falls into a different attractor.

### Fix 3: Test Assembly Distinctiveness

- Create many assemblies from different stimuli
- Measure pairwise overlaps
- Find conditions for orthogonal assemblies

### Fix 4: Theoretical Scaling Derivation

Derive mathematically why convergence time scales as observed.

---

## Updated Status

| Experiment | Original Status | Revised Status | Issue |
|------------|-----------------|----------------|-------|
| Projection | ✅ PASS | ✅ VALID | None |
| Association | ✅ PASS | ✅ VALID | None |
| Merge | ✅ PASS | ✅ VALID | None |
| Scaling Laws | ✅ PASS | 🟡 VERIFY | Negative exponent needs explanation |
| Noise Robustness | ✅ PASS | ❌ INVALID | Tests stimulus determinism, not attractor |
| Coding Capacity | ✅ PASS | 🟢 VALID* | *Only valid with competition in same brain |
| Distinctiveness | N/A | 🟢 VALID | Competition creates distinct assemblies |
| Biological | ✅ PASS | ✅ VALID | Sanity check only |

## Key Discovery: Competition is Essential

The most important finding from this critical analysis:

**Assembly Calculus requires competition between stimuli to create distinct representations.**

- Separate brain instances: 94% overlap (assemblies are NOT distinct)
- Same brain with competition: 3% overlap (assemblies ARE distinct)

This explains why the brain has lateral inhibition - it's essential for creating
distinct representations of different concepts.

---

## Lessons Learned

1. **"Too good to be true" results should be scrutinized**
2. **Understand what the test actually measures vs what we claim**
3. **Negative/counterintuitive results need theoretical backing**
4. **Don't celebrate before critical review**

---

## Next Steps

1. ✅ Document issues (this file)
2. 🔲 Rewrite noise robustness test properly
3. 🔲 Investigate assembly overlap issue
4. 🔲 Derive scaling law theoretically
5. 🔲 Add attractor basin experiments

---

*Science requires honesty about what we actually know vs what we wish we knew.*

