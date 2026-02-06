# Critical Scientific Analysis of Assembly Calculus Implementation

**Date:** November 29, 2025

## Executive Summary

After rigorous testing, we have identified **fundamental limitations** in what the Assembly Calculus sparse simulation actually computes. Many of our previous "validations" were either tautological or tested the wrong thing.

---

## Critical Finding 1: Assemblies Are NOT Attractors

### The Claim
Assembly Calculus is often described as creating "attractor states" where assemblies are stable patterns stored in the weight matrix.

### The Reality
**Self-projection destroys assemblies immediately.**

```
After creating assembly with 50 rounds:
  Support (w): 55
  Winners: [0, 1, 2, 3, 4, 5, ...]

After T → T self-projection:
  Round 1: w = 55 → 105, overlap = 0.0%
  Round 2: w = 105 → 154, overlap = 2.0%
  Round 3: w = 154 → 200, overlap = 2.0%
```

### Why This Happens
The sparse simulation algorithm:
1. Samples "potential new winners" from neurons that have NEVER fired
2. These new neurons win (no competition from existing assembly)
3. Support (w) grows indefinitely
4. Original assembly is completely lost

### Implication
**Assemblies only exist while the stimulus is active.** They are not stored in weights as attractors. The stimulus is doing all the computational work.

---

## Critical Finding 2: Pattern Completion Does Not Work

### The Claim
If we activate half an assembly, the weights should "complete" the pattern.

### The Reality
Starting with 50% of assembly 1:
```
After T→T projection:
  Round 1: overlap with assembly 1 = 4%
  Round 2: overlap with assembly 1 = 6%
  Round 3: overlap with assembly 1 = 4%
```

**The weights do NOT guide activation back to the original assembly.**

### Implication
This is not a content-addressable memory. Partial cues do not retrieve stored patterns.

---

## Critical Finding 3: Biological Validation Was Tautological

### What We Did
1. Set `n_neurons = target_assembly / target_sparsity`
2. Set `k_active = target_assembly`
3. Checked if `k/n` matches biological sparsity

### The Problem
This is circular! We designed parameters to match, then "validated" they match.

### What Real Validation Would Require
- **Emergent sparsity** from dynamics (not prescribed k)
- Matching firing rate **distributions**, not just means
- Matching **temporal correlation structure**
- Matching **trial-to-trial variability**

---

## Critical Finding 4: "Phase Transition" is Just Combinatorics

### What We Found
At n/k < 6, assemblies collapse to 100% overlap.

### Why This is NOT a Phase Transition
If n=250 and k=50, creating 5 assemblies:
- Total neurons needed: 5 × 50 = 250
- Available neurons: 250
- Of course they overlap!

A real phase transition would show:
- Critical exponents
- Power-law distributions
- Diverging correlation lengths
- Universal behavior

**We found a capacity limit, not a phase transition.**

---

## Critical Finding 5: Language/Syntax Test Was Too Easy

### What We Tested
Can we project words to areas and merge them?

### What This Proves
- Assembly Calculus can store distinct patterns ✓
- Assembly Calculus can combine patterns ✓

### What This Does NOT Prove
- Correct syntactic parsing
- Subject-verb agreement
- Long-distance dependencies
- Garden path recovery
- Ambiguity resolution

**We built a fixed structure by hand, not a parser.**

---

## What Assembly Calculus Actually Computes

### Stimulus → Area Projection
This is **associative learning**:
1. Stimulus neurons have fixed connections to area
2. Repeated projection strengthens specific pathways
3. Winners become deterministic given stimulus
4. Assembly = set of neurons that reliably fire for this stimulus

### Cross-Area Association (with fix_assembly)
This is **heteroassociative memory**:
1. Fix assembly X, fix assembly Y
2. Project X↔Y to strengthen cross-area weights
3. Later, Y can retrieve X via learned weights

**But X cannot self-sustain without Y driving it.**

### What It's NOT
- Attractor network (assemblies don't self-sustain)
- Content-addressable memory (partial cues don't work)
- Autonomous recurrent dynamics

---

## Revised Assessment

| Claim | Status | Reality |
|-------|--------|---------|
| Assemblies are attractors | ❌ FALSE | Destroyed by self-projection |
| Pattern completion works | ❌ FALSE | Partial cues don't retrieve |
| Biological validation | ❌ TAUTOLOGICAL | Parameters designed to match |
| Phase transition at n/k≈6 | ⚠️ MISLEADING | Just capacity limit |
| Language/syntax works | ⚠️ MISLEADING | Hand-built structure, not parsing |
| Cross-area retrieval | ✅ TRUE | But requires fix_assembly |
| Distinct assemblies | ✅ TRUE | Competition creates distinctiveness |
| No catastrophic forgetting | ✅ TRUE | Stimulus-driven retrieval works |

---

## What Assembly Calculus IS Good For

1. **Stimulus-driven pattern formation**: Given input, create stable output pattern
2. **Heteroassociative memory**: Link patterns across areas
3. **Competitive coding**: Multiple stimuli create distinct representations
4. **Compositional structure**: Merge operation combines patterns

---

## What Assembly Calculus is NOT Good For

1. **Autonomous dynamics**: Cannot maintain activity without input
2. **Attractor memory**: Patterns not stored as stable states
3. **Pattern completion**: Partial cues don't work
4. **True recurrence**: Self-projection destroys patterns

---

## Recommendations

### For Future Research
1. **Implement explicit areas** for attractor dynamics (dense simulation)
2. **Test with real neural data** (not parameter matching)
3. **Build actual parser** with incremental input
4. **Study theoretical limits** of sparse simulation

### For Publications
1. Be precise about what "assembly" means
2. Don't claim attractor dynamics without testing
3. Acknowledge fix_assembly limitation
4. Distinguish stimulus-driven vs autonomous

---

---

## Root Cause Analysis: Why No Attractors?

### The Core Problem

When we project S→T (stimulus to area), the Hebbian learning updates **S→T weights**, not T→T weights.

When we then project T→T (self-recurrence), the system:
1. Computes inputs from current T winners
2. Selects NEW winners based on these inputs
3. Updates T→T weights for the NEW winners

**The original assembly neurons are not favored** because:
- T→T weights were never strengthened for them during S→T training
- The winner-take-all selection picks whoever gets most input NOW
- Random connectivity means non-assembly neurons get similar input

### Numerical Evidence

```
After 20 rounds of S→T:
  Internal (asm→asm) weights: mean = 0.20
  External (non-asm→asm) weights: mean = 0.20

After T→T projection:
  Overlap with original: 0%
```

The internal and external weights are IDENTICAL because S→T training doesn't strengthen T→T connections.

### What Would Be Needed for True Attractors

1. **Simultaneous S→T and T→T training**: When S activates T, the T→T weights between co-active neurons should also strengthen.

2. **Asymmetric weight growth**: Internal weights need to grow FASTER than external weights.

3. **Inhibition**: Non-assembly neurons need to be actively suppressed, not just lose the competition.

### This is a Design Limitation, Not a Bug

The current implementation follows the Assembly Calculus paper's specification. The paper focuses on **projection** and **association** as the core operations, not autonomous recurrence.

The theoretical framework may support attractors, but this implementation does not realize them.

---

## Conclusion

Assembly Calculus (as implemented in this sparse simulation) is a **stimulus-driven heteroassociative memory**, not an attractor network. The assemblies exist only while inputs are active. This is still useful for modeling certain cognitive functions, but claims about attractor dynamics and pattern completion are not supported by the implementation.

### What This Implementation CAN Do
- Form stable assemblies given consistent input ✅
- Associate assemblies across areas ✅
- Maintain distinct representations via competition ✅
- Compose assemblies via merge ✅

### What This Implementation CANNOT Do
- Maintain assemblies without input ❌
- Complete patterns from partial cues ❌
- Function as content-addressable memory ❌
- Support autonomous recurrent dynamics ❌

The framework may be theoretically sound, but the implementation has fundamental limitations that prevent true recurrent dynamics. Future work should investigate whether modifying the learning rule (e.g., simultaneous intra-area plasticity) could enable attractor behavior.

