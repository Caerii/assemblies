# Discrete Assembly Calculus - Deep Dive

## Core Concepts

### What is an Assembly?
A fixed set of k neurons that fire together in response to a stimulus.

```
Assembly = {neuron_1, neuron_2, ..., neuron_k}
```

### Key Parameters

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| n | `area.n` | Total neurons in area |
| k | `area.k` | Number of winners (assembly size) |
| w | `area.w` | Support (neurons that have ever fired) |
| p | `brain.p` | Connection probability |
| β | `area.beta` | Plasticity rate |

---

## The Primitives

### 1. PROJECTION (stimulus → area)

```python
brain.project({'STIM': ['AREA']}, {})
```

**Algorithm:**
```
1. Compute input to each neuron:
   input[i] = sum_j(w[j,i] * stim[j])

2. Select top-k:
   winners = argtop_k(input)

3. Update weights:
   w[j,i] *= (1 + β)  for j ∈ stim, i ∈ winners
```

**Effect:** Creates stable assembly for stimulus.

### 2. AREA-TO-AREA PROJECTION

```python
brain.project({}, {'AREA1': ['AREA2']})
```

**Algorithm:**
```
1. input[i] = sum_j(w[j,i] * winners1[j])
2. winners2 = argtop_k(input)
3. w[j,i] *= (1 + β)  for j ∈ winners1, i ∈ winners2
```

**Effect:** Transfers representation between areas.

### 3. ASSOCIATION (bidirectional)

```python
area1.fix_assembly()
area2.fix_assembly()
brain.project({}, {'AREA1': ['AREA2'], 'AREA2': ['AREA1']})
```

**Requires `fix_assembly()`** to prevent assemblies from changing during learning.

**Effect:** Creates bidirectional links between assemblies.

### 4. MERGE

```python
brain.project({}, {'AREA1': ['AREA3'], 'AREA2': ['AREA3']})
```

**Algorithm:**
```
input[i] = sum_j(w1[j,i] * winners1[j]) + sum_k(w2[k,i] * winners2[k])
winners3 = argtop_k(input)
```

**Effect:** Creates combined representation.

---

## Sparse Simulation

### The Problem
With n = 10,000 neurons, storing full weight matrix is O(n²) = 100M entries.

### The Solution: Support Tracking
Only track neurons that have ever fired.

```
w = support size (starts at 0)
w grows when new neurons fire
w never shrinks
```

### How It Works
```
1. Keep track of existing winners (indices 0 to w-1)
2. Sample potential NEW winners from unused neurons
3. Compare inputs: existing vs potential
4. Best k overall become winners
5. If new winner selected, expand support
```

### Memory Savings
```
Sparse: O(w × k) where w << n
Explicit: O(n²)

For n=10000, k=50, w=200:
  Sparse: ~10,000 entries
  Explicit: 100,000,000 entries
```

---

## What It CAN Do

### 1. Stable Pattern Formation ✅
```
Step 1: overlap = 0.54
Step 2: overlap = 0.98
Step 3+: overlap = 1.0 (converged)
```

### 2. Distinct Representations ✅
```
CAT assembly ∩ DOG assembly = 0%
Different stimuli → different assemblies
```

### 3. Cross-Area Association ✅ (with fix_assembly)
```
Learn: WORD ↔ IMAGE
Retrieve: IMAGE → WORD (96% accuracy)
```

### 4. Composition/Merge ✅
```
RED + APPLE → RED-APPLE assembly
```

### 5. High Capacity ✅
```
500+ assemblies in one area with <1% overlap
```

---

## What It CANNOT Do

### 1. Maintain Activity Without Input ❌
```
Assembly exists only while stimulus is active.
Remove stimulus → assembly vanishes (6% overlap after 1 step)
```

### 2. Pattern Completion ❌
```
Partial cue (50% of assembly) does not retrieve full pattern.
Need full stimulus.
```

### 3. Autonomous Sequences ❌
```
Cannot replay A→B→C without external prompting.
Each step needs explicit input.
```

### 4. Learn Temporal Order ❌
```
No timing information.
Must explicitly program sequences.
```

---

## Key Equations

### Input Computation
```
input_i = Σ_j w_ji × active_j
```

### Winner Selection
```
winners = argmax_k(input)
```

### Weight Update (Hebbian)
```
w_ji ← w_ji × (1 + β)  for active pairs
```

### Convergence
Assembly typically converges in 2-3 steps.

---

## The Fundamental Nature

**Assembly Calculus is a stimulus-response system, not a memory system.**

- Input present → Assembly active
- Input removed → Assembly gone
- It's a hash function, not an attractor

**Useful for:**
- Real-time pattern recognition
- Associations between modalities
- Compositional representations

**Not useful for:**
- Autonomous cognition
- Memory recall without cues
- Sequence generation

---

## Summary Table

| Feature | Status | Notes |
|---------|--------|-------|
| Pattern formation | ✅ | Converges in 2-3 steps |
| Distinctiveness | ✅ | 0% overlap between stimuli |
| Association | ✅ | Requires fix_assembly |
| Merge/Composition | ✅ | Works well |
| Capacity | ✅ | 500+ assemblies |
| Persistence | ❌ | Needs continuous input |
| Pattern completion | ❌ | Not supported |
| Autonomous sequences | ❌ | Not supported |

