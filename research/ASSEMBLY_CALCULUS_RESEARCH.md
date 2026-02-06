# Assembly Calculus - Research Summary

## Source: Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (PNAS 2020)

## Core Concept

**Assemblies** are large populations of neurons (hundreds to thousands) whose synchronous firing represents:
- Memories
- Concepts  
- Words
- Other cognitive elements

The hypothesis: Assemblies are the fundamental units of brain computation.

---

## The Five Operations

### 1. PROJECTION
Copy an assembly from one brain area to another.

```
Area A has assembly x
Project A → B
Now B has assembly y that corresponds to x
```

**Mechanism:** Neurons in B that receive most input from x become the new assembly.

### 2. RECIPROCAL PROJECTION
Bidirectional projection that creates mutual connections.

```
Project A ↔ B
Now A.x and B.y are linked in both directions
```

### 3. ASSOCIATION
Increase overlap between two assemblies in the same area.

```
Assemblies x and y in area A
Repeatedly co-activate x and y
Now x and y share more neurons
```

**Mechanism:** Hebbian learning strengthens shared connections.

### 4. MERGE
Combine two assemblies into a new one.

```
Assembly x in area A
Assembly y in area B
Project both to area C
C gets new assembly z = merge(x, y)
```

**Use case:** Combining subject + verb in language.

### 5. PATTERN COMPLETION
Activate full assembly from partial input.

```
Assembly x has neurons {1, 2, 3, ..., 100}
Activate only {1, 2, 3, ..., 50}
Recurrent connections complete to full x
```

**This is the controversial one!**

---

## Key Claims from Literature

### Claim 1: Turing Complete
"The Assembly Calculus can perform arbitrary computations."

The operations are sufficient to implement any algorithm.

### Claim 2: Biologically Plausible
- Uses Hebbian plasticity ("neurons that fire together wire together")
- Uses inhibition (winner-take-all)
- Works with random connectivity
- No special wiring required

### Claim 3: Explains Language
The MERGE operation maps to linguistic merge (Chomsky's Minimalist Program).

```
[the] + [cat] → [the cat]
[the cat] + [sat] → [the cat sat]
```

### Claim 4: Pattern Completion Works
From partial activation, recurrent connections restore full pattern.

---

## What We Found in Our Implementation

### Works ✅
1. **Projection** - Assemblies form reliably
2. **Association** - Cross-area linking works (with fix_assembly)
3. **Merge** - Combination works
4. **Distinct representations** - Different stimuli → different assemblies
5. **High capacity** - 500+ assemblies per area

### Doesn't Work ❌
1. **Pattern completion** - Partial cues don't restore full pattern
2. **Persistence** - Assemblies vanish without input
3. **Autonomous recurrence** - Self-projection destroys pattern

---

## The Gap: Theory vs Implementation

### What the Theory Says
"Pattern completion is realizable by generic, randomly connected populations of neurons with Hebbian plasticity and inhibition."

### What We Observe
```
After training with stimulus + recurrence:
  30% cue → 39% recovery
  50% cue → 39% recovery  
  70% cue → 35% recovery
  90% cue → 36% recovery
```

Pattern completion is NOT working in our implementation.

### Possible Reasons

1. **Sparse simulation limitation**
   - The sparse algorithm samples NEW neurons on each projection
   - This prevents true recurrent stabilization

2. **Missing inhibition dynamics**
   - Theory requires lateral inhibition
   - Implementation uses instant winner-take-all
   - No gradual competition dynamics

3. **Weight update rule**
   - Theory: Hebbian with proper normalization
   - Implementation: Simple multiplicative update
   - No LTD (long-term depression)

4. **Recurrent weight training**
   - Pattern completion requires strong INTRA-assembly connections
   - Our training (stim + recurrence) may not strengthen these enough

---

## The Original Paper's Approach

From the PNAS paper, pattern completion requires:

1. **Strong recurrent connections within assembly**
   - Neurons in assembly should excite each other

2. **Inhibition between assemblies**
   - Prevents other patterns from activating

3. **Proper balance**
   - Excitation within assembly > inhibition
   - Inhibition between assemblies > random noise

Our implementation may be missing these dynamics.

---

## Key Parameters from Literature

| Parameter | Typical Value | Meaning |
|-----------|---------------|---------|
| n | 10^5 - 10^6 | Neurons per area |
| k | √n ≈ 300-1000 | Assembly size |
| p | 0.01 - 0.1 | Connection probability |
| β | 0.01 - 0.1 | Plasticity rate |

The choice k ≈ √n is theoretically motivated for optimal capacity.

---

## Open Questions

1. **Why doesn't pattern completion work in our implementation?**
   - Is it a bug or a fundamental limitation?

2. **Is the sparse simulation compatible with pattern completion?**
   - Sparse simulation samples new neurons
   - This may prevent attractor dynamics

3. **What modifications would enable true attractors?**
   - Explicit areas?
   - Different learning rule?
   - Proper inhibition?

4. **How does the original codebase handle this?**
   - The simulations.py has pattern completion code
   - But it uses `b.areas` which is not populated in current brain.py

---

## Comparison to Other Models

| Model | Pattern Completion | Attractors | Biological |
|-------|-------------------|------------|------------|
| Hopfield Network | ✅ | ✅ | ❌ |
| Assembly Calculus (theory) | ✅ | ✅ | ✅ |
| Assembly Calculus (our impl) | ❌ | ❌ | ✅ |
| Modern Hopfield | ✅ | ✅ | ❌ |

Our implementation is biologically plausible but lacks attractor dynamics.

---

## Next Steps for Research

1. **Fix the areas/area_by_name inconsistency**
   - simulations.py uses `b.areas`
   - brain.py only populates `area_by_name`

2. **Test with explicit areas**
   - Full weight matrix
   - May enable true recurrence

3. **Study the original paper's simulation code**
   - What parameters did they use?
   - How did they achieve pattern completion?

4. **Consider alternative implementations**
   - Spiking version
   - Continuous-time dynamics

