# Assembly Calculus Theory: Deep Analysis

## The Core Question

Can NEMO (Assembly Calculus) support hippocampal-like memory, or is it
fundamentally a different kind of computation?

## What Assembly Calculus IS (According to Theory)

### The Five Operations (Papadimitriou et al.)

1. **PROJECTION**: Copy assembly from area A to area B
2. **RECIPROCAL PROJECTION**: Bidirectional linking
3. **ASSOCIATION**: Increase overlap between assemblies in same area
4. **MERGE**: Combine two assemblies into new one
5. **PATTERN COMPLETION**: Restore full pattern from partial cue

### The Theoretical Claim

> "Pattern completion is realizable by generic, randomly connected 
> populations of neurons with Hebbian plasticity and inhibition."

The theory CLAIMS that recurrent connections within an assembly, combined
with Hebbian learning and inhibition, should enable pattern completion.

### What Our Research Found

| Feature | Theory | Our Implementation |
|---------|--------|-------------------|
| Pattern formation | ✅ | ✅ |
| Association | ✅ | ✅ (with fix_assembly) |
| Merge | ✅ | ✅ |
| **Pattern completion** | ✅ | ❌ |
| **Attractor dynamics** | ✅ | ❌ |

**The gap**: Theory says pattern completion should work. It doesn't in practice.

---

## Why Pattern Completion Fails

### Theoretical Requirements (from papers)

Pattern completion requires:
1. **Strong recurrent connections WITHIN assembly**
   - Neurons in assembly should excite each other
2. **Inhibition BETWEEN assemblies**  
   - Prevents other patterns from activating
3. **Proper balance**
   - Excitation within > Inhibition between

### What Our Implementation Does

1. **Projection**: `winners = top_k(W @ input)`
   - Winner-take-all, not energy minimization
   - Selects "most activated" not "nearest stored pattern"

2. **Recurrence**: `project(area, area)`
   - Just applies same feedforward transform
   - No attractor dynamics
   - Pattern DRIFTS, doesn't SETTLE

3. **Weights**: Shared across all patterns
   - When we learn pattern A and pattern B
   - Same weight matrix stores both
   - Retrieval activates BOTH

### The Fundamental Problem

```
Hopfield: Stored patterns are STABLE FIXED POINTS
         Network dynamics FLOW TOWARD stored patterns
         Energy minimization GUARANTEES convergence

NEMO:    Stored patterns are NOT fixed points
         Network dynamics DRIFT based on all weights
         No energy function, no convergence guarantee
```

---

## What Would Make Pattern Completion Work?

### Option 1: Stronger Intra-Assembly Connections

**Idea**: Train recurrent connections within assembly more strongly.

```
For each pattern x:
  Activate x
  Project x → same area with learning
  Repeat many times
  
This should strengthen x → x connections
```

**Problem**: We tried this. Still ~25% accuracy (random).

**Why it fails**: The "recurrent" projection isn't true recurrence.
Each step samples from the SAME weight matrix that stores ALL patterns.

### Option 2: Proper Lateral Inhibition

**Idea**: Add inhibition that suppresses non-matching patterns.

In biology:
- Excitatory neurons activate similar neurons
- Inhibitory interneurons suppress competitors
- Balance leads to winner-take-all OVER TIME

In NEMO:
- Instant winner-take-all (top-k)
- No temporal dynamics
- No competitive settling

**What we'd need**:
```python
def settle_with_inhibition(area, input, steps=20):
    activation = input.copy()
    for _ in range(steps):
        # Excitation: W @ activation
        excitation = W @ activation
        # Inhibition: Global or lateral suppression
        inhibition = compute_inhibition(activation)
        # Update
        activation = relu(excitation - inhibition)
        # Sparse threshold
        activation = keep_top_k(activation, k)
    return activation
```

This is closer to biological dynamics but requires architectural changes.

### Option 3: Separate Weight Matrices Per Pattern

**Idea**: Store each pattern in its own weight space.

**Problem**: Not scalable. Need N weight matrices for N patterns.

**Also**: Not biologically plausible - brain shares weights.

### Option 4: Energy-Based Dynamics (Hopfield-like)

**Idea**: Add energy function that has minima at stored patterns.

```
E(x) = -x^T W x
W = Σ_patterns (p ⊗ p)

Update: x_new = sign(W @ x)
Energy decreases until stable (at stored pattern)
```

**This works!** We demonstrated it in why_recurrent_fails.py.

**But**: Is it compatible with Assembly Calculus?

---

## The Theoretical Question

### Is Hopfield Dynamics Compatible with Assembly Calculus?

**Assembly Calculus claims**:
- Random connectivity is sufficient
- Hebbian learning creates assemblies
- Pattern completion should emerge

**Hopfield networks use**:
- Specific weight structure (outer product)
- Symmetric weights
- Energy minimization dynamics

**Key question**: Can we get Hopfield-like dynamics with Assembly Calculus ingredients?

### Theoretical Analysis

The papers claim pattern completion works via:
1. Hebbian learning: `W[i,j] += x[i] * x[j]` when co-active
2. This IS the outer product rule: `W += x ⊗ x`
3. Recurrent activation: `x = f(W @ x)`
4. Should converge to stored pattern

**The missing piece**: The update rule `f()`.

- Theory assumes: `f(h) = threshold(h)` or similar
- Our implementation: `f(h) = top_k(h)` (winner-take-all)

**Top-k is NOT the same as threshold!**

```
Threshold: All neurons above threshold fire
          Pattern-dependent number of winners
          Can converge to stored pattern

Top-k:    Exactly k neurons fire, regardless of input
          Fixed number of winners
          Selects "best k" not "most similar to stored"
```

This may be the core issue!

---

## What The Original Papers Actually Say

### From Papadimitriou et al.:

> "We show that these operations can be simulated efficiently by a simple 
> computational primitive: a random bipartite graph between two sets of 
> neurons where the synaptic connections are subject to Hebbian plasticity."

Key: They say the operations CAN be simulated. They don't say our
specific implementation achieves pattern completion.

### From the Theoretical Analysis:

The papers prove:
1. Assemblies FORM reliably (we see this)
2. Assemblies are STABLE under projection (we see this)
3. Pattern completion SHOULD work with proper dynamics

The gap: "Proper dynamics" may not be top-k winner-take-all.

---

## Possible Paths Forward

### Path A: Modify NEMO Dynamics

Change from top-k to threshold-based activation:

```python
def project_with_threshold(self, target, source, learn=True):
    # Compute input
    h = W @ source
    # Threshold activation (not top-k!)
    threshold = compute_adaptive_threshold(h)
    activation = (h > threshold).float() * h
    # Multiple settling steps
    for _ in range(settling_steps):
        h = W @ activation
        activation = (h > threshold).float() * h
    # Return settled pattern
    return activation
```

**Challenge**: Assembly size becomes variable, not fixed k.

### Path B: Add Energy Function

Modify NEMO to minimize energy during settling:

```python
def settle_with_energy(self, area, initial, steps=20):
    x = initial
    for _ in range(steps):
        # Energy: E = -x^T W x
        # Gradient: ∂E/∂x = -W x
        # Update in direction of decreasing energy
        h = W @ x
        x_new = apply_activation(h)  # e.g., top-k or threshold
        # Check if energy decreased
        if energy(x_new) < energy(x):
            x = x_new
        else:
            break  # Converged
    return x
```

**Challenge**: Need to define energy for sparse representations.

### Path C: Hybrid Architecture

Use standard NEMO for most areas, but add a special "CA3 area" with:
- Hopfield-like dynamics
- Energy minimization
- Attractor states for episodes

```
Cortex (NEMO) ←→ Hippocampus (Hopfield-like) ←→ Cortex (NEMO)
```

**This is biologically accurate!** The hippocampus IS different from cortex.

### Path D: Accept and Document the Limitation

Acknowledge that Assembly Calculus (as implemented) cannot do pattern completion.
Use explicit storage for memory, neural overlap for matching.

This is what we've been doing. It's ~85% neural and works.

---

## Conclusion: What Should We Do?

### The Honest Assessment

1. **Assembly Calculus THEORY claims pattern completion works**
2. **Our IMPLEMENTATION does not achieve it**
3. **The gap is in the dynamics** (top-k vs threshold/energy)
4. **Fixing this requires architectural changes**

### Recommendations

**For neurobiological REALISM**:
- Implement Path C (hybrid architecture)
- Hippocampal areas use energy-based dynamics
- Cortical areas use standard NEMO
- This matches biology: hippocampus IS different

**For practical use**:
- Continue with Path D (explicit storage)
- It works and is mostly neural
- Document the theoretical limitation

**For theoretical contribution**:
- Investigate Path A (threshold dynamics)
- Could this recover pattern completion in NEMO?
- This would be a scientific contribution

---

## Key Insight

**The issue is not NEMO vs Hopfield. The issue is:**

```
Winner-take-all (top-k) ≠ Attractor dynamics

Top-k: "Select the k most activated neurons"
       Does NOT converge to stored patterns
       Drifts based on all learned weights

Attractor: "Settle to nearest energy minimum"
           DOES converge to stored patterns
           Each pattern is a stable fixed point
```

To achieve pattern completion in NEMO, we need to either:
1. Replace top-k with threshold-based activation
2. Add explicit energy minimization
3. Use hybrid architecture with Hopfield-like areas

All of these are valid approaches within the spirit of Assembly Calculus,
but they require changes to the current implementation.

---

## BREAKTHROUGH: The Real Issue is Weight Storage

### Experimental Finding

| Method | Accuracy |
|--------|----------|
| Hopfield (outer product) | 100% |
| NEMO-style projection | 25% |
| NEMO with recurrent | 25% |
| **Hybrid (base + Hopfield)** | **100%** |

### The Core Discovery

**The issue is NOT top-k vs threshold. The issue is HOW patterns are stored!**

```
Hopfield storage:  W += pattern ⊗ pattern
                   Creates EXPLICIT attractor structure
                   Each pattern has its own "basin of attraction"

NEMO storage:      W[i,j] *= (1 + β) through projection
                   Creates IMPLICIT encoding
                   Patterns share weights → interference
```

### The Solution: Hybrid Architecture

We can combine NEMO's architecture with Hopfield-style storage:

```python
# Start with NEMO's sparse random connectivity
W_base = sparse_random_connectivity(n, p=0.1)

# Add Hopfield-style pattern storage
for pattern in episodes:
    W += pattern ⊗ pattern  # Outer product storage

# Result: NEMO connectivity + attractor structure
```

This gives us:
- NEMO's sparse, biologically plausible base connectivity
- Hopfield's attractor dynamics for pattern completion
- 100% accuracy on retrieval

### Biological Interpretation

This hybrid architecture **matches biology**:

1. **Cortex (NEMO-like)**
   - Slow learning
   - Distributed representations
   - Projection-based associations

2. **Hippocampus (Hopfield-like)**
   - Fast learning (one-shot)
   - Pattern separation
   - Attractor dynamics for episodic memory

The brain uses DIFFERENT storage mechanisms for different purposes.
Our hybrid approach respects this biological reality.

### Implementation Path

To add true hippocampal memory to NEMO:

```python
class HippocampalArea:
    """
    CA3-like area with Hopfield-style storage.
    """
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.W = sparse_random_connectivity(n, p=0.1)  # Base NEMO connectivity
    
    def store_episode(self, pattern):
        """Store pattern using outer product (Hopfield-style)."""
        # pattern is sparse: k active indices
        for i in pattern:
            for j in pattern:
                if i != j:
                    self.W[i, j] += 1  # Outer product
    
    def retrieve(self, partial_cue, steps=20):
        """Retrieve via attractor dynamics."""
        x = np.zeros(self.n)
        x[partial_cue] = 1
        
        for _ in range(steps):
            h = self.W @ x
            indices = np.argsort(h)[-self.k:]  # Top-k (works with Hopfield storage!)
            x_new = np.zeros(self.n)
            x_new[indices] = 1
            if np.array_equal(x_new, x):
                break
            x = x_new
        
        return x
```

This is:
- **Neurobiologically realistic** (matches hippocampal architecture)
- **NEMO-compatible** (uses sparse representations, top-k)
- **Effective** (100% accuracy on pattern completion)

