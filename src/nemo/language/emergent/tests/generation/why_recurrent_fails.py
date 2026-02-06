"""
WHY RECURRENT PROJECTION FAILS IN NEMO
======================================

Finding: At all scales and sparsities, accuracy ≈ 25% (random chance).

This is NOT a parameter tuning problem. It's a FUNDAMENTAL ARCHITECTURAL issue.

Let's understand WHY.
"""

import sys
import os
import cupy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.nemo.language.emergent.brain import EmergentNemoBrain
from src.nemo.language.emergent.areas import Area
from src.nemo.language.emergent.params import EmergentParams


def compute_overlap(a1, a2, k):
    if a1 is None or a2 is None:
        return 0.0
    s1 = set(a1.get().tolist())
    s2 = set(a2.get().tolist())
    return len(s1 & s2) / k


def analyze_weight_structure():
    """
    Analyze what happens to the weight matrix during learning.
    
    KEY INSIGHT: All episodes share the SAME weight matrix.
    """
    print("="*70)
    print("ANALYSIS: Weight Matrix Structure")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    CA3 = Area.SEQ
    
    # Create words
    words = {}
    for i, (name, area) in enumerate([
        ('dog', Area.NOUN_CORE), ('cat', Area.NOUN_CORE),
        ('runs', Area.VERB_CORE), ('sleeps', Area.VERB_CORE)
    ]):
        cp.random.seed((i + 1) * 1000)
        phon = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(area)
        for _ in range(20):
            brain._project(area, phon, learn=True)
        words[name] = (area, brain.current[area].copy())
    
    print("\n1. Before learning any episodes:")
    print(f"   (Weights are implicit/hash-based + learned deltas)")
    
    # Learn "dog runs"
    print("\n2. Learning 'dog runs'...")
    subj_asm = words['dog'][1]
    verb_asm = words['runs'][1]
    
    for _ in range(50):
        brain.clear_all()
        brain.current[Area.NOUN_CORE] = subj_asm.copy()
        brain.prev[Area.NOUN_CORE] = subj_asm.copy()
        brain.current[Area.VERB_CORE] = verb_asm.copy()
        brain.prev[Area.VERB_CORE] = verb_asm.copy()
        
        brain._project(CA3, subj_asm, learn=True)
        brain._project(CA3, verb_asm, learn=True)
        ca3_dog_runs = brain.current[CA3].copy()
        
        # Recurrent
        for _ in range(3):
            brain._project(CA3, brain.current[CA3], learn=True)
        
        # Reverse
        brain._project(Area.NOUN_CORE, brain.current[CA3], learn=True)
    
    print(f"   (Learned weights accumulated)")
    
    # Learn "cat sleeps"
    print("\n3. Learning 'cat sleeps'...")
    subj_asm = words['cat'][1]
    verb_asm = words['sleeps'][1]
    
    for _ in range(50):
        brain.clear_all()
        brain.current[Area.NOUN_CORE] = subj_asm.copy()
        brain.prev[Area.NOUN_CORE] = subj_asm.copy()
        brain.current[Area.VERB_CORE] = verb_asm.copy()
        brain.prev[Area.VERB_CORE] = verb_asm.copy()
        
        brain._project(CA3, subj_asm, learn=True)
        brain._project(CA3, verb_asm, learn=True)
        ca3_cat_sleeps = brain.current[CA3].copy()
        
        for _ in range(3):
            brain._project(CA3, brain.current[CA3], learn=True)
        
        brain._project(Area.NOUN_CORE, brain.current[CA3], learn=True)
    
    print(f"   (More learned weights accumulated)")
    
    # Check overlap of CA3 patterns
    print("\n4. CA3 pattern analysis:")
    overlap = compute_overlap(ca3_dog_runs, ca3_cat_sleeps, k)
    print(f"   dog_runs ∩ cat_sleeps: {overlap:.3f}")
    print(f"   Random expectation: {k/n:.3f}")
    
    # THE PROBLEM: Test what CA3 → NOUN_CORE retrieves
    print("\n5. THE PROBLEM - CA3 → NOUN_CORE retrieval:")
    
    # Query with dog_runs CA3 pattern
    brain._clear_area(Area.NOUN_CORE)
    brain._project(Area.NOUN_CORE, ca3_dog_runs, learn=False)
    retrieved_from_dog_runs = brain.current[Area.NOUN_CORE].copy()
    
    dog_overlap = compute_overlap(retrieved_from_dog_runs, words['dog'][1], k)
    cat_overlap = compute_overlap(retrieved_from_dog_runs, words['cat'][1], k)
    
    print(f"\n   From CA3[dog_runs]:")
    print(f"     → dog: {dog_overlap:.3f}")
    print(f"     → cat: {cat_overlap:.3f}")
    print(f"     Correct? {dog_overlap > cat_overlap}")
    
    # Query with cat_sleeps CA3 pattern
    brain._clear_area(Area.NOUN_CORE)
    brain._project(Area.NOUN_CORE, ca3_cat_sleeps, learn=False)
    retrieved_from_cat_sleeps = brain.current[Area.NOUN_CORE].copy()
    
    dog_overlap = compute_overlap(retrieved_from_cat_sleeps, words['dog'][1], k)
    cat_overlap = compute_overlap(retrieved_from_cat_sleeps, words['cat'][1], k)
    
    print(f"\n   From CA3[cat_sleeps]:")
    print(f"     → dog: {dog_overlap:.3f}")
    print(f"     → cat: {cat_overlap:.3f}")
    print(f"     Correct? {cat_overlap > dog_overlap}")
    
    print("\n" + "="*70)
    print("WHY THIS HAPPENS:")
    print("="*70)
    print("""
The CA3 → NOUN_CORE weights are SHARED across all episodes!

When we learn:
  - CA3[dog_runs] → NOUN_CORE: strengthens weights to 'dog' neurons
  - CA3[cat_sleeps] → NOUN_CORE: strengthens weights to 'cat' neurons

BUT these weights overlap because:
  - Both CA3 patterns have k=100 neurons
  - They share ~10 neurons (overlap ≈ k²/n)
  - Those shared neurons have weights to BOTH dog AND cat

So when we project CA3[dog_runs] → NOUN_CORE:
  - The 90 unique neurons activate 'dog'
  - The 10 shared neurons activate BOTH 'dog' AND 'cat'
  - Plus any general strengthening from training
  
Result: Both get activated, discrimination fails.
""")


def analyze_hopfield_vs_nemo():
    """
    Compare what a TRUE Hopfield network does vs what NEMO does.
    """
    print("\n" + "="*70)
    print("HOPFIELD vs NEMO: Fundamental Differences")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│              TRUE HOPFIELD NETWORK (CA3)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Structure:                                                         │
│  - N neurons with RECURRENT connections (W is NxN)                  │
│  - Each neuron connects to many others                              │
│  - Symmetric weights: W[i,j] = W[j,i]                               │
│                                                                     │
│  Learning (Hebbian):                                                │
│  - Present pattern x (binary, N-dimensional)                        │
│  - Update: W += x * x^T (outer product)                             │
│  - This creates ATTRACTORS at stored patterns                       │
│                                                                     │
│  Retrieval:                                                         │
│  - Initialize with partial/noisy pattern                            │
│  - Iterate: x = sign(W @ x)                                         │
│  - Network SETTLES to nearest attractor                             │
│  - Energy: E = -x^T @ W @ x (decreases until stable)                │
│                                                                     │
│  Key property: ATTRACTOR DYNAMICS                                   │
│  - Stored patterns are STABLE FIXED POINTS                          │
│  - Network naturally flows toward stored patterns                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│              NEMO PROJECTION                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Structure:                                                         │
│  - M source areas, N target areas                                   │
│  - Implicit random connections (hash-based)                         │
│  - Learned weight DELTAS on top of random                           │
│                                                                     │
│  Learning:                                                          │
│  - Project source → target                                          │
│  - Top-k selection (winner-take-all)                                │
│  - Strengthen connections from prev_active to new_active            │
│                                                                     │
│  "Recurrent" projection (area → same area):                         │
│  - NOT true recurrence!                                             │
│  - Each step: new_pattern = top_k(W @ old_pattern)                  │
│  - Pattern CHANGES each step (not settling)                         │
│  - No energy function, no attractors                                │
│                                                                     │
│  Key property: NO ATTRACTOR DYNAMICS                                │
│  - Patterns drift based on learned weights                          │
│  - No guarantee of convergence to stored pattern                    │
│  - Winner-take-all selects "most activated" not "most similar"      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

THE FUNDAMENTAL ISSUE:

NEMO's projection is NOT Hopfield dynamics.
- Hopfield: Energy minimization → stable attractors
- NEMO: Winner-take-all → drifting patterns

When we do "recurrent projection" in NEMO:
- We're not settling to an attractor
- We're just applying the same feedforward transform repeatedly
- The pattern drifts based on ALL learned weights, not just one episode
- With multiple episodes, weights are MIXED → interference

This is why accuracy ≈ 25% regardless of scale/sparsity.
The architecture fundamentally doesn't support attractor dynamics.
""")


def test_true_hopfield():
    """
    Implement a TRUE Hopfield network to show it DOES work.
    """
    print("\n" + "="*70)
    print("TRUE HOPFIELD NETWORK (Reference Implementation)")
    print("="*70)
    
    # Simple Hopfield implementation
    n = 1000  # Number of neurons
    k = 100   # Sparsity
    
    # Create sparse binary patterns
    def make_pattern(seed):
        np.random.seed(seed)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        return p
    
    # Patterns to store
    dog = make_pattern(1)
    cat = make_pattern(2)
    runs = make_pattern(3)
    sleeps = make_pattern(4)
    
    # Episode patterns (outer product style)
    # In sparse Hopfield, we typically store the pattern itself
    dog_runs = np.clip(dog + runs, 0, 1)  # OR combination
    cat_sleeps = np.clip(cat + sleeps, 0, 1)
    
    print(f"\n1. Pattern statistics:")
    print(f"   dog_runs active neurons: {dog_runs.sum()}")
    print(f"   cat_sleeps active neurons: {cat_sleeps.sum()}")
    print(f"   Overlap: {(dog_runs * cat_sleeps).sum()}")
    
    # Hopfield weight matrix
    W = np.zeros((n, n))
    
    # Store patterns using Hebbian rule
    # W += (2*p - 1) @ (2*p - 1).T for bipolar
    # For sparse binary, we use: W += p @ p.T
    W += np.outer(dog_runs, dog_runs)
    W += np.outer(cat_sleeps, cat_sleeps)
    np.fill_diagonal(W, 0)  # No self-connections
    
    print(f"\n2. Weight matrix:")
    print(f"   Non-zero weights: {(W != 0).sum()}")
    print(f"   Weight range: [{W.min():.2f}, {W.max():.2f}]")
    
    # Retrieval function
    def hopfield_retrieve(W, query, k_active, steps=20):
        """Hopfield retrieval with sparse threshold."""
        x = query.copy()
        for step in range(steps):
            # Compute input
            h = W @ x
            # Sparse threshold: keep top-k
            threshold = np.sort(h)[-k_active] if k_active < len(h) else h.min()
            x_new = (h >= threshold).astype(float)
            # Check convergence
            if np.array_equal(x_new, x):
                print(f"   Converged at step {step}")
                break
            x = x_new
        return x
    
    # Test retrieval
    print(f"\n3. Testing retrieval:")
    
    # Query with "runs" only
    print(f"\n   Query: 'runs' pattern")
    retrieved = hopfield_retrieve(W, runs, k_active=150, steps=50)
    
    dog_overlap = (retrieved * dog).sum() / dog.sum()
    cat_overlap = (retrieved * cat).sum() / cat.sum()
    runs_overlap = (retrieved * runs).sum() / runs.sum()
    sleeps_overlap = (retrieved * sleeps).sum() / sleeps.sum()
    
    print(f"   Retrieved overlaps:")
    print(f"     dog: {dog_overlap:.3f}")
    print(f"     cat: {cat_overlap:.3f}")
    print(f"     runs: {runs_overlap:.3f}")
    print(f"     sleeps: {sleeps_overlap:.3f}")
    print(f"   'dog' > 'cat'? {'YES ✓' if dog_overlap > cat_overlap else 'NO ✗'}")
    
    # Query with "sleeps" only
    print(f"\n   Query: 'sleeps' pattern")
    retrieved = hopfield_retrieve(W, sleeps, k_active=150, steps=50)
    
    dog_overlap = (retrieved * dog).sum() / dog.sum()
    cat_overlap = (retrieved * cat).sum() / cat.sum()
    
    print(f"   Retrieved overlaps:")
    print(f"     dog: {dog_overlap:.3f}")
    print(f"     cat: {cat_overlap:.3f}")
    print(f"   'cat' > 'dog'? {'YES ✓' if cat_overlap > dog_overlap else 'NO ✗'}")


def propose_solution():
    """
    Propose what would be needed for true hippocampal realism in NEMO.
    """
    print("\n" + "="*70)
    print("WHAT WOULD BE NEEDED FOR TRUE HIPPOCAMPAL REALISM")
    print("="*70)
    
    print("""
To implement TRUE CA3-like memory in NEMO, we would need:

OPTION 1: Implement True Hopfield Dynamics in NEMO
─────────────────────────────────────────────────────
- Add a new area type: HopfieldArea
- Store patterns using outer product rule: W += p ⊗ p
- Retrieval uses energy minimization, not winner-take-all
- This is a significant architectural change

OPTION 2: Sparse Distributed Memory (SDM)
─────────────────────────────────────────────────────
- Create "hard locations" (random reference patterns)
- Write: activate nearby locations, store data
- Read: activate nearby locations, sum data
- More biologically plausible than Hopfield
- Also requires architectural changes

OPTION 3: Accept the Limitation, Use Explicit Storage
─────────────────────────────────────────────────────
- NEMO cannot do CA3-like pattern completion
- Store (episode, components) explicitly
- Use neural overlap for matching
- This is what we've been doing!

OPTION 4: Hybrid - Neural Index + Explicit Storage
─────────────────────────────────────────────────────
- Episode patterns (neural, emergent)
- Component storage (explicit, like hippocampal index)
- Matching via neural overlap (emergent)
- ~85% neural, scientifically defensible

THE HONEST ASSESSMENT:
─────────────────────────────────────────────────────
NEMO's projection-based architecture is fundamentally different
from Hopfield/CA3 attractor networks.

To achieve TRUE neurobiological realism for memory, we would need
to EXTEND NEMO with new area types that support attractor dynamics.

This is possible but requires significant new implementation.

For now, Option 4 (Hybrid) is the most practical path that:
- Uses NEMO's strengths (assembly formation, projection, learning)
- Acknowledges its limitation (no attractor dynamics)
- Achieves working memory with mostly neural operations
""")


if __name__ == "__main__":
    analyze_weight_structure()
    analyze_hopfield_vs_nemo()
    test_true_hopfield()
    propose_solution()

