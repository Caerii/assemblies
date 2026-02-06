"""
Can NEMO Primitives Alone Achieve Hippocampal Memory?
=====================================================

QUESTION: Do we NEED outer product storage, or can we achieve the same
effect using NEMO's existing primitives?

NEMO Primitives:
1. PROJECT(A → B): Create assembly in B from A, Hebbian learning
2. RECIPROCAL_PROJECT(A ↔ B): Bidirectional projection
3. ASSOCIATION: Link assemblies in same area
4. MERGE: Combine assemblies from different areas
5. PATTERN_COMPLETION: (controversial - doesn't work in our impl)

KEY INSIGHT: What does outer product W += p ⊗ p actually mean?
- For each pair (i, j) in pattern p, strengthen W[i,j]
- This is EXACTLY what reciprocal projection A ↔ A should do!

HYPOTHESIS: Reciprocal projection within the same area (self-projection)
should create attractor structure IF done correctly.

Let's test this systematically.
"""

import sys
import os
import cupy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from nemo.language.emergent.brain import EmergentNemoBrain
from nemo.language.emergent.areas import Area
from nemo.language.emergent.params import EmergentParams


def compute_overlap(a1, a2, k):
    if a1 is None or a2 is None:
        return 0.0
    s1 = set(a1.get().tolist())
    s2 = set(a2.get().tolist())
    return len(s1 & s2) / k


def test_self_projection():
    """
    Test if self-projection (area → same area) creates attractor structure.
    
    Protocol:
    1. Create pattern in CA3
    2. Project CA3 → CA3 repeatedly with learning
    3. This should strengthen intra-pattern connections
    4. Test if partial cue → full pattern
    """
    print("="*70)
    print("TEST 1: Self-Projection for Attractor Formation")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    CA3 = Area.SEQ
    
    # Create two patterns
    print("\n1. Creating patterns...")
    patterns = {}
    for name, seed in [('pattern1', 1), ('pattern2', 2)]:
        cp.random.seed(seed * 1000)
        # Create pattern through projection (NEMO style)
        stim = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(CA3)
        for _ in range(20):
            brain._project(CA3, stim, learn=True)
        patterns[name] = brain.current[CA3].copy()
        print(f"   {name}: {k} neurons")
    
    overlap = compute_overlap(patterns['pattern1'], patterns['pattern2'], k)
    print(f"   Initial overlap: {overlap:.3f}")
    
    # Store patterns using self-projection
    print("\n2. Storing patterns via self-projection...")
    
    for name, pattern in patterns.items():
        # Activate pattern and do self-projection
        for epoch in range(100):
            brain.clear_all()
            brain.current[CA3] = pattern.copy()
            brain.prev[CA3] = pattern.copy()
            
            # Self-projection: CA3 → CA3 with learning
            # This strengthens connections from pattern neurons to pattern neurons
            brain._project(CA3, pattern, learn=True)
        
        print(f"   {name}: 100 self-projection epochs")
    
    # Test retrieval with partial cue
    print("\n3. Testing retrieval with 50% partial cue...")
    
    for name, pattern in patterns.items():
        # Create 50% partial cue
        pattern_indices = pattern.get().tolist()
        partial_indices = pattern_indices[:len(pattern_indices)//2]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        # Activate partial cue
        brain.clear_all()
        brain.current[CA3] = partial_cue
        brain.prev[CA3] = partial_cue
        
        # Self-projection to complete pattern (retrieval)
        for _ in range(10):
            brain._project(CA3, brain.current[CA3], learn=False)
        
        retrieved = brain.current[CA3]
        
        # Check overlap with original
        overlap_target = compute_overlap(retrieved, pattern, k)
        overlap_other = compute_overlap(retrieved, 
            patterns['pattern2'] if name == 'pattern1' else patterns['pattern1'], k)
        
        print(f"   {name}: target={overlap_target:.3f}, other={overlap_other:.3f}, "
              f"correct={'YES' if overlap_target > overlap_other else 'NO'}")


def test_fix_assembly_self_projection():
    """
    Test if fix_assembly() + self-projection creates better attractors.
    
    The idea: fix_assembly() prevents the pattern from drifting during learning.
    This should strengthen EXACTLY the pattern neurons.
    """
    print("\n" + "="*70)
    print("TEST 2: fix_assembly() + Self-Projection")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    CA3 = Area.SEQ
    
    # Create patterns
    print("\n1. Creating and fixing patterns...")
    patterns = {}
    for name, seed in [('pattern1', 1), ('pattern2', 2)]:
        cp.random.seed(seed * 1000)
        stim = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(CA3)
        for _ in range(20):
            brain._project(CA3, stim, learn=True)
        patterns[name] = brain.current[CA3].copy()
        
        # Store in brain's learned assemblies (this is like fix_assembly)
        brain.store_learned_assembly(CA3, name, patterns[name])
        print(f"   {name}: stored as fixed assembly")
    
    # Learn with fixed assemblies
    print("\n2. Learning with fixed assemblies...")
    
    for name, pattern in patterns.items():
        for epoch in range(100):
            brain.clear_all()
            # Set BOTH current and prev to the fixed pattern
            brain.current[CA3] = pattern.copy()
            brain.prev[CA3] = pattern.copy()
            
            # Project with learning, but target will be computed
            # Since prev = pattern, Hebbian strengthens pattern → winners
            brain._project(CA3, pattern, learn=True)
        
        print(f"   {name}: 100 epochs with fixed source")
    
    # Test retrieval
    print("\n3. Testing retrieval...")
    
    for name, pattern in patterns.items():
        pattern_indices = pattern.get().tolist()
        partial_indices = pattern_indices[:len(pattern_indices)//2]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[CA3] = partial_cue
        brain.prev[CA3] = partial_cue
        
        for _ in range(10):
            brain._project(CA3, brain.current[CA3], learn=False)
        
        retrieved = brain.current[CA3]
        
        overlap_target = compute_overlap(retrieved, pattern, k)
        overlap_other = compute_overlap(retrieved, 
            patterns['pattern2'] if name == 'pattern1' else patterns['pattern1'], k)
        
        print(f"   {name}: target={overlap_target:.3f}, other={overlap_other:.3f}, "
              f"correct={'YES' if overlap_target > overlap_other else 'NO'}")


def test_additive_learning():
    """
    Test if additive learning (vs multiplicative) would help.
    
    Current NEMO: W *= (1 + β)  [multiplicative]
    Hopfield:     W += α        [additive]
    
    Can we simulate additive learning within NEMO?
    """
    print("\n" + "="*70)
    print("TEST 3: Simulating Additive Learning")
    print("="*70)
    
    print("""
    NEMO uses multiplicative Hebbian: W[i,j] *= (1 + β)
    Hopfield uses additive: W[i,j] += α
    
    For pattern completion, additive creates better attractor structure.
    
    Can we achieve additive-like effect with NEMO?
    
    Option 1: Many small multiplicative updates ≈ additive
              W * (1+β)^n ≈ W + n*β*W for small β
    
    Option 2: Initialize with uniform weights, then multiplicative
              If W[i,j] = 1 initially, W *= (1+β) = W + β
              
    Option 3: Implement additive learning explicitly (modify NEMO)
    """)
    
    # Let's test Option 1: Many small updates
    brain = EmergentNemoBrain(verbose=False)
    
    # Modify beta to be very small
    original_beta = brain.p.beta
    brain.p.beta = 0.01  # Very small learning rate
    
    k = brain.p.k
    n = brain.p.n
    CA3 = Area.SEQ
    
    print("\n1. Testing with small β (approximating additive)...")
    print(f"   β = {brain.p.beta} (was {original_beta})")
    
    patterns = {}
    for name, seed in [('pattern1', 1), ('pattern2', 2)]:
        cp.random.seed(seed * 1000)
        stim = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(CA3)
        for _ in range(20):
            brain._project(CA3, stim, learn=True)
        patterns[name] = brain.current[CA3].copy()
    
    # Many epochs with small beta
    print("\n2. Training with many epochs (500)...")
    for name, pattern in patterns.items():
        for epoch in range(500):
            brain.clear_all()
            brain.current[CA3] = pattern.copy()
            brain.prev[CA3] = pattern.copy()
            brain._project(CA3, pattern, learn=True)
    
    # Test
    print("\n3. Testing retrieval...")
    for name, pattern in patterns.items():
        pattern_indices = pattern.get().tolist()
        partial_indices = pattern_indices[:len(pattern_indices)//2]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[CA3] = partial_cue
        brain.prev[CA3] = partial_cue
        
        for _ in range(20):
            brain._project(CA3, brain.current[CA3], learn=False)
        
        retrieved = brain.current[CA3]
        
        overlap_target = compute_overlap(retrieved, pattern, k)
        overlap_other = compute_overlap(retrieved, 
            patterns['pattern2'] if name == 'pattern1' else patterns['pattern1'], k)
        
        print(f"   {name}: target={overlap_target:.3f}, other={overlap_other:.3f}, "
              f"correct={'YES' if overlap_target > overlap_other else 'NO'}")


def test_association_primitive():
    """
    Test if the ASSOCIATION primitive can create attractors.
    
    ASSOCIATION: "Increase overlap between two assemblies in the same area"
    
    What if we associate a pattern with ITSELF?
    This should strengthen intra-pattern connections.
    """
    print("\n" + "="*70)
    print("TEST 4: Using ASSOCIATION Primitive")
    print("="*70)
    
    print("""
    ASSOCIATION in NEMO: Link two assemblies in the same area.
    
    Key idea: What if we "associate" a pattern with itself?
    - The pattern neurons become more strongly connected to each other
    - This creates attractor structure
    
    This is conceptually: W[pattern, pattern] increases
    Which is exactly what we need!
    """)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    CA3 = Area.SEQ
    
    patterns = {}
    for name, seed in [('pattern1', 1), ('pattern2', 2)]:
        cp.random.seed(seed * 1000)
        stim = cp.random.randint(0, n, k, dtype=cp.uint32)
        brain._clear_area(CA3)
        for _ in range(20):
            brain._project(CA3, stim, learn=True)
        patterns[name] = brain.current[CA3].copy()
    
    print("\n1. Creating 'self-association' (pattern associated with itself)...")
    
    for name, pattern in patterns.items():
        # Self-association: repeatedly activate pattern and strengthen internal connections
        for epoch in range(100):
            brain.clear_all()
            # Activate pattern
            brain.current[CA3] = pattern.copy()
            brain.prev[CA3] = pattern.copy()
            
            # "Associate" by projecting back and forth
            # This simulates: pattern ↔ pattern (bidirectional)
            brain._project(CA3, pattern, learn=True)
            
            # Also do reverse: set new state as prev, project again
            brain.prev[CA3] = brain.current[CA3].copy()
            brain._project(CA3, pattern, learn=True)
        
        print(f"   {name}: self-associated")
    
    print("\n2. Testing retrieval...")
    for name, pattern in patterns.items():
        pattern_indices = pattern.get().tolist()
        partial_indices = pattern_indices[:len(pattern_indices)//2]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[CA3] = partial_cue
        brain.prev[CA3] = partial_cue
        
        for _ in range(10):
            brain._project(CA3, brain.current[CA3], learn=False)
        
        retrieved = brain.current[CA3]
        
        overlap_target = compute_overlap(retrieved, pattern, k)
        overlap_other = compute_overlap(retrieved, 
            patterns['pattern2'] if name == 'pattern1' else patterns['pattern1'], k)
        
        print(f"   {name}: target={overlap_target:.3f}, other={overlap_other:.3f}, "
              f"correct={'YES' if overlap_target > overlap_other else 'NO'}")


def analyze_the_problem():
    """
    Deep analysis of what's happening.
    """
    print("\n" + "="*70)
    print("DEEP ANALYSIS: Why NEMO Primitives Might Not Be Enough")
    print("="*70)
    
    print("""
THE CORE ISSUE:

NEMO's projection: winners = top_k(W @ input)

When we project pattern → same area:
1. Compute h = W @ pattern
2. Select winners = top_k(h)
3. Hebbian: strengthen pattern → winners

THE PROBLEM: winners ≠ pattern!

Even with self-projection, the top-k neurons receiving most input
from the pattern are NOT necessarily the pattern neurons themselves.

Why? Because:
- The random base connectivity sends pattern signal to MANY neurons
- Some non-pattern neurons might receive more total input
- Top-k selects these non-pattern neurons as winners

WHAT HOPFIELD DOES DIFFERENTLY:

Hopfield storage: W[i,j] += pattern[i] * pattern[j]
Only strengthens connections BETWEEN pattern neurons.

Hopfield retrieval: x = sign(W @ x) or threshold
Doesn't force exactly k winners.
Pattern neurons collectively have more input from each other.

THE FUNDAMENTAL GAP:

NEMO: Top-k selection is GLOBAL competition
      Best k neurons overall win, regardless of pattern membership

Hopfield: Each neuron decides LOCALLY based on its input
          Pattern neurons have strong mutual excitation, win together

Can we bridge this gap with NEMO primitives?

POSSIBLE SOLUTION: Use multiple areas to "lock in" the pattern.

Instead of: pattern → CA3 → CA3 (loses pattern identity)

Do: pattern → CORTEX (lock) → CA3 → CORTEX (decode)

The CORTEX area preserves the pattern identity.
The CA3 area stores associations.
The reverse projection CORTEX → CA3 → CORTEX maintains integrity.

This is actually more biologically realistic!
Hippocampus doesn't store patterns in isolation.
It stores associations BETWEEN cortical patterns.
""")


def test_cortex_mediated_storage():
    """
    Test storage with cortex as anchor.
    
    Protocol:
    1. Create pattern in CORTEX (stable)
    2. Project CORTEX → CA3 (encode)
    3. Strengthen CA3 → CORTEX (decode path)
    4. Retrieval: partial → CA3 → CORTEX → decode
    """
    print("\n" + "="*70)
    print("TEST 5: Cortex-Mediated Hippocampal Storage")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    CORTEX = Area.NOUN_CORE
    CA3 = Area.SEQ
    
    print("\n1. Creating patterns in CORTEX...")
    patterns = {}
    ca3_patterns = {}
    
    for name, seed in [('pattern1', 1), ('pattern2', 2)]:
        cp.random.seed(seed * 1000)
        stim = cp.random.randint(0, n, k, dtype=cp.uint32)
        
        # Create stable cortex pattern
        brain._clear_area(CORTEX)
        for _ in range(30):
            brain._project(CORTEX, stim, learn=True)
        patterns[name] = brain.current[CORTEX].copy()
        
        # Store in brain for later
        brain.store_learned_assembly(CORTEX, name, patterns[name])
        print(f"   {name} in CORTEX: {k} neurons")
    
    print("\n2. Encoding in CA3 with bidirectional learning...")
    
    for name, pattern in patterns.items():
        for epoch in range(50):
            brain.clear_all()
            
            # Activate cortex pattern
            brain.current[CORTEX] = pattern.copy()
            brain.prev[CORTEX] = pattern.copy()
            
            # Project CORTEX → CA3
            brain._project(CA3, pattern, learn=True)
            ca3_state = brain.current[CA3].copy()
            
            # Project CA3 → CORTEX (reverse path)
            brain._project(CORTEX, ca3_state, learn=True)
        
        # Store CA3 pattern
        ca3_patterns[name] = brain.current[CA3].copy()
        print(f"   {name}: CORTEX ↔ CA3 bidirectional trained")
    
    print("\n3. Testing retrieval: partial CA3 → CORTEX → decode...")
    
    for name, pattern in patterns.items():
        # Create partial CA3 cue
        ca3_full = ca3_patterns[name]
        ca3_indices = ca3_full.get().tolist()
        partial_indices = ca3_indices[:len(ca3_indices)//2]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        brain.clear_all()
        
        # Activate partial CA3
        brain.current[CA3] = partial_cue
        brain.prev[CA3] = partial_cue
        
        # Project CA3 → CORTEX
        brain._clear_area(CORTEX)
        brain._project(CORTEX, partial_cue, learn=False)
        
        retrieved_cortex = brain.current[CORTEX]
        
        # Check overlap with original cortex patterns
        overlap_target = compute_overlap(retrieved_cortex, pattern, k)
        overlap_other = compute_overlap(retrieved_cortex, 
            patterns['pattern2'] if name == 'pattern1' else patterns['pattern1'], k)
        
        print(f"   {name}: target={overlap_target:.3f}, other={overlap_other:.3f}, "
              f"correct={'YES' if overlap_target > overlap_other else 'NO'}")


if __name__ == "__main__":
    test_self_projection()
    test_fix_assembly_self_projection()
    test_additive_learning()
    test_association_primitive()
    analyze_the_problem()
    test_cortex_mediated_storage()

