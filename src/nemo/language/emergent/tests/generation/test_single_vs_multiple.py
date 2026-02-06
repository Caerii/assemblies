"""
HYPOTHESIS: Pattern completion works for SINGLE assembly,
but DISCRIMINATION between multiple assemblies is a different problem.

The Assembly Calculus theory says:
"Pattern completion is realizable by generic, randomly connected 
populations of neurons with Hebbian plasticity and inhibition."

This might mean: Given partial assembly A, complete to full A.
NOT: Given partial A, distinguish from B, C, D.

Let's test this hypothesis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import cupy as cp
import numpy as np
from nemo.language.emergent.brain import EmergentNemoBrain
from nemo.language.emergent.areas import Area
from nemo.language.emergent.params import EmergentParams

def compute_overlap(a1, a2, k):
    s1 = set(a1.get().tolist())
    s2 = set(a2.get().tolist())
    return len(s1 & s2) / k


def test_single_assembly_completion():
    """
    Test pattern completion with ONLY ONE assembly in the area.
    No discrimination needed - just complete the pattern.
    """
    print("="*60)
    print("TEST 1: Single Assembly Pattern Completion")
    print("="*60)
    
    params = EmergentParams()
    params.n = 10000
    params.k = 100
    params.p = 0.1
    params.beta = 1.0
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    k = params.k
    n = params.n
    AREA = Area.NOUN_CORE
    
    print(f"\nParameters: n={n}, k={k}, p={params.p}, beta={params.beta}")
    
    # Create ONE stable assembly
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(200):
        brain._project(AREA, stimulus, learn=True)
    stable_assembly = brain.current[AREA].copy()
    
    print(f"\n1. Created stable assembly with {k} neurons")
    
    # Build intra-assembly connections
    for _ in range(500):
        brain.clear_all()
        brain.current[AREA] = stable_assembly.copy()
        brain.prev[AREA] = stable_assembly.copy()
        brain._project(AREA, stable_assembly, learn=True)
    
    print("2. Built intra-assembly connections (500 self-projections)")
    
    # Test pattern completion at different cue levels
    print("\n3. Pattern completion test:")
    
    asm_indices = stable_assembly.get().tolist()
    
    for cue_pct in [90, 70, 50, 30, 10]:
        cue_size = int(k * cue_pct / 100)
        partial_cue = cp.array(asm_indices[:cue_size], dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[AREA] = partial_cue
        brain.prev[AREA] = partial_cue
        
        initial = compute_overlap(partial_cue, stable_assembly, k)
        
        for _ in range(30):
            brain._project(AREA, brain.current[AREA], learn=False)
        
        final = compute_overlap(brain.current[AREA], stable_assembly, k)
        
        improved = final > initial
        print(f"   {cue_pct}% cue: {initial:.2f} → {final:.2f} "
              f"({'✓ IMPROVED' if improved else '✗ worse'})")


def test_fresh_area_each_assembly():
    """
    Test: Create each assembly in a FRESH area (no interference).
    
    This simulates what would happen if each assembly had its own
    dedicated area/weight space.
    """
    print("\n" + "="*60)
    print("TEST 2: Each Assembly in Fresh Brain (No Interference)")
    print("="*60)
    
    params = EmergentParams()
    params.n = 10000
    params.k = 100
    params.p = 0.1
    params.beta = 1.0
    
    k = params.k
    n = params.n
    AREA = Area.NOUN_CORE
    
    print(f"\nParameters: n={n}, k={k}, p={params.p}, beta={params.beta}")
    
    results = []
    
    for i in range(4):
        # Create FRESH brain for each assembly
        brain = EmergentNemoBrain(params=params, verbose=False)
        
        cp.random.seed((i + 1) * 1000)
        stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
        
        brain._clear_area(AREA)
        for _ in range(200):
            brain._project(AREA, stimulus, learn=True)
        stable_assembly = brain.current[AREA].copy()
        
        # Build intra-assembly connections
        for _ in range(500):
            brain.clear_all()
            brain.current[AREA] = stable_assembly.copy()
            brain.prev[AREA] = stable_assembly.copy()
            brain._project(AREA, stable_assembly, learn=True)
        
        # Test 50% cue completion
        asm_indices = stable_assembly.get().tolist()
        partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[AREA] = partial_cue
        brain.prev[AREA] = partial_cue
        
        initial = compute_overlap(partial_cue, stable_assembly, k)
        
        for _ in range(30):
            brain._project(AREA, brain.current[AREA], learn=False)
        
        final = compute_overlap(brain.current[AREA], stable_assembly, k)
        
        improved = final > initial
        results.append((initial, final, improved))
        print(f"   Assembly {i}: {initial:.2f} → {final:.2f} "
              f"({'✓' if improved else '✗'})")
    
    success_rate = sum(1 for _, _, improved in results if improved) / len(results)
    print(f"\n   Success rate: {success_rate*100:.0f}%")


def test_shared_area_problem():
    """
    Demonstrate the problem: When multiple assemblies share an area,
    their weights interfere.
    """
    print("\n" + "="*60)
    print("TEST 3: Shared Area Problem Demonstration")
    print("="*60)
    
    params = EmergentParams()
    params.n = 10000
    params.k = 100
    params.p = 0.1
    params.beta = 1.0
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    k = params.k
    n = params.n
    AREA = Area.NOUN_CORE
    
    # Create assembly 1 and test completion
    print("\n1. Create assembly 1, test completion:")
    
    cp.random.seed(1000)
    stim1 = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(200):
        brain._project(AREA, stim1, learn=True)
    asm1 = brain.current[AREA].copy()
    
    for _ in range(500):
        brain.clear_all()
        brain.current[AREA] = asm1.copy()
        brain.prev[AREA] = asm1.copy()
        brain._project(AREA, asm1, learn=True)
    
    # Test
    asm1_indices = asm1.get().tolist()
    partial = cp.array(asm1_indices[:k//2], dtype=cp.uint32)
    
    brain.clear_all()
    brain.current[AREA] = partial
    brain.prev[AREA] = partial
    
    for _ in range(30):
        brain._project(AREA, brain.current[AREA], learn=False)
    
    overlap1_before = compute_overlap(brain.current[AREA], asm1, k)
    print(f"   Assembly 1 completion: 0.50 → {overlap1_before:.2f}")
    
    # Now add assembly 2 and see if assembly 1 completion still works
    print("\n2. Add assembly 2, test assembly 1 completion again:")
    
    cp.random.seed(2000)
    stim2 = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(200):
        brain._project(AREA, stim2, learn=True)
    asm2 = brain.current[AREA].copy()
    
    for _ in range(500):
        brain.clear_all()
        brain.current[AREA] = asm2.copy()
        brain.prev[AREA] = asm2.copy()
        brain._project(AREA, asm2, learn=True)
    
    # Test assembly 1 again
    brain.clear_all()
    brain.current[AREA] = partial
    brain.prev[AREA] = partial
    
    for _ in range(30):
        brain._project(AREA, brain.current[AREA], learn=False)
    
    overlap1_after = compute_overlap(brain.current[AREA], asm1, k)
    overlap2_after = compute_overlap(brain.current[AREA], asm2, k)
    
    print(f"   Assembly 1 completion: 0.50 → {overlap1_after:.2f} (was {overlap1_before:.2f})")
    print(f"   Interference from asm2: {overlap2_after:.2f}")
    
    if overlap1_after < overlap1_before:
        print("\n   ⚠️ INTERFERENCE: Adding assembly 2 degraded assembly 1 completion!")
    
    print("""
CONCLUSION:
───────────
Pattern completion CAN work for a single assembly.
But adding more assemblies to the SAME area causes interference.

This is because all assemblies share the same weight matrix.
Self-projection for asm2 also strengthens connections that 
overlap with asm1, corrupting asm1's attractor.

This is the fundamental problem with shared weight storage.
""")


if __name__ == "__main__":
    test_single_assembly_completion()
    test_fresh_area_each_assembly()
    test_shared_area_problem()

