"""
Testing Pattern Completion PROPERLY
===================================

The user is RIGHT - Assembly Calculus theory says pattern completion SHOULD work.

Let me re-read the theory and test correctly:

1. STABLE ASSEMBLY FORMATION:
   - Project stimulus → area MANY times
   - The SAME neurons keep winning (Hebbian strengthening)
   - Eventually, assembly becomes STABLE (same neurons every time)

2. INTRA-ASSEMBLY CONNECTIONS:
   - After assembly forms, do SELF-PROJECTION (area → same area)
   - This strengthens connections WITHIN the assembly
   - Neurons n_i and n_j in assembly become strongly connected

3. PATTERN COMPLETION:
   - Activate PARTIAL assembly (e.g., 50% of neurons)
   - Self-project (area → same area)
   - The strong intra-assembly connections should complete the pattern

What we might have done wrong:
- Not enough training iterations
- Not enough self-projection for intra-assembly connections
- Testing with wrong protocol

Let's test this PROPERLY following the theory.
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


def test_assembly_stability():
    """
    Step 1: Verify that assemblies become STABLE with repeated projection.
    
    This is fundamental - if assemblies aren't stable, nothing else will work.
    """
    print("="*70)
    print("TEST 1: Assembly Stability")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    AREA = Area.NOUN_CORE
    
    print(f"\nParameters: n={n}, k={k}, beta={brain.p.beta}")
    
    # Create stimulus
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    print("\n1. Projecting stimulus → area repeatedly...")
    
    prev_assembly = None
    brain._clear_area(AREA)
    
    for step in range(50):
        brain._project(AREA, stimulus, learn=True)
        current = brain.current[AREA].copy()
        
        if prev_assembly is not None:
            stability = compute_overlap(current, prev_assembly, k)
            if step < 10 or step % 10 == 0:
                print(f"   Step {step+1}: stability = {stability:.3f}")
            if stability == 1.0:
                print(f"   STABLE at step {step+1}!")
                break
        
        prev_assembly = current
    
    stable_assembly = brain.current[AREA].copy()
    print(f"\n   Final assembly: {k} neurons")
    
    return stable_assembly, stimulus


def test_intra_assembly_connections():
    """
    Step 2: Build intra-assembly connections via self-projection.
    
    After assembly is stable, project area → same area to strengthen
    connections WITHIN the assembly.
    """
    print("\n" + "="*70)
    print("TEST 2: Intra-Assembly Connection Building")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    AREA = Area.NOUN_CORE
    
    # Create stable assembly
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(30):
        brain._project(AREA, stimulus, learn=True)
    stable_assembly = brain.current[AREA].copy()
    
    print(f"\n1. Created stable assembly with {k} neurons")
    
    # Now do SELF-PROJECTION to build intra-assembly connections
    print("\n2. Self-projection (area → same area) to build intra-assembly connections...")
    
    for step in range(100):
        brain.clear_all()
        # Activate the FULL stable assembly
        brain.current[AREA] = stable_assembly.copy()
        brain.prev[AREA] = stable_assembly.copy()
        
        # Self-project WITH learning
        brain._project(AREA, stable_assembly, learn=True)
        
        # Check if winners are the same as assembly
        new_assembly = brain.current[AREA]
        overlap = compute_overlap(new_assembly, stable_assembly, k)
        
        if step < 5 or step % 20 == 0:
            print(f"   Step {step+1}: self-projection overlap = {overlap:.3f}")
    
    print(f"\n   After 100 self-projections, intra-assembly connections should be strong")
    
    return stable_assembly


def test_pattern_completion_single_assembly():
    """
    Step 3: Test pattern completion on a SINGLE assembly.
    
    Protocol:
    1. Create stable assembly
    2. Build intra-assembly connections
    3. Activate PARTIAL assembly
    4. Self-project to complete
    """
    print("\n" + "="*70)
    print("TEST 3: Pattern Completion (Single Assembly)")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    AREA = Area.NOUN_CORE
    
    # Step 1: Create stable assembly
    print("\n1. Creating stable assembly...")
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(30):
        brain._project(AREA, stimulus, learn=True)
    stable_assembly = brain.current[AREA].copy()
    
    # Step 2: Build intra-assembly connections
    print("2. Building intra-assembly connections (200 self-projections)...")
    for _ in range(200):
        brain.clear_all()
        brain.current[AREA] = stable_assembly.copy()
        brain.prev[AREA] = stable_assembly.copy()
        brain._project(AREA, stable_assembly, learn=True)
    
    # Step 3: Test pattern completion with different cue sizes
    print("\n3. Testing pattern completion...")
    
    assembly_indices = stable_assembly.get().tolist()
    
    for cue_pct in [90, 70, 50, 30, 10]:
        cue_size = int(k * cue_pct / 100)
        partial_indices = assembly_indices[:cue_size]
        partial_cue = cp.array(partial_indices, dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[AREA] = partial_cue
        brain.prev[AREA] = partial_cue
        
        # Self-project to complete (without learning)
        initial_overlap = compute_overlap(partial_cue, stable_assembly, k)
        
        for step in range(20):
            brain._project(AREA, brain.current[AREA], learn=False)
        
        final_overlap = compute_overlap(brain.current[AREA], stable_assembly, k)
        
        print(f"   {cue_pct}% cue ({cue_size} neurons): "
              f"initial={initial_overlap:.3f} → final={final_overlap:.3f} "
              f"({'✓ COMPLETION' if final_overlap > initial_overlap + 0.1 else '✗ no completion'})")


def test_pattern_completion_multiple_assemblies():
    """
    Step 4: Test pattern completion with MULTIPLE assemblies.
    
    This is the harder case - can we complete patterns without interference?
    """
    print("\n" + "="*70)
    print("TEST 4: Pattern Completion (Multiple Assemblies)")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    AREA = Area.NOUN_CORE
    
    # Create multiple stable assemblies
    print("\n1. Creating 4 stable assemblies...")
    assemblies = {}
    
    for i in range(4):
        cp.random.seed((i + 1) * 1000)
        stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
        
        brain._clear_area(AREA)
        for _ in range(30):
            brain._project(AREA, stimulus, learn=True)
        assemblies[f'asm{i}'] = brain.current[AREA].copy()
    
    # Check overlaps
    print("   Assembly overlaps:")
    for i in range(4):
        for j in range(i+1, 4):
            ov = compute_overlap(assemblies[f'asm{i}'], assemblies[f'asm{j}'], k)
            print(f"     asm{i} ∩ asm{j}: {ov:.3f}")
    
    # Build intra-assembly connections for ALL
    print("\n2. Building intra-assembly connections for all assemblies...")
    
    for name, asm in assemblies.items():
        for _ in range(100):
            brain.clear_all()
            brain.current[AREA] = asm.copy()
            brain.prev[AREA] = asm.copy()
            brain._project(AREA, asm, learn=True)
        print(f"   {name}: 100 self-projections")
    
    # Test pattern completion
    print("\n3. Testing pattern completion (50% cue)...")
    
    correct = 0
    for i in range(4):
        name = f'asm{i}'
        asm = assemblies[name]
        asm_indices = asm.get().tolist()
        
        # 50% partial cue
        partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[AREA] = partial_cue
        brain.prev[AREA] = partial_cue
        
        # Self-project to complete
        for _ in range(20):
            brain._project(AREA, brain.current[AREA], learn=False)
        
        retrieved = brain.current[AREA]
        
        # Check which assembly it matches best
        overlaps = [compute_overlap(retrieved, assemblies[f'asm{j}'], k) for j in range(4)]
        best_match = np.argmax(overlaps)
        is_correct = best_match == i
        correct += is_correct
        
        print(f"   {name}: overlaps={[f'{o:.2f}' for o in overlaps]}, "
              f"best=asm{best_match}, correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")


def analyze_what_happens():
    """
    Deep dive: What EXACTLY happens during self-projection?
    """
    print("\n" + "="*70)
    print("ANALYSIS: What Happens During Self-Projection")
    print("="*70)
    
    brain = EmergentNemoBrain(verbose=False)
    k = brain.p.k
    n = brain.p.n
    
    AREA = Area.NOUN_CORE
    
    # Create stable assembly
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(30):
        brain._project(AREA, stimulus, learn=True)
    stable_assembly = brain.current[AREA].copy()
    
    print(f"\n1. Stable assembly has {k} neurons")
    print(f"   Assembly indices: {sorted(stable_assembly.get().tolist())[:10]}... (first 10)")
    
    # Analyze self-projection
    print("\n2. Self-projection analysis:")
    
    brain.clear_all()
    brain.current[AREA] = stable_assembly.copy()
    brain.prev[AREA] = stable_assembly.copy()
    
    # What happens when we project?
    brain._project(AREA, stable_assembly, learn=False)
    
    after_projection = brain.current[AREA]
    overlap_with_original = compute_overlap(after_projection, stable_assembly, k)
    
    print(f"   Project assembly → same area (no learning)")
    print(f"   Overlap with original: {overlap_with_original:.3f}")
    
    # The issue: Are the SAME neurons winning?
    asm_set = set(stable_assembly.get().tolist())
    new_set = set(after_projection.get().tolist())
    
    same_neurons = len(asm_set & new_set)
    new_neurons = len(new_set - asm_set)
    lost_neurons = len(asm_set - new_set)
    
    print(f"   Same neurons: {same_neurons}/{k}")
    print(f"   New neurons (not in assembly): {new_neurons}")
    print(f"   Lost neurons (were in assembly): {lost_neurons}")
    
    print("""
    
THE ISSUE:
──────────
When we project assembly → same area, TOP-K selects the k neurons
with highest total input. But these might NOT be the assembly neurons!

The assembly neurons receive input from the stimulus.
But OTHER neurons might receive input from the random base connectivity.

If the random connectivity is too strong, non-assembly neurons win.

POSSIBLE FIXES:
───────────────
1. Weaker random connectivity (lower p)
2. Stronger learned weights (higher beta, more training)
3. Larger k (more assembly neurons → more votes)
4. Explicitly tracking assemblies (not pure NEMO)
""")


if __name__ == "__main__":
    stable_assembly, stimulus = test_assembly_stability()
    test_intra_assembly_connections()
    test_pattern_completion_single_assembly()
    test_pattern_completion_multiple_assemblies()
    analyze_what_happens()

