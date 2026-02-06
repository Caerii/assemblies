"""
Parameter Sweep: Finding Conditions for Assembly Stability
==========================================================

PROBLEM IDENTIFIED:
- Assemblies only reach ~60% stability
- Self-projection loses ~45% of neurons
- Random connectivity overpowers learned weights

HYPOTHESIS: With the right parameters, assemblies CAN become stable
and pattern completion CAN work.

Parameters to test:
- p (connection probability): Lower = less noise
- beta (learning rate): Higher = stronger learning
- Training iterations: More = accumulated strength
- k (assembly size): Different sparsity levels
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


def test_stability_with_params(n, k, p, beta, num_train, num_self_proj):
    """Test assembly stability and pattern completion with given parameters."""
    
    params = EmergentParams()
    params.n = n
    params.k = k
    params.p = p
    params.beta = beta
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    
    AREA = Area.NOUN_CORE
    
    # Create stimulus
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    # Train assembly
    brain._clear_area(AREA)
    for _ in range(num_train):
        brain._project(AREA, stimulus, learn=True)
    stable_assembly = brain.current[AREA].copy()
    
    # Measure stability after training
    brain._project(AREA, stimulus, learn=False)
    train_stability = compute_overlap(brain.current[AREA], stable_assembly, k)
    
    # Do self-projection to build intra-assembly connections
    for _ in range(num_self_proj):
        brain.clear_all()
        brain.current[AREA] = stable_assembly.copy()
        brain.prev[AREA] = stable_assembly.copy()
        brain._project(AREA, stable_assembly, learn=True)
    
    # Test pattern completion (50% cue)
    asm_indices = stable_assembly.get().tolist()
    partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
    
    brain.clear_all()
    brain.current[AREA] = partial_cue
    brain.prev[AREA] = partial_cue
    
    for _ in range(20):
        brain._project(AREA, brain.current[AREA], learn=False)
    
    completion_overlap = compute_overlap(brain.current[AREA], stable_assembly, k)
    
    return train_stability, completion_overlap


def sweep_parameters():
    """Sweep parameters to find what makes assemblies stable."""
    
    print("="*80)
    print("PARAMETER SWEEP: Finding Stable Assembly Conditions")
    print("="*80)
    
    # Fixed parameters
    n = 10000
    num_train = 50
    num_self_proj = 100
    
    results = []
    
    # Sweep over k, p, beta
    print(f"\n{'k':>6} {'p':>6} {'beta':>6} {'train_iters':>12} {'self_proj':>10} "
          f"{'stability':>10} {'completion':>12}")
    print("-" * 80)
    
    for k in [50, 100, 200]:
        for p in [0.01, 0.05, 0.1]:
            for beta in [0.1, 0.3, 0.5, 1.0]:
                try:
                    stability, completion = test_stability_with_params(
                        n=n, k=k, p=p, beta=beta,
                        num_train=num_train, num_self_proj=num_self_proj
                    )
                    
                    print(f"{k:>6} {p:>6.2f} {beta:>6.2f} {num_train:>12} {num_self_proj:>10} "
                          f"{stability:>10.3f} {completion:>12.3f}")
                    
                    results.append({
                        'k': k, 'p': p, 'beta': beta,
                        'stability': stability, 'completion': completion
                    })
                except Exception as e:
                    print(f"{k:>6} {p:>6.2f} {beta:>6.2f} ERROR: {e}")
    
    # Find best
    if results:
        best = max(results, key=lambda r: r['stability'] + r['completion'])
        print(f"\nBest configuration:")
        print(f"  k={best['k']}, p={best['p']}, beta={best['beta']}")
        print(f"  Stability: {best['stability']:.3f}")
        print(f"  Completion: {best['completion']:.3f}")
    
    return results


def test_more_training():
    """Test if MORE training iterations help."""
    
    print("\n" + "="*80)
    print("TEST: Effect of More Training")
    print("="*80)
    
    n = 10000
    k = 100
    p = 0.05
    beta = 0.5
    
    print(f"\nFixed: n={n}, k={k}, p={p}, beta={beta}")
    print(f"\n{'train_iters':>12} {'self_proj':>10} {'stability':>10} {'completion':>12}")
    print("-" * 50)
    
    for num_train in [50, 100, 200, 500]:
        for num_self_proj in [100, 200, 500]:
            stability, completion = test_stability_with_params(
                n=n, k=k, p=p, beta=beta,
                num_train=num_train, num_self_proj=num_self_proj
            )
            
            print(f"{num_train:>12} {num_self_proj:>10} {stability:>10.3f} {completion:>12.3f}")


def test_very_high_beta():
    """Test with very high learning rate."""
    
    print("\n" + "="*80)
    print("TEST: Very High Learning Rate")
    print("="*80)
    
    n = 10000
    k = 100
    p = 0.01  # Low connectivity
    
    print(f"\nFixed: n={n}, k={k}, p={p}")
    print(f"\n{'beta':>6} {'stability':>10} {'completion':>12}")
    print("-" * 30)
    
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0]:
        stability, completion = test_stability_with_params(
            n=n, k=k, p=p, beta=beta,
            num_train=100, num_self_proj=200
        )
        
        print(f"{beta:>6.1f} {stability:>10.3f} {completion:>12.3f}")


def test_pattern_completion_with_best_params():
    """Test pattern completion with the best parameters we found."""
    
    print("\n" + "="*80)
    print("TEST: Pattern Completion with Best Parameters")
    print("="*80)
    
    # Use aggressive parameters
    params = EmergentParams()
    params.n = 10000
    params.k = 100
    params.p = 0.01  # Very sparse connectivity
    params.beta = 2.0  # High learning rate
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    k = params.k
    n = params.n
    
    AREA = Area.NOUN_CORE
    
    print(f"\nParameters: n={n}, k={k}, p={params.p}, beta={params.beta}")
    
    # Create multiple assemblies
    assemblies = {}
    for i in range(4):
        cp.random.seed((i + 1) * 1000)
        stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
        
        brain._clear_area(AREA)
        for _ in range(200):  # More training
            brain._project(AREA, stimulus, learn=True)
        assemblies[f'asm{i}'] = brain.current[AREA].copy()
    
    print("\n1. Created 4 assemblies with 200 training iterations each")
    
    # Build intra-assembly connections
    for name, asm in assemblies.items():
        for _ in range(500):  # Many self-projections
            brain.clear_all()
            brain.current[AREA] = asm.copy()
            brain.prev[AREA] = asm.copy()
            brain._project(AREA, asm, learn=True)
    
    print("2. Built intra-assembly connections (500 self-projections each)")
    
    # Test pattern completion
    print("\n3. Pattern completion test (50% cue):")
    
    correct = 0
    for i in range(4):
        name = f'asm{i}'
        asm = assemblies[name]
        asm_indices = asm.get().tolist()
        
        partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
        
        brain.clear_all()
        brain.current[AREA] = partial_cue
        brain.prev[AREA] = partial_cue
        
        for _ in range(30):
            brain._project(AREA, brain.current[AREA], learn=False)
        
        retrieved = brain.current[AREA]
        overlaps = [compute_overlap(retrieved, assemblies[f'asm{j}'], k) for j in range(4)]
        best_match = np.argmax(overlaps)
        is_correct = best_match == i
        correct += is_correct
        
        print(f"   {name}: overlaps={[f'{o:.2f}' for o in overlaps]}, "
              f"best=asm{best_match}, correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")


if __name__ == "__main__":
    results = sweep_parameters()
    test_more_training()
    test_very_high_beta()
    test_pattern_completion_with_best_params()

