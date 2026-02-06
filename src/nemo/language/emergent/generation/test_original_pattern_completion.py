"""
Test pattern completion using the ORIGINAL assembly calculus code
from simulations.py with original parameters.

Key parameters from original:
- n = 100000 (10x what we've been testing!)
- k = 317 (sqrt(n))
- p = 0.01 to 0.05
- beta = 0.05
- project_iter = 10-25
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))

import numpy as np
import random
import copy

# Import the ORIGINAL brain implementation
import brain
import brain_util as bu

def test_original_pattern_completion():
    """
    Direct copy of pattern_com_alphas from simulations.py
    """
    print("="*70)
    print("TESTING ORIGINAL PATTERN COMPLETION (from simulations.py)")
    print("="*70)
    
    # ORIGINAL parameters from simulations.py
    n = 100000
    k = 317
    p = 0.01
    beta = 0.05
    project_iter = 25
    comp_iter = 5
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"\nParameters: n={n}, k={k}, p={p}, beta={beta}")
    print(f"Training iterations: {project_iter}")
    print(f"Completion iterations: {comp_iter}")
    
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    
    # Train: stimulus + self-recurrence
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    
    print(f"\nAfter training: area A has {b.areas['A'].w} neurons that have fired")
    
    results = {}
    A_winners = b.areas["A"].winners
    
    print("\nPattern completion results:")
    print("-" * 50)
    
    for alpha in alphas:
        # Pick random subset of the neurons to fire
        subsample_size = int(k * alpha)
        b_copy = copy.deepcopy(b)
        subsample = random.sample(list(b_copy.areas["A"].winners), subsample_size)
        b_copy.areas["A"].winners = np.array(subsample, dtype=np.uint32)
        
        # Complete pattern (self-recurrence only)
        for i in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        
        final_winners = b_copy.areas["A"].winners
        o = bu.overlap(final_winners, A_winners)
        pct = float(o) / float(k) * 100
        results[alpha] = pct
        
        status = "✓" if pct > 90 else ("~" if pct > 70 else "✗")
        print(f"   {int(alpha*100):3d}% cue → {pct:5.1f}% recovery {status}")
    
    return results


def test_original_pattern_completion_with_more_iterations():
    """
    Same test but with more project_iter to see if it helps
    """
    print("\n" + "="*70)
    print("TESTING WITH MORE TRAINING ITERATIONS")
    print("="*70)
    
    n = 100000
    k = 317
    p = 0.05  # Higher connectivity
    beta = 0.05
    project_iter = 50  # More training
    comp_iter = 10  # More completion iterations
    
    print(f"\nParameters: n={n}, k={k}, p={p}, beta={beta}")
    print(f"Training iterations: {project_iter}")
    print(f"Completion iterations: {comp_iter}")
    
    b = brain.Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    
    print(f"\nAfter training: area A has {b.areas['A'].w} neurons that have fired")
    
    A_winners = b.areas["A"].winners
    
    print("\nPattern completion results:")
    print("-" * 50)
    
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        subsample_size = int(k * alpha)
        b_copy = copy.deepcopy(b)
        subsample = random.sample(list(b_copy.areas["A"].winners), subsample_size)
        b_copy.areas["A"].winners = np.array(subsample, dtype=np.uint32)
        
        for i in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        
        final_winners = b_copy.areas["A"].winners
        o = bu.overlap(final_winners, A_winners)
        pct = float(o) / float(k) * 100
        
        status = "✓" if pct > 90 else ("~" if pct > 70 else "✗")
        print(f"   {int(alpha*100):3d}% cue → {pct:5.1f}% recovery {status}")


def test_pattern_com_repeated():
    """
    Use the pattern_com_repeated function from simulations.py
    """
    print("\n" + "="*70)
    print("TESTING pattern_com_repeated (convergence until no new winners)")
    print("="*70)
    
    n = 100000
    k = 317
    p = 0.05
    beta = 0.05
    project_iter = 12
    alpha = 0.4
    trials = 3
    max_recurrent_iter = 10
    
    print(f"\nParameters: n={n}, k={k}, p={p}, beta={beta}")
    print(f"Training iterations: {project_iter}, alpha={alpha}")
    
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter - 1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    
    # Save the original assembly
    original_winners = b.areas["A"].winners.copy()
    
    subsample_size = int(k * alpha)
    rounds_to_completion = []
    
    # Pick random subset of the neurons to fire
    subsample = random.sample(list(b.areas["A"].winners), subsample_size)
    
    print(f"\nRunning {trials} trials with {int(alpha*100)}% cue:")
    
    for trial in range(trials):
        b.areas["A"].winners = np.array(subsample, dtype=np.uint32)
        rounds = 0
        while True:
            rounds += 1
            b.project({}, {"A": ["A"]})
            if (b.areas["A"].num_first_winners == 0) or (rounds == max_recurrent_iter):
                break
        rounds_to_completion.append(rounds)
        
        # Check overlap with original
        o = bu.overlap(b.areas["A"].winners, original_winners)
        pct = float(o) / float(k) * 100
        print(f"   Trial {trial+1}: {rounds} rounds to converge, {pct:.1f}% overlap")
    
    print(f"\n   Average rounds: {np.mean(rounds_to_completion):.1f}")


def test_our_implementation_with_original_params():
    """
    Test our NEMO implementation with the same original parameters
    """
    print("\n" + "="*70)
    print("TESTING OUR IMPLEMENTATION with original parameters")
    print("="*70)
    
    import cupy as cp
    from nemo.language.emergent.brain import EmergentNemoBrain
    from nemo.language.emergent.areas import Area
    from nemo.language.emergent.params import EmergentParams
    
    # Use ORIGINAL parameters
    params = EmergentParams()
    params.n = 100000  # Original: 100k
    params.k = 317     # Original: sqrt(n)
    params.p = 0.01    # Original
    params.beta = 0.05 # Original
    
    print(f"\nParameters: n={params.n}, k={params.k}, p={params.p}, beta={params.beta}")
    
    brain_obj = EmergentNemoBrain(params=params, verbose=False)
    k = params.k
    n = params.n
    AREA = Area.NOUN_CORE
    
    # Create stable assembly (original does 25 iterations)
    cp.random.seed(42)
    stimulus = cp.random.randint(0, n, k, dtype=cp.uint32)
    
    brain_obj._clear_area(AREA)
    for _ in range(25):  # Same as original project_iter
        brain_obj._project(AREA, stimulus, learn=True)
    
    original_assembly = brain_obj.current[AREA].copy()
    print(f"Created assembly with {k} neurons")
    
    print("\nPattern completion results:")
    print("-" * 50)
    
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        asm_indices = original_assembly.get().tolist()
        cue_size = int(k * alpha)
        partial_cue = cp.array(asm_indices[:cue_size], dtype=cp.uint32)
        
        brain_obj.clear_all()
        brain_obj.current[AREA] = partial_cue
        brain_obj.prev[AREA] = partial_cue
        
        # 5 completion iterations (same as original comp_iter)
        for _ in range(5):
            brain_obj._project(AREA, brain_obj.current[AREA], learn=False)
        
        final = brain_obj.current[AREA]
        s1 = set(final.get().tolist())
        s2 = set(original_assembly.get().tolist())
        overlap = len(s1 & s2) / k * 100
        
        status = "✓" if overlap > 90 else ("~" if overlap > 70 else "✗")
        print(f"   {int(alpha*100):3d}% cue → {overlap:5.1f}% recovery {status}")


if __name__ == "__main__":
    test_original_pattern_completion()
    test_original_pattern_completion_with_more_iterations()
    test_pattern_com_repeated()
    test_our_implementation_with_original_params()

