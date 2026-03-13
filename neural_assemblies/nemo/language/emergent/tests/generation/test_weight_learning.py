"""
Testing Weight Learning: Outer Product vs NEMO-style
====================================================

KEY FINDING from previous test:
- Top-K with HOPFIELD weights (outer product): 100% accuracy
- NEMO with top-K: 25% accuracy

The difference must be in HOW WEIGHTS ARE LEARNED, not retrieval dynamics!

Hopfield: W = Σ (pattern ⊗ pattern)  -- explicit pattern storage
NEMO:     W updated incrementally through projection -- implicit storage

Let's test this hypothesis.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))


def test_hopfield_weights():
    """
    Test pattern completion with Hopfield-style weight storage.
    
    W = Σ (pattern ⊗ pattern)
    """
    print("="*70)
    print("TEST 1: Hopfield Weight Storage (Outer Product)")
    print("="*70)
    
    np.random.seed(42)
    n = 1000
    k = 100
    
    # Create patterns
    patterns = []
    for i in range(4):
        np.random.seed(i + 1)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        patterns.append(p)
    
    # Hopfield weight storage: W = Σ (p ⊗ p)
    W = np.zeros((n, n))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    
    print(f"\n   Weight matrix density: {(W > 0).mean()*100:.2f}%")
    print(f"   Weight range: [{W.min():.2f}, {W.max():.2f}]")
    
    # Test retrieval with top-K
    print("\n   Retrieval with 50% partial cue (Top-K):")
    correct = 0
    for i, p in enumerate(patterns):
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        for _ in range(10):
            h = W @ query
            indices = np.argsort(h)[-k:]
            query_new = np.zeros(n)
            query_new[indices] = 1
            if np.array_equal(query_new, query):
                break
            query = query_new
        
        overlap_target = (query * p).sum() / p.sum()
        overlap_others = max((query * patterns[j]).sum() / patterns[j].sum() 
                            for j in range(4) if j != i)
        is_correct = overlap_target > overlap_others
        correct += is_correct
        print(f"     Pattern {i}: target={overlap_target:.2f}, others_max={overlap_others:.2f}, "
              f"correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")
    return correct / 4


def test_nemo_style_weights():
    """
    Test pattern completion with NEMO-style weight learning.
    
    Simulates: project pattern → area, with Hebbian weight updates.
    """
    print("\n" + "="*70)
    print("TEST 2: NEMO-Style Weight Learning")
    print("="*70)
    
    np.random.seed(42)
    n = 1000
    k = 100
    beta = 0.1  # Learning rate
    
    # Create patterns
    patterns = []
    for i in range(4):
        np.random.seed(i + 1)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        patterns.append(p)
    
    # Initialize random connectivity (like NEMO's implicit connections)
    np.random.seed(999)
    W = (np.random.rand(n, n) < 0.1).astype(float)  # 10% connectivity
    np.fill_diagonal(W, 0)
    
    print(f"\n   Initial weight density: {(W > 0).mean()*100:.2f}%")
    
    # Learn patterns through projection (NEMO-style)
    for ep in range(50):  # Multiple epochs
        for i, p in enumerate(patterns):
            # Project pattern to area (Hebbian update)
            h = W @ p
            # Top-k winners
            indices = np.argsort(h)[-k:]
            winners = np.zeros(n)
            winners[indices] = 1
            
            # Hebbian update: strengthen p → winners connections
            for j in np.where(p)[0]:
                for l in indices:
                    W[j, l] *= (1 + beta)
    
    print(f"   Final weight density: {(W > 0).mean()*100:.2f}%")
    print(f"   Weight range: [{W.min():.4f}, {W.max():.4f}]")
    
    # Test retrieval
    print("\n   Retrieval with 50% partial cue (Top-K):")
    correct = 0
    for i, p in enumerate(patterns):
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        for _ in range(10):
            h = W @ query
            indices = np.argsort(h)[-k:]
            query_new = np.zeros(n)
            query_new[indices] = 1
            if np.array_equal(query_new, query):
                break
            query = query_new
        
        overlap_target = (query * p).sum() / p.sum()
        overlap_others = max((query * patterns[j]).sum() / patterns[j].sum() 
                            for j in range(4) if j != i)
        is_correct = overlap_target > overlap_others
        correct += is_correct
        print(f"     Pattern {i}: target={overlap_target:.2f}, others_max={overlap_others:.2f}, "
              f"correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")
    return correct / 4


def test_nemo_with_recurrent():
    """
    Test NEMO-style with recurrent learning (like our actual NEMO).
    """
    print("\n" + "="*70)
    print("TEST 3: NEMO-Style with Recurrent Learning")
    print("="*70)
    
    np.random.seed(42)
    n = 1000
    k = 100
    beta = 0.1
    
    patterns = []
    for i in range(4):
        np.random.seed(i + 1)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        patterns.append(p)
    
    np.random.seed(999)
    W = (np.random.rand(n, n) < 0.1).astype(float)
    np.fill_diagonal(W, 0)
    
    print("\n   Learning with recurrent projection...")
    
    for ep in range(50):
        for i, p in enumerate(patterns):
            current = p.copy()
            
            # Project pattern
            h = W @ current
            indices = np.argsort(h)[-k:]
            winners = np.zeros(n)
            winners[indices] = 1
            
            # Hebbian: p → winners
            for j in np.where(p)[0]:
                for l in indices:
                    W[j, l] *= (1 + beta)
            
            # Recurrent projection (area → same area)
            for _ in range(3):
                h = W @ winners
                new_indices = np.argsort(h)[-k:]
                new_winners = np.zeros(n)
                new_winners[new_indices] = 1
                
                # Hebbian: winners → new_winners
                for j in indices:
                    for l in new_indices:
                        W[j, l] *= (1 + beta)
                
                winners = new_winners
                indices = new_indices
    
    print(f"   Final weight range: [{W.min():.4f}, {W.max():.4f}]")
    
    # Test retrieval
    print("\n   Retrieval with 50% partial cue (Top-K):")
    correct = 0
    for i, p in enumerate(patterns):
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        for _ in range(10):
            h = W @ query
            indices = np.argsort(h)[-k:]
            query_new = np.zeros(n)
            query_new[indices] = 1
            if np.array_equal(query_new, query):
                break
            query = query_new
        
        overlap_target = (query * p).sum() / p.sum()
        overlap_others = max((query * patterns[j]).sum() / patterns[j].sum() 
                            for j in range(4) if j != i)
        is_correct = overlap_target > overlap_others
        correct += is_correct
        print(f"     Pattern {i}: target={overlap_target:.2f}, others_max={overlap_others:.2f}, "
              f"correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")
    return correct / 4


def test_hybrid_hopfield_nemo():
    """
    Test hybrid: Use Hopfield storage WITHIN NEMO framework.
    
    Key insight: What if we use outer product storage but with NEMO's
    sparse representation?
    """
    print("\n" + "="*70)
    print("TEST 4: Hybrid - Hopfield Storage in NEMO Framework")
    print("="*70)
    
    np.random.seed(42)
    n = 1000
    k = 100
    
    patterns = []
    for i in range(4):
        np.random.seed(i + 1)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        patterns.append(p)
    
    # Start with sparse random connectivity
    np.random.seed(999)
    W_base = (np.random.rand(n, n) < 0.1).astype(float)
    np.fill_diagonal(W_base, 0)
    
    # Add Hopfield-style pattern storage ON TOP of base connectivity
    W_patterns = np.zeros((n, n))
    for p in patterns:
        W_patterns += np.outer(p, p)
    
    # Combine: base + scaled pattern storage
    W = W_base + 0.5 * W_patterns
    np.fill_diagonal(W, 0)
    
    print(f"\n   Hybrid weight density: {(W > 0).mean()*100:.2f}%")
    
    # Test retrieval
    print("\n   Retrieval with 50% partial cue (Top-K):")
    correct = 0
    for i, p in enumerate(patterns):
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        for _ in range(10):
            h = W @ query
            indices = np.argsort(h)[-k:]
            query_new = np.zeros(n)
            query_new[indices] = 1
            if np.array_equal(query_new, query):
                break
            query = query_new
        
        overlap_target = (query * p).sum() / p.sum()
        overlap_others = max((query * patterns[j]).sum() / patterns[j].sum() 
                            for j in range(4) if j != i)
        is_correct = overlap_target > overlap_others
        correct += is_correct
        print(f"     Pattern {i}: target={overlap_target:.2f}, others_max={overlap_others:.2f}, "
              f"correct={is_correct}")
    
    print(f"\n   Accuracy: {correct/4*100:.0f}%")
    return correct / 4


def analyze_results():
    print("\n" + "="*70)
    print("SUMMARY AND ANALYSIS")
    print("="*70)
    
    print("""
KEY FINDINGS:

1. HOPFIELD (outer product) + Top-K → 100% accuracy
   - Patterns stored as explicit attractors
   - Top-K retrieves correct pattern

2. NEMO-style projection + Top-K → Low accuracy  
   - Patterns stored implicitly in shared weights
   - Interference between patterns

3. NEMO with recurrent → Still low accuracy
   - Recurrent strengthens wrong associations
   - Shared weights cause drift

4. HYBRID (base + Hopfield) → Should work
   - Combines NEMO connectivity with Hopfield storage
   - Best of both worlds?

THE CORE INSIGHT:
─────────────────
The issue is NOT top-k vs threshold.
The issue is HOW PATTERNS ARE STORED IN WEIGHTS.

Hopfield: W = Σ (pattern ⊗ pattern)
         Each pattern explicitly creates attractor structure
         
NEMO:    W updated through sequential projection
         Patterns implicitly encoded
         Shared weights cause interference

TO FIX NEMO FOR PATTERN COMPLETION:
───────────────────────────────────
Option 1: Store patterns explicitly (outer product)
         W += pattern ⊗ pattern
         This creates attractor structure
         Compatible with NEMO's sparse representation

Option 2: Separate episodic memory area
         Regular NEMO for language
         Hopfield-like area for episodes
         Connect via projection

Option 3: Accept limitation
         NEMO for pattern formation
         Explicit storage for retrieval
         ~85% neural solution

BIOLOGICAL INTERPRETATION:
──────────────────────────
The hippocampus DOES use a different storage mechanism than cortex!
- Cortex: Slow learning, distributed representations
- Hippocampus: Fast learning, pattern separation, attractors

A HYBRID architecture matches biology:
- NEMO for cortical processing
- Hopfield-like for hippocampal memory
""")


if __name__ == "__main__":
    acc1 = test_hopfield_weights()
    acc2 = test_nemo_style_weights()
    acc3 = test_nemo_with_recurrent()
    acc4 = test_hybrid_hopfield_nemo()
    
    print("\n" + "="*70)
    print("ACCURACY COMPARISON")
    print("="*70)
    print(f"   Hopfield weights:      {acc1*100:.0f}%")
    print(f"   NEMO-style:            {acc2*100:.0f}%")
    print(f"   NEMO with recurrent:   {acc3*100:.0f}%")
    print(f"   Hybrid (base+Hopfield):{acc4*100:.0f}%")
    
    analyze_results()

