"""
Testing Threshold vs Top-K Dynamics
===================================

KEY HYPOTHESIS: The reason pattern completion fails in NEMO is that
top-k winner-take-all is NOT equivalent to attractor dynamics.

Top-k: "Select exactly k most activated neurons"
       - Fixed number of winners regardless of pattern
       - Selects "best" not "most similar to stored"

Threshold: "Select all neurons above threshold"
           - Pattern-dependent number of winners
           - Can converge to stored patterns

Let's test if threshold-based dynamics enable pattern completion.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))


def test_topk_vs_threshold():
    """
    Compare top-k and threshold-based dynamics for pattern completion.
    """
    print("="*70)
    print("TEST: Top-K vs Threshold Dynamics")
    print("="*70)
    
    np.random.seed(42)
    
    n = 1000  # neurons
    k = 100   # target assembly size (for top-k)
    
    # Create sparse binary patterns
    def make_sparse_pattern(seed, density=0.1):
        np.random.seed(seed)
        p = np.zeros(n)
        active = np.random.choice(n, int(n * density), replace=False)
        p[active] = 1
        return p
    
    # Patterns to store
    pattern1 = make_sparse_pattern(1)
    pattern2 = make_sparse_pattern(2)
    
    print(f"\nPattern 1: {pattern1.sum():.0f} active neurons")
    print(f"Pattern 2: {pattern2.sum():.0f} active neurons")
    print(f"Overlap: {(pattern1 * pattern2).sum():.0f} neurons")
    
    # =========================================================================
    # METHOD 1: Hopfield with Threshold
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 1: Hopfield with Threshold")
    print("-"*70)
    
    # Store patterns using outer product
    W_hopfield = np.outer(pattern1, pattern1) + np.outer(pattern2, pattern2)
    np.fill_diagonal(W_hopfield, 0)
    
    def hopfield_update(W, x, steps=20):
        """Threshold-based update with adaptive threshold."""
        for step in range(steps):
            h = W @ x
            # Adaptive threshold: keep neurons significantly above mean
            threshold = np.mean(h) + np.std(h)
            x_new = (h > threshold).astype(float)
            if np.array_equal(x_new, x):
                print(f"   Converged at step {step}")
                break
            x = x_new
        return x
    
    # Test: Query with partial pattern1
    print("\n   Query: 50% of pattern1")
    query = pattern1.copy()
    query[np.where(pattern1)[0][:int(pattern1.sum()//2)]] = 0
    print(f"   Query has {query.sum():.0f} active neurons")
    
    retrieved = hopfield_update(W_hopfield, query)
    
    overlap1 = (retrieved * pattern1).sum() / pattern1.sum()
    overlap2 = (retrieved * pattern2).sum() / pattern2.sum()
    print(f"   Retrieved overlaps: pattern1={overlap1:.3f}, pattern2={overlap2:.3f}")
    print(f"   Correct? {'YES' if overlap1 > overlap2 else 'NO'}")
    
    # =========================================================================
    # METHOD 2: Top-K (like NEMO)
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 2: Top-K (NEMO-style)")
    print("-"*70)
    
    # Same weights
    W_topk = W_hopfield.copy()
    
    def topk_update(W, x, k, steps=20):
        """Top-k winner-take-all update."""
        for step in range(steps):
            h = W @ x
            # Top-k selection
            threshold = np.sort(h)[-k] if k < len(h) else h.min() - 1
            x_new = (h >= threshold).astype(float)
            # Ensure exactly k winners
            if x_new.sum() > k:
                indices = np.argsort(h)[-k:]
                x_new = np.zeros(n)
                x_new[indices] = 1
            if np.array_equal(x_new, x):
                print(f"   Converged at step {step}")
                break
            x = x_new
        return x
    
    # Test: Query with partial pattern1
    print("\n   Query: 50% of pattern1")
    query = pattern1.copy()
    query[np.where(pattern1)[0][:int(pattern1.sum()//2)]] = 0
    
    retrieved = topk_update(W_topk, query, k)
    
    overlap1 = (retrieved * pattern1).sum() / pattern1.sum()
    overlap2 = (retrieved * pattern2).sum() / pattern2.sum()
    print(f"   Retrieved overlaps: pattern1={overlap1:.3f}, pattern2={overlap2:.3f}")
    print(f"   Correct? {'YES' if overlap1 > overlap2 else 'NO'}")
    
    # =========================================================================
    # METHOD 3: Energy-based with Top-K
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 3: Energy-Based with Top-K Constraint")
    print("-"*70)
    
    def energy(W, x):
        """Hopfield energy function."""
        return -0.5 * x @ W @ x
    
    def energy_topk_update(W, x, k, steps=20):
        """Top-k update but only accept if energy decreases."""
        for step in range(steps):
            h = W @ x
            # Top-k selection
            indices = np.argsort(h)[-k:]
            x_new = np.zeros(n)
            x_new[indices] = 1
            
            # Only accept if energy decreases
            if energy(W, x_new) < energy(W, x):
                if np.array_equal(x_new, x):
                    print(f"   Converged at step {step}")
                    break
                x = x_new
            else:
                print(f"   Energy stopped decreasing at step {step}")
                break
        return x
    
    print("\n   Query: 50% of pattern1")
    query = pattern1.copy()
    query[np.where(pattern1)[0][:int(pattern1.sum()//2)]] = 0
    
    retrieved = energy_topk_update(W_hopfield, query, k)
    
    overlap1 = (retrieved * pattern1).sum() / pattern1.sum()
    overlap2 = (retrieved * pattern2).sum() / pattern2.sum()
    print(f"   Retrieved overlaps: pattern1={overlap1:.3f}, pattern2={overlap2:.3f}")
    print(f"   Correct? {'YES' if overlap1 > overlap2 else 'NO'}")


def test_multiple_patterns():
    """
    Test with more patterns to see scaling behavior.
    """
    print("\n" + "="*70)
    print("TEST: Multiple Patterns (4 episodes)")
    print("="*70)
    
    np.random.seed(42)
    n = 1000
    k = 100
    
    # Create 4 patterns (like our 4 episodes)
    patterns = []
    for i in range(4):
        np.random.seed(i + 1)
        p = np.zeros(n)
        active = np.random.choice(n, k, replace=False)
        p[active] = 1
        patterns.append(p)
    
    # Store all patterns
    W = np.zeros((n, n))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    
    print("\n   Testing retrieval with 50% partial cue:")
    
    # Method 1: Threshold
    print("\n   THRESHOLD DYNAMICS:")
    correct_threshold = 0
    for i, p in enumerate(patterns):
        # Create 50% partial cue
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        # Retrieve with threshold
        h = W @ query
        threshold = np.mean(h) + 0.5 * np.std(h)
        for _ in range(20):
            h = W @ query
            query = (h > threshold).astype(float)
        
        # Check overlap
        overlaps = [(query * p).sum() / p.sum() for p in patterns]
        correct = np.argmax(overlaps) == i
        correct_threshold += correct
        print(f"     Pattern {i}: overlaps={[f'{o:.2f}' for o in overlaps]}, correct={correct}")
    
    print(f"   Threshold accuracy: {correct_threshold/4*100:.0f}%")
    
    # Method 2: Top-K
    print("\n   TOP-K DYNAMICS:")
    correct_topk = 0
    for i, p in enumerate(patterns):
        query = p.copy()
        active_idx = np.where(p)[0]
        query[active_idx[:len(active_idx)//2]] = 0
        
        for _ in range(20):
            h = W @ query
            indices = np.argsort(h)[-k:]
            query = np.zeros(n)
            query[indices] = 1
        
        overlaps = [(query * p).sum() / p.sum() for p in patterns]
        correct = np.argmax(overlaps) == i
        correct_topk += correct
        print(f"     Pattern {i}: overlaps={[f'{o:.2f}' for o in overlaps]}, correct={correct}")
    
    print(f"   Top-K accuracy: {correct_topk/4*100:.0f}%")


def analyze_the_difference():
    """
    Deep analysis of WHY top-k fails.
    """
    print("\n" + "="*70)
    print("ANALYSIS: Why Top-K Fails")
    print("="*70)
    
    print("""
THRESHOLD DYNAMICS:
  - Each pattern activates neurons PROPORTIONAL to similarity
  - Stored pattern neurons get high input from recurrent connections
  - Non-stored pattern neurons get low input
  - Threshold separates these → converges to stored pattern

TOP-K DYNAMICS:
  - Always selects exactly k neurons
  - If stored pattern has high overlap with multiple patterns,
    top-k may select neurons from MULTIPLE patterns
  - No mechanism to "commit" to one pattern
  - Pattern drifts rather than settles

THE CORE ISSUE:
  Top-k is COMPETITIVE but not CONVERGENT
  - It finds "the k best neurons" at each step
  - But "best" depends on ALL learned weights
  - With multiple patterns, "best k" may include neurons from many patterns
  
  Threshold is CONVERGENT
  - Neurons above threshold fire
  - Stored patterns create high mutual excitation
  - System settles to ONE stored pattern

BIOLOGICAL INTERPRETATION:
  - Real neurons don't do "top-k"
  - They have activation thresholds
  - Inhibition is LATERAL (between neurons) not GLOBAL (top-k)
  - This enables attractor dynamics

WHAT THIS MEANS FOR NEMO:
  - NEMO's winner-take-all is computationally convenient
  - But it's NOT equivalent to biological dynamics
  - To achieve pattern completion, need threshold-based or energy-based update
""")


if __name__ == "__main__":
    test_topk_vs_threshold()
    test_multiple_patterns()
    analyze_the_difference()

