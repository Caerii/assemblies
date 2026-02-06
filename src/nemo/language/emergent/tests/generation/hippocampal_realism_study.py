"""
Hippocampal Realism Study
=========================

QUESTION: Can NEMO implement hippocampal-like memory with neurobiological REALISM?

Key hippocampal properties:
1. SPARSE CODING: CA3 has ~1M neurons, ~1-5% active per episode
2. RECURRENT CONNECTIONS: CA3 has extensive recurrent connectivity
3. PATTERN COMPLETION: Partial cue → full pattern retrieval
4. ONE-SHOT LEARNING: Episodic memories form quickly
5. PATTERN SEPARATION: DG orthogonalizes similar inputs

NEMO properties:
1. Sparse coding: Configurable (k/n ratio)
2. Projection: Feedforward, but can project area → same area
3. Hebbian learning: Co-active neurons strengthen connections
4. Winner-take-all: Top-k selection

HYPOTHESIS: At sufficient scale and sparsity, NEMO's recurrent projection
can implement CA3-like pattern completion without explicit storage.

Let's test this systematically...
"""

import sys
import os
import cupy as cp
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.nemo.language.emergent.brain import EmergentNemoBrain
from src.nemo.language.emergent.areas import Area
from src.nemo.language.emergent.params import EmergentParams


def compute_overlap(a1, a2, k):
    """Compute Jaccard-like overlap between two assemblies."""
    if a1 is None or a2 is None:
        return 0.0
    s1 = set(a1.get().tolist())
    s2 = set(a2.get().tolist())
    return len(s1 & s2) / k


@dataclass
class ExperimentConfig:
    n: int          # Total neurons
    k: int          # Active neurons (assembly size)
    beta: float     # Learning rate
    p: float        # Connection probability
    num_episodes: int    # Number of episodes to learn
    train_iters: int     # Training iterations per episode
    settle_iters: int    # Settling iterations for retrieval


def run_hippocampal_experiment(config: ExperimentConfig, verbose: bool = False):
    """
    Test CA3-like memory with given configuration.
    
    Protocol:
    1. Create word assemblies (simulate cortical patterns)
    2. Learn episodes by projecting words → CA3, then CA3 recurrent
    3. Also learn CA3 → cortex (reverse projection)
    4. Test retrieval: partial cue → CA3 settle → decode
    """
    # Create brain with custom parameters
    params = EmergentParams()
    params.n = config.n
    params.k = config.k
    params.beta = config.beta
    params.p = config.p
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    
    # Use SEQ as CA3 (our hippocampal area)
    CA3 = Area.SEQ
    
    # Create word assemblies in cortical areas
    words = {}
    word_list = ['dog', 'cat', 'bird', 'fish', 'runs', 'sleeps', 'flies', 'swims']
    
    for i, name in enumerate(word_list):
        area = Area.NOUN_CORE if i < 4 else Area.VERB_CORE
        cp.random.seed((i + 1) * 1000)
        phon = cp.random.randint(0, config.n, config.k, dtype=cp.uint32)
        brain._clear_area(area)
        for _ in range(20):
            brain._project(area, phon, learn=True)
        words[name] = (area, brain.current[area].copy())
    
    # Define episodes (subject-verb pairs)
    episodes = [
        ('dog', 'runs'),
        ('cat', 'sleeps'),
        ('bird', 'flies'),
        ('fish', 'swims'),
    ][:config.num_episodes]
    
    # Store episode patterns for reference
    episode_patterns = {}
    
    # =========================================================================
    # LEARNING PHASE: Create episodes in CA3 with recurrent strengthening
    # =========================================================================
    for subject, verb in episodes:
        subj_area, subj_asm = words[subject]
        verb_area, verb_asm = words[verb]
        
        for _ in range(config.train_iters):
            brain.clear_all()
            
            # Activate cortical patterns
            brain.current[subj_area] = subj_asm.copy()
            brain.prev[subj_area] = subj_asm.copy()
            brain.current[verb_area] = verb_asm.copy()
            brain.prev[verb_area] = verb_asm.copy()
            
            # Project to CA3 (episodic encoding)
            brain._project(CA3, subj_asm, learn=True)
            brain._project(CA3, verb_asm, learn=True)
            ca3_pattern = brain.current[CA3].copy()
            
            # CRITICAL: Recurrent projection in CA3 (strengthens episode)
            for _ in range(3):
                brain._project(CA3, brain.current[CA3], learn=True)
            
            # Learn reverse: CA3 → cortex (for retrieval)
            brain._project(subj_area, brain.current[CA3], learn=True)
            brain._project(verb_area, brain.current[CA3], learn=True)
        
        # Store final episode pattern
        episode_patterns[f"{subject}_{verb}"] = brain.current[CA3].copy()
    
    # =========================================================================
    # RETRIEVAL PHASE: Test pattern completion
    # =========================================================================
    results = []
    
    for subject, verb in episodes:
        subj_area, subj_asm = words[subject]
        verb_area, verb_asm = words[verb]
        
        brain.clear_all()
        
        # Activate ONLY verb (partial cue)
        brain.current[verb_area] = verb_asm.copy()
        brain.prev[verb_area] = verb_asm.copy()
        
        # Project to CA3
        brain._project(CA3, verb_asm, learn=False)
        
        # Recurrent settling in CA3 (pattern completion)
        for _ in range(config.settle_iters):
            brain._project(CA3, brain.current[CA3], learn=False)
        
        # Project CA3 → NOUN_CORE (retrieve subject)
        brain._clear_area(Area.NOUN_CORE)
        brain._project(Area.NOUN_CORE, brain.current[CA3], learn=False)
        
        # Measure retrieval accuracy
        retrieved = brain.current[Area.NOUN_CORE]
        
        # Check overlap with all subjects
        overlaps = {}
        for name in ['dog', 'cat', 'bird', 'fish'][:config.num_episodes]:
            _, asm = words[name]
            overlaps[name] = compute_overlap(retrieved, asm, config.k)
        
        # Correct if target has highest overlap
        target_overlap = overlaps[subject]
        others_max = max(v for k, v in overlaps.items() if k != subject)
        correct = target_overlap > others_max
        discrimination = target_overlap / max(others_max, 0.001)
        
        results.append({
            'query': f"Who {verb}?",
            'expected': subject,
            'target_overlap': target_overlap,
            'others_max': others_max,
            'discrimination': discrimination,
            'correct': correct,
        })
        
        if verbose:
            print(f"  {verb} → {subject}: target={target_overlap:.3f}, others_max={others_max:.3f}, "
                  f"ratio={discrimination:.2f}x, {'✓' if correct else '✗'}")
    
    # Summary metrics
    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_discrimination = np.mean([r['discrimination'] for r in results])
    avg_target_overlap = np.mean([r['target_overlap'] for r in results])
    
    return {
        'config': config,
        'accuracy': accuracy,
        'avg_discrimination': avg_discrimination,
        'avg_target_overlap': avg_target_overlap,
        'results': results,
    }


def hyperparameter_sweep():
    """
    Systematic sweep over key parameters to find what enables CA3-like memory.
    """
    print("="*80)
    print("HYPERPARAMETER SWEEP: Finding Conditions for CA3-like Memory")
    print("="*80)
    
    # Parameters to sweep
    n_values = [10000, 25000, 50000]  # Total neurons
    k_values = [25, 50, 100, 200]      # Active neurons
    beta_values = [0.1, 0.3, 0.5]      # Learning rate
    
    all_results = []
    
    print(f"\n{'n':>8} {'k':>6} {'k/n%':>8} {'beta':>6} {'Acc':>6} {'Discrim':>8} {'Target':>8}")
    print("-" * 60)
    
    for n in n_values:
        for k in k_values:
            for beta in beta_values:
                sparsity = k / n * 100
                
                config = ExperimentConfig(
                    n=n,
                    k=k,
                    beta=beta,
                    p=0.1,  # Connection probability
                    num_episodes=4,
                    train_iters=50,
                    settle_iters=10,
                )
                
                try:
                    result = run_hippocampal_experiment(config, verbose=False)
                    
                    print(f"{n:>8} {k:>6} {sparsity:>7.3f}% {beta:>6.2f} "
                          f"{result['accuracy']*100:>5.1f}% {result['avg_discrimination']:>8.2f}x "
                          f"{result['avg_target_overlap']:>8.3f}")
                    
                    all_results.append(result)
                except Exception as e:
                    print(f"{n:>8} {k:>6} {sparsity:>7.3f}% {beta:>6.2f} ERROR: {e}")
    
    # Find best configuration
    best = max(all_results, key=lambda r: (r['accuracy'], r['avg_discrimination']))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    print(f"  n = {best['config'].n}")
    print(f"  k = {best['config'].k}")
    print(f"  beta = {best['config'].beta}")
    print(f"  Sparsity = {best['config'].k / best['config'].n * 100:.3f}%")
    print(f"  Accuracy = {best['accuracy']*100:.1f}%")
    print(f"  Discrimination = {best['avg_discrimination']:.2f}x")
    
    return all_results, best


def analyze_dynamics(config: ExperimentConfig):
    """
    Analyze the dynamics of CA3 settling to understand what's happening.
    """
    print("\n" + "="*80)
    print(f"DYNAMICS ANALYSIS: n={config.n}, k={config.k}, sparsity={config.k/config.n*100:.3f}%")
    print("="*80)
    
    params = EmergentParams()
    params.n = config.n
    params.k = config.k
    params.beta = config.beta
    
    brain = EmergentNemoBrain(params=params, verbose=False)
    CA3 = Area.SEQ
    
    # Create two episodes
    words = {}
    for i, (name, area) in enumerate([
        ('dog', Area.NOUN_CORE), ('cat', Area.NOUN_CORE),
        ('runs', Area.VERB_CORE), ('sleeps', Area.VERB_CORE)
    ]):
        cp.random.seed((i + 1) * 1000)
        phon = cp.random.randint(0, config.n, config.k, dtype=cp.uint32)
        brain._clear_area(area)
        for _ in range(20):
            brain._project(area, phon, learn=True)
        words[name] = (area, brain.current[area].copy())
    
    # Learn episodes
    print("\n1. Learning episodes...")
    
    episodes = [('dog', 'runs'), ('cat', 'sleeps')]
    episode_ca3 = {}
    
    for subject, verb in episodes:
        subj_area, subj_asm = words[subject]
        verb_area, verb_asm = words[verb]
        
        for _ in range(config.train_iters):
            brain.clear_all()
            brain.current[subj_area] = subj_asm.copy()
            brain.prev[subj_area] = subj_asm.copy()
            brain.current[verb_area] = verb_asm.copy()
            brain.prev[verb_area] = verb_asm.copy()
            
            brain._project(CA3, subj_asm, learn=True)
            brain._project(CA3, verb_asm, learn=True)
            
            for _ in range(3):
                brain._project(CA3, brain.current[CA3], learn=True)
            
            brain._project(subj_area, brain.current[CA3], learn=True)
            brain._project(verb_area, brain.current[CA3], learn=True)
        
        episode_ca3[f"{subject}_{verb}"] = brain.current[CA3].copy()
        print(f"   Learned: {subject} {verb}")
    
    # Analyze CA3 episode separation
    print("\n2. CA3 Episode Separation:")
    ep1 = episode_ca3['dog_runs']
    ep2 = episode_ca3['cat_sleeps']
    ca3_overlap = compute_overlap(ep1, ep2, config.k)
    print(f"   dog_runs vs cat_sleeps overlap: {ca3_overlap:.3f}")
    print(f"   Expected for random: {config.k/config.n:.3f}")
    print(f"   Separation: {ca3_overlap / (config.k/config.n):.2f}x random")
    
    # Analyze settling dynamics
    print("\n3. Settling Dynamics for 'Who runs?':")
    
    brain.clear_all()
    _, runs_asm = words['runs']
    brain._project(CA3, runs_asm, learn=False)
    
    print(f"   Step 0: overlap with dog_runs={compute_overlap(brain.current[CA3], ep1, config.k):.3f}, "
          f"cat_sleeps={compute_overlap(brain.current[CA3], ep2, config.k):.3f}")
    
    for step in range(1, 21):
        prev_ca3 = brain.current[CA3].copy()
        brain._project(CA3, brain.current[CA3], learn=False)
        
        change = 1.0 - compute_overlap(brain.current[CA3], prev_ca3, config.k)
        ep1_overlap = compute_overlap(brain.current[CA3], ep1, config.k)
        ep2_overlap = compute_overlap(brain.current[CA3], ep2, config.k)
        
        if step <= 5 or step % 5 == 0 or change < 0.01:
            print(f"   Step {step:2d}: change={change:.3f}, dog_runs={ep1_overlap:.3f}, cat_sleeps={ep2_overlap:.3f}")
        
        if change < 0.01:
            print(f"   Converged at step {step}")
            break
    
    # Final retrieval
    print("\n4. Final Retrieval:")
    brain._clear_area(Area.NOUN_CORE)
    brain._project(Area.NOUN_CORE, brain.current[CA3], learn=False)
    
    dog_overlap = compute_overlap(brain.current[Area.NOUN_CORE], words['dog'][1], config.k)
    cat_overlap = compute_overlap(brain.current[Area.NOUN_CORE], words['cat'][1], config.k)
    
    print(f"   Retrieved NOUN_CORE vs 'dog': {dog_overlap:.3f}")
    print(f"   Retrieved NOUN_CORE vs 'cat': {cat_overlap:.3f}")
    print(f"   Correct: {'✓' if dog_overlap > cat_overlap else '✗'}")
    print(f"   Discrimination: {dog_overlap / max(cat_overlap, 0.001):.2f}x")


def test_biological_scale():
    """
    Test at more biologically realistic scales.
    
    Real CA3: ~1M neurons, ~1-5% active
    Let's test scaling behavior.
    """
    print("\n" + "="*80)
    print("BIOLOGICAL SCALE TEST")
    print("="*80)
    
    print("\nTesting how scale affects CA3-like memory...")
    print("(Targeting ~1% sparsity like biological CA3)")
    
    # Test at increasing scales while maintaining ~1% sparsity
    scales = [
        (10000, 100),   # 1.0% - baseline
        (20000, 200),   # 1.0%
        (50000, 500),   # 1.0%
        (100000, 1000), # 1.0%
    ]
    
    print(f"\n{'n':>10} {'k':>8} {'Sparsity':>10} {'Memory':>12} {'Accuracy':>10} {'Discrim':>10}")
    print("-" * 70)
    
    for n, k in scales:
        config = ExperimentConfig(
            n=n,
            k=k,
            beta=0.3,
            p=0.05,  # Lower p for larger networks
            num_episodes=4,
            train_iters=50,
            settle_iters=15,
        )
        
        # Estimate memory usage
        memory_mb = (n * n * 4) / (1024 * 1024)  # Full weight matrix (worst case)
        
        try:
            result = run_hippocampal_experiment(config, verbose=False)
            print(f"{n:>10} {k:>8} {k/n*100:>9.2f}% {memory_mb:>10.1f}MB "
                  f"{result['accuracy']*100:>9.1f}% {result['avg_discrimination']:>10.2f}x")
        except Exception as e:
            print(f"{n:>10} {k:>8} {k/n*100:>9.2f}% {memory_mb:>10.1f}MB ERROR: {e}")


def test_very_sparse():
    """
    Test with very sparse coding (biological CA3 levels).
    """
    print("\n" + "="*80)
    print("VERY SPARSE CODING TEST (Biological Levels)")
    print("="*80)
    
    print("\nBiological CA3: ~0.1-1% active neurons")
    print("Testing ultra-sparse configurations...")
    
    configs = [
        # (n, k, description)
        (10000, 10, "0.1% - very sparse"),
        (10000, 25, "0.25% - sparse"),
        (10000, 50, "0.5% - moderate"),
        (10000, 100, "1.0% - baseline"),
        (50000, 50, "0.1% - scaled"),
        (50000, 125, "0.25% - scaled"),
        (50000, 250, "0.5% - scaled"),
        (50000, 500, "1.0% - scaled"),
    ]
    
    print(f"\n{'Config':>20} {'Sparsity':>10} {'Accuracy':>10} {'Discrim':>10} {'Target':>10}")
    print("-" * 70)
    
    for n, k, desc in configs:
        config = ExperimentConfig(
            n=n,
            k=k,
            beta=0.3,
            p=0.1,
            num_episodes=4,
            train_iters=100,  # More training for sparse
            settle_iters=20,  # More settling for sparse
        )
        
        try:
            result = run_hippocampal_experiment(config, verbose=False)
            print(f"{desc:>20} {k/n*100:>9.3f}% {result['accuracy']*100:>9.1f}% "
                  f"{result['avg_discrimination']:>10.2f}x {result['avg_target_overlap']:>10.3f}")
        except Exception as e:
            print(f"{desc:>20} {k/n*100:>9.3f}% ERROR: {e}")


def main():
    print("="*80)
    print("HIPPOCAMPAL REALISM STUDY")
    print("="*80)
    print("""
QUESTION: Can NEMO implement CA3-like episodic memory with neurobiological realism?

We will test:
1. Hyperparameter sweep to find working configurations
2. Dynamics analysis to understand what's happening
3. Biological scale testing
4. Very sparse coding (like real CA3)
""")
    
    # Run experiments
    print("\n" + "="*80)
    print("EXPERIMENT 1: Hyperparameter Sweep")
    print("="*80)
    all_results, best = hyperparameter_sweep()
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: Dynamics Analysis (Best Config)")
    print("="*80)
    analyze_dynamics(best['config'])
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: Very Sparse Coding")
    print("="*80)
    test_very_sparse()
    
    # Summary
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
Based on these experiments, we can determine:

1. Does scale help? (More neurons = less interference?)
2. Does sparsity help? (Sparser coding = less overlap?)
3. Can recurrent settling achieve pattern completion?
4. What parameters are needed for reliable CA3-like memory?

If NEMO can achieve >90% accuracy with biologically plausible parameters,
then hippocampal-like memory IS achievable without explicit storage.
""")


if __name__ == "__main__":
    main()

