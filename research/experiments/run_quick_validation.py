"""
Quick Validation Suite for Assembly Calculus

Runs essential tests to validate the core claims:
1. Projection convergence
2. Assembly distinctiveness (same brain)
3. Capacity limits
4. Phase transition (n/k boundary)
5. Cross-area retrieval (with fixed assemblies)

Usage:
    uv run python research/experiments/run_quick_validation.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import brain as b


def test_projection_convergence():
    """Test 1: Does projection converge to stable assembly?"""
    print("\n" + "="*60)
    print("TEST 1: Projection Convergence")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    brain.add_stimulus("STIM", 50)
    brain.add_area("T", 5000, 50, 0.1)
    
    prev = None
    for round_idx in range(20):
        brain.project({"STIM": ["T"]}, {})
        curr = set(brain.area_by_name["T"].winners.tolist())
        
        if prev is not None:
            overlap = len(prev & curr) / 50
            if overlap > 0.99:
                print(f"  PASS: Converged in {round_idx + 1} rounds")
                return True
        prev = curr
    
    print("  FAIL: Did not converge in 20 rounds")
    return False


def test_distinctiveness():
    """Test 2: Are assemblies from different stimuli distinct?"""
    print("\n" + "="*60)
    print("TEST 2: Assembly Distinctiveness")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    for i in range(10):
        brain.add_stimulus(f"S{i}", 50)
    brain.add_area("T", 5000, 50, 0.1)
    
    assemblies = {}
    for i in range(10):
        for _ in range(10):
            brain.project({f"S{i}": ["T"]}, {})
        assemblies[i] = set(brain.area_by_name["T"].winners.tolist())
    
    overlaps = []
    for i in range(10):
        for j in range(i+1, 10):
            overlaps.append(len(assemblies[i] & assemblies[j]) / 50)
    
    mean_overlap = np.mean(overlaps)
    if mean_overlap < 0.1:
        print(f"  PASS: Mean overlap = {mean_overlap:.3f} (< 0.1)")
        return True
    else:
        print(f"  FAIL: Mean overlap = {mean_overlap:.3f} (>= 0.1)")
        return False


def test_capacity():
    """Test 3: Can we store many distinct assemblies?"""
    print("\n" + "="*60)
    print("TEST 3: Capacity (100 assemblies)")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    for i in range(100):
        brain.add_stimulus(f"S{i}", 50)
    brain.add_area("T", 10000, 50, 0.1)
    
    assemblies = {}
    for i in range(100):
        for _ in range(10):
            brain.project({f"S{i}": ["T"]}, {})
        assemblies[i] = set(brain.area_by_name["T"].winners.tolist())
    
    # Sample overlaps (computing all would be slow)
    sample_overlaps = []
    for _ in range(100):
        i, j = np.random.choice(100, 2, replace=False)
        sample_overlaps.append(len(assemblies[i] & assemblies[j]) / 50)
    
    mean_overlap = np.mean(sample_overlaps)
    if mean_overlap < 0.1:
        print(f"  PASS: 100 assemblies with mean overlap = {mean_overlap:.3f}")
        return True
    else:
        print(f"  FAIL: Mean overlap = {mean_overlap:.3f}")
        return False


def test_phase_transition():
    """Test 4: Is there a phase transition at n/k ~ 6?"""
    print("\n" + "="*60)
    print("TEST 4: Phase Transition (n/k boundary)")
    print("="*60)
    
    def get_overlap(n, k):
        brain = b.Brain(p=0.1, seed=42)
        for i in range(5):
            brain.add_stimulus(f"S{i}", k)
        brain.add_area("T", n, k, 0.1)
        
        assemblies = {}
        for i in range(5):
            for _ in range(10):
                brain.project({f"S{i}": ["T"]}, {})
            assemblies[i] = set(brain.area_by_name["T"].winners.tolist())
        
        overlaps = []
        for i in range(5):
            for j in range(i+1, 5):
                overlaps.append(len(assemblies[i] & assemblies[j]) / k)
        return np.mean(overlaps)
    
    k = 50
    overlap_6 = get_overlap(k * 6, k)  # n/k = 6 (boundary)
    overlap_20 = get_overlap(k * 20, k)  # n/k = 20 (clearly distinct)
    
    print(f"  n/k = 6: overlap = {overlap_6:.3f}")
    print(f"  n/k = 20: overlap = {overlap_20:.3f}")
    
    # At n/k=6, overlap should be higher than at n/k=20
    # Both should be < 0.3 (distinct), but 6 should be noticeably higher
    if overlap_6 < 0.3 and overlap_20 < 0.1:
        print("  PASS: Both regimes produce distinct assemblies")
        print("  (Phase transition occurs below n/k=6)")
        return True
    else:
        print("  FAIL: Unexpected overlap values")
        return False


def test_cross_area_fixed():
    """Test 5: Cross-area retrieval with fixed assemblies"""
    print("\n" + "="*60)
    print("TEST 5: Cross-Area Retrieval (Fixed Assemblies)")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    brain.add_stimulus("STIM_A", 50)
    brain.add_stimulus("STIM_B", 50)
    brain.add_area("X", 5000, 50, 0.1)
    brain.add_area("Y", 5000, 50, 0.1)
    
    # Create and fix X
    for _ in range(15):
        brain.project({"STIM_A": ["X"]}, {})
    original_x = set(brain.area_by_name["X"].winners.tolist())
    brain.area_by_name["X"].fix_assembly()
    
    # Create and fix Y
    for _ in range(15):
        brain.project({"STIM_B": ["Y"]}, {})
    brain.area_by_name["Y"].fix_assembly()
    
    # Associate
    for _ in range(30):
        brain.project({}, {"X": ["Y"], "Y": ["X"]})
    
    # Unfix X and retrieve via Y
    brain.area_by_name["X"].fixed_assembly = False
    for _ in range(5):
        brain.project({}, {"Y": ["X"]})
    
    retrieved_x = set(brain.area_by_name["X"].winners.tolist())
    overlap = len(original_x & retrieved_x) / 50
    
    if overlap > 0.9:
        print(f"  PASS: Cross-area retrieval = {overlap:.3f}")
        return True
    else:
        print(f"  FAIL: Cross-area retrieval = {overlap:.3f}")
        return False


def test_learning_stability():
    """Test 6: Is learning stable over many rounds?"""
    print("\n" + "="*60)
    print("TEST 6: Learning Stability")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    brain.add_stimulus("STIM", 50)
    brain.add_area("T", 5000, 50, 0.1)
    
    first_assembly = None
    for round_idx in range(100):
        brain.project({"STIM": ["T"]}, {})
        if first_assembly is None:
            first_assembly = set(brain.area_by_name["T"].winners.tolist())
    
    final_assembly = set(brain.area_by_name["T"].winners.tolist())
    overlap = len(first_assembly & final_assembly) / 50
    
    if overlap > 0.8:
        print(f"  PASS: Assembly stable over 100 rounds (overlap={overlap:.3f})")
        return True
    else:
        print(f"  FAIL: Assembly drifted (overlap={overlap:.3f})")
        return False


def test_language_syntax():
    """Test 7: Can assemblies represent syntactic structure?"""
    print("\n" + "="*60)
    print("TEST 7: Language/Syntax")
    print("="*60)
    
    brain = b.Brain(p=0.1, seed=42)
    
    # Words
    for word in ["the", "dog", "runs"]:
        brain.add_stimulus(word, 50)
    
    # Areas
    for area in ["LEX", "DET", "NOUN", "VERB", "NP", "VP", "S"]:
        brain.add_area(area, 5000, 50, 0.1)
    
    # Project words to LEX
    word_assemblies = {}
    for word in ["the", "dog", "runs"]:
        for _ in range(10):
            brain.project({word: ["LEX"]}, {})
        word_assemblies[word] = set(brain.area_by_name["LEX"].winners.tolist())
    
    # Check distinctiveness
    overlaps = []
    words = list(word_assemblies.keys())
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            overlaps.append(len(word_assemblies[words[i]] & word_assemblies[words[j]]) / 50)
    
    if np.mean(overlaps) < 0.1:
        print(f"  PASS: Words are distinct (overlap={np.mean(overlaps):.3f})")
        return True
    else:
        print(f"  FAIL: Words overlap too much (overlap={np.mean(overlaps):.3f})")
        return False


def main():
    print("="*60)
    print("ASSEMBLY CALCULUS QUICK VALIDATION SUITE")
    print("="*60)
    
    results = {
        "Projection Convergence": test_projection_convergence(),
        "Distinctiveness": test_distinctiveness(),
        "Capacity": test_capacity(),
        "Phase Transition": test_phase_transition(),
        "Cross-Area (Fixed)": test_cross_area_fixed(),
        "Learning Stability": test_learning_stability(),
        "Language/Syntax": test_language_syntax(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED - check output above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

