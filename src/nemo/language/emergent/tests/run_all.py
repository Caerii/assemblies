#!/usr/bin/env python
"""
Run All NEMO Emergent Language Tests
====================================

Version: 2.0.0
Date: 2025-11-30

Runs all test modules using a SHARED trained model for speed.
Model is trained ONCE and reused across all test suites.

Run:
    cd src/nemo/language/emergent/tests
    uv run python run_all.py
    
Or from project root:
    uv run python -m nemo.language.emergent.tests.run_all
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def main():
    """Run all test suites with shared model."""
    print("="*70)
    print("NEMO EMERGENT LANGUAGE LEARNER - FULL TEST SUITE")
    print("="*70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Train shared model ONCE
    from src.nemo.language.emergent.tests.shared_model import get_trained_learner, get_training_time
    learner = get_trained_learner(epochs=3)
    
    # Import and run each test module with shared learner
    from src.nemo.language.emergent.tests import test_training
    from src.nemo.language.emergent.tests import test_parser
    from src.nemo.language.emergent.tests import test_comprehension
    
    test_modules = [
        ("Training Tests", test_training),
        ("Parser Tests", test_parser),
        ("Comprehension Tests", test_comprehension),
    ]
    
    all_results = []
    
    for name, module in test_modules:
        print("\n" + "="*70)
        print(f"RUNNING: {name}")
        print("="*70)
        
        try:
            # Pass shared learner to tests
            if hasattr(module, 'run_all_tests_with_learner'):
                success = module.run_all_tests_with_learner(learner)
            else:
                success = module.run_all_tests(learner)
            all_results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append((name, False))
    
    # Final summary
    elapsed = time.time() - start_time
    training_time = get_training_time()
    test_time = elapsed - training_time
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in all_results if r)
    total = len(all_results)
    
    for name, result in all_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    print(f"Test Suites: {passed}/{total} passed")
    print(f"Training Time: {training_time:.1f}s (shared model)")
    print(f"Test Time: {test_time:.1f}s")
    print(f"Total Time: {elapsed:.1f}s")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST SUITE(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

