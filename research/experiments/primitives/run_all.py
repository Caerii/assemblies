"""
Run all primitive validation experiments.

This script runs the complete suite of Assembly Calculus primitive tests:
1. Projection Convergence
2. Association Binding
3. Merge Composition

Results are saved to research/results/primitives/
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from research.experiments.primitives.test_projection import ProjectionConvergenceExperiment
from research.experiments.primitives.test_association import AssociationBindingExperiment
from research.experiments.primitives.test_merge import MergeCompositionExperiment


def run_quick_suite():
    """Run quick tests for all primitives."""
    print("="*70)
    print("PRIMITIVE VALIDATION SUITE - QUICK TEST")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # 1. Projection
    print("\n" + "="*70)
    print("1. PROJECTION CONVERGENCE")
    print("="*70)
    exp = ProjectionConvergenceExperiment(verbose=True)
    results["projection"] = exp.run(
        n_neurons_range=[1000, 10000],
        k_active_range=[10, 50],
        p_connect_range=[0.05, 0.1],
        beta_range=[0.05, 0.1],
        n_trials=3,
    )
    exp.save_result(results["projection"], "_quick")
    
    # 2. Association
    print("\n" + "="*70)
    print("2. ASSOCIATION BINDING")
    print("="*70)
    exp = AssociationBindingExperiment(verbose=True)
    results["association"] = exp.run(
        n_neurons_range=[1000, 10000],
        k_active_range=[50],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(results["association"], "_quick")
    
    # 3. Merge
    print("\n" + "="*70)
    print("3. MERGE COMPOSITION")
    print("="*70)
    exp = MergeCompositionExperiment(verbose=True)
    results["merge"] = exp.run(
        n_neurons_range=[1000, 10000],
        k_active_range=[50],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(results["merge"], "_quick")
    
    # Summary
    print("\n" + "="*70)
    print("PRIMITIVE VALIDATION SUMMARY")
    print("="*70)
    print()
    print(f"{'Primitive':<20} {'Success Rate':<15} {'Key Metric':<20} {'Value':<10}")
    print("-"*70)
    
    proj_rate = results["projection"].metrics.get("overall_convergence_rate", 0)
    proj_steps = results["projection"].metrics.get("mean_convergence_steps", 0)
    print(f"{'Projection':<20} {proj_rate:.1%}{'':>10} {'Mean Conv. Steps':<20} {proj_steps:.1f}")
    
    assoc_rate = results["association"].metrics.get("overall_success_rate", 0)
    assoc_increase = results["association"].metrics.get("mean_overlap_increase", 0)
    print(f"{'Association':<20} {assoc_rate:.1%}{'':>10} {'Overlap Increase':<20} {assoc_increase:.3f}")
    
    merge_rate = results["merge"].metrics.get("overall_success_rate", 0)
    merge_quality = results["merge"].metrics.get("mean_merge_quality", 0)
    print(f"{'Merge':<20} {merge_rate:.1%}{'':>10} {'Merge Quality':<20} {merge_quality:.3f}")
    
    print()
    print(f"Finished: {datetime.now().isoformat()}")
    
    # Overall assessment
    all_passed = proj_rate > 0.9 and assoc_rate > 0.9 and merge_rate > 0.9
    print()
    if all_passed:
        print("[PASS] ALL PRIMITIVES VALIDATED SUCCESSFULLY")
    else:
        print("[WARN] SOME PRIMITIVES NEED ATTENTION")
    
    return results


def run_full_suite():
    """Run comprehensive tests for all primitives."""
    print("="*70)
    print("PRIMITIVE VALIDATION SUITE - FULL TEST")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # 1. Projection - comprehensive parameter sweep
    print("\n" + "="*70)
    print("1. PROJECTION CONVERGENCE (FULL)")
    print("="*70)
    exp = ProjectionConvergenceExperiment(verbose=True)
    results["projection"] = exp.run(
        n_neurons_range=[1000, 5000, 10000, 50000],
        k_active_range=[10, 25, 50, 100],
        p_connect_range=[0.01, 0.05, 0.1],
        beta_range=[0.05, 0.1, 0.2],
        n_trials=5,
    )
    exp.save_result(results["projection"], "_full")
    
    # 2. Association
    print("\n" + "="*70)
    print("2. ASSOCIATION BINDING (FULL)")
    print("="*70)
    exp = AssociationBindingExperiment(verbose=True)
    results["association"] = exp.run(
        n_neurons_range=[1000, 5000, 10000],
        k_active_range=[25, 50, 100],
        p_connect_range=[0.05, 0.1],
        beta_range=[0.05, 0.1],
        n_trials=5,
    )
    exp.save_result(results["association"], "_full")
    
    # 3. Merge
    print("\n" + "="*70)
    print("3. MERGE COMPOSITION (FULL)")
    print("="*70)
    exp = MergeCompositionExperiment(verbose=True)
    results["merge"] = exp.run(
        n_neurons_range=[1000, 5000, 10000],
        k_active_range=[25, 50, 100],
        p_connect_range=[0.05, 0.1],
        beta_range=[0.05, 0.1],
        n_trials=5,
    )
    exp.save_result(results["merge"], "_full")
    
    # Summary
    print("\n" + "="*70)
    print("PRIMITIVE VALIDATION SUMMARY (FULL)")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run primitive validation suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive tests")
    
    args = parser.parse_args()
    
    if args.full:
        run_full_suite()
    else:
        run_quick_suite()

