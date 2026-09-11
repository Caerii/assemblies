"""
Legacy aggregate launcher (not a registered validation protocol)

Historical experiment inventory; quick outputs are scientifically VOID.
Several call configurations still require migration to research.runner:

1. PRIMITIVES
   - Projection convergence
   - Association binding
   - Merge composition

2. STABILITY
   - Phase diagram mapping
   - Scaling laws
   - Noise robustness

3. INFORMATION THEORY
   - Coding capacity

4. BIOLOGICAL VALIDATION
   - Parameter validation against literature

Usage:
    uv run python research/experiments/run_all_experiments.py --quick
    # --full is unsupported and fails before computation.
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from research.json_documents import write_new_document

# Import all experiments
from research.experiments.primitives.test_projection import ProjectionExperiment
from research.experiments.primitives.test_association import AssociationExperiment
from research.experiments.primitives.test_merge import MergeExperiment
from research.experiments.stability.test_phase_diagram import PhaseDiagramExperiment
from research.experiments.stability.test_scaling_laws import ScalingLawsExperiment
from research.experiments.stability.test_noise_robustness import NoiseRobustnessExperiment
from research.experiments.information_theory.test_coding_capacity import CodingCapacityExperiment
from research.experiments.biological_validation.test_biological_parameters import BiologicalParameterExperiment


def run_quick_suite():
    """Run quick version of all experiments (~5-10 minutes)."""
    print("="*70)
    print("LEGACY EXPERIMENT SUITE - QUICK - SCIENTIFIC STATUS VOID")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    all_results = {}
    
    # =========================================================================
    # 1. PRIMITIVES
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 1: PRIMITIVE VALIDATION")
    print("="*70)
    
    # Projection
    print("\n--- 1.1 Projection Convergence ---")
    exp = ProjectionExperiment(verbose=True)
    all_results["projection"] = exp.run(
        n_neurons_range=[1000, 10000],
        k_active_range=[10, 50],
        p_connect_range=[0.05, 0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(all_results["projection"], "_quick")
    
    # Association
    print("\n--- 1.2 Association Binding ---")
    exp = AssociationExperiment(verbose=True)
    all_results["association"] = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[50],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(all_results["association"], "_quick")
    
    # Merge
    print("\n--- 1.3 Merge Composition ---")
    exp = MergeExperiment(verbose=True)
    all_results["merge"] = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[50],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(all_results["merge"], "_quick")
    
    # =========================================================================
    # 2. STABILITY
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 2: STABILITY ANALYSIS")
    print("="*70)
    
    # Phase Diagram
    print("\n--- 2.1 Phase Diagram ---")
    exp = PhaseDiagramExperiment(verbose=True)
    all_results["phase_diagram"] = exp.run(
        n_neurons_range=[1000, 5000],
        sparsity_range=[0.01, 0.05, 0.1],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    exp.save_result(all_results["phase_diagram"], "_quick")
    
    # Scaling Laws
    print("\n--- 2.2 Scaling Laws ---")
    exp = ScalingLawsExperiment(verbose=True)
    all_results["scaling_laws"] = exp.run(
        n_neurons_range=[500, 1000, 5000, 10000, 50000],
        fixed_sparsity=0.05,
        n_trials=5,
    )
    exp.save_result(all_results["scaling_laws"], "_quick")
    
    # Noise Robustness
    print("\n--- 2.3 Noise Robustness ---")
    exp = NoiseRobustnessExperiment(verbose=True)
    all_results["noise_robustness"] = exp.run(
        n_neurons=5000,
        k_active=50,
        noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        n_trials=5,
    )
    exp.save_result(all_results["noise_robustness"], "_quick")
    
    # =========================================================================
    # 3. INFORMATION THEORY
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 3: INFORMATION THEORY")
    print("="*70)
    
    # Coding Capacity
    print("\n--- 3.1 Coding Capacity ---")
    exp = CodingCapacityExperiment(verbose=True)
    all_results["coding_capacity"] = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[20, 50],
        n_assemblies_to_test=[5, 10, 20],
    )
    exp.save_result(all_results["coding_capacity"], "_quick")
    
    # =========================================================================
    # 4. BIOLOGICAL VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("SECTION 4: BIOLOGICAL VALIDATION")
    print("="*70)
    
    # Biological Parameters
    print("\n--- 4.1 Biological Parameter Validation ---")
    exp = BiologicalParameterExperiment(verbose=True)
    all_results["biological"] = exp.run(
        test_cortical=True,
        test_hippocampal=True,
        test_cerebellar=False,
        n_steps=50,
    )
    exp.save_result(all_results["biological"], "_quick")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUICK EXECUTION SUMMARY - SCIENTIFIC STATUS VOID")
    print("="*70)
    print()
    
    summary = generate_summary(all_results)
    print_summary(summary)
    
    # Save master summary
    summary_path = Path(__file__).parent.parent / "results" / f"master_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_new_document(summary_path, summary)
    print(f"\nMaster summary saved to: {summary_path}")
    
    print(f"\nFinished: {datetime.now().isoformat()}")
    
    return all_results, summary


def generate_summary(results: dict) -> dict:
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-aggregate-summary-2026-09-10

    Quick-run measurements are VOID for scientific adoption, even if execution succeeds.
    """
    if not results:
        raise ValueError("cannot summarize an empty experiment suite")
    experiments = {}
    for name, result in results.items():
        record = result.to_dict()
        experiments[name] = {
            "execution_success": record["success"],
            "error_message": record["error_message"],
            "scientific_status": "VOID",
            "metrics": record["metrics"],
            "parameters": record["parameters"],
        }
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "smoke",
        "scientific_status": "VOID",
        "execution_success": all(item["execution_success"] for item in experiments.values()),
        "experiments": experiments,
    }


def print_summary(summary: dict):
    """Print execution status without inferring a scientific verdict from metrics."""
    for name, item in summary["experiments"].items():
        execution = "completed" if item["execution_success"] else "failed"
        print(f"{name}: execution {execution}; scientific status {item['scientific_status']}")
        if item["error_message"]:
            print(f"  {item['error_message']}")
        print(f"  metrics: {item['metrics']}")
    print(f"Scientific status: {summary['scientific_status']} (quick suite)")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run legacy quick experiments (scientific status VOID)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run quick tests")
    mode.add_argument("--full", action="store_true", help="Unsupported; refuses to substitute a quick run")
    args = parser.parse_args(argv)
    if args.full:
        parser.error("full suite is not implemented; choose a registered protocol with python -m research.runner")
    return run_quick_suite()


if __name__ == "__main__":
    main()
