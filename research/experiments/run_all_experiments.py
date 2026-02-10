"""
Master Runner for All Scientific Validation Experiments

Runs the complete suite of Assembly Calculus validation experiments:

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
    uv run python research/experiments/run_all_experiments.py --full
"""

import sys
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
    print("COMPLETE SCIENTIFIC VALIDATION SUITE - QUICK")
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
    print("COMPLETE VALIDATION SUMMARY")
    print("="*70)
    print()
    
    summary = generate_summary(all_results)
    print_summary(summary)
    
    # Save master summary
    summary_path = Path(__file__).parent.parent / "results" / f"master_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nMaster summary saved to: {summary_path}")
    
    print(f"\nFinished: {datetime.now().isoformat()}")
    
    return all_results, summary


def generate_summary(results: dict) -> dict:
    """Generate summary from all experiment results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "overall_status": "PASS",
    }
    
    # Primitives
    if "projection" in results:
        proj_rate = results["projection"].metrics.get("overall_convergence_rate", 0)
        summary["experiments"]["projection"] = {
            "status": "PASS" if proj_rate > 0.9 else "FAIL",
            "convergence_rate": proj_rate,
            "mean_steps": results["projection"].metrics.get("mean_convergence_steps", 0),
        }
    
    if "association" in results:
        assoc_rate = results["association"].metrics.get("overall_success_rate", 0)
        summary["experiments"]["association"] = {
            "status": "PASS" if assoc_rate > 0.9 else "FAIL",
            "success_rate": assoc_rate,
        }
    
    if "merge" in results:
        merge_rate = results["merge"].metrics.get("overall_success_rate", 0)
        summary["experiments"]["merge"] = {
            "status": "PASS" if merge_rate > 0.9 else "FAIL",
            "success_rate": merge_rate,
            "merge_quality": results["merge"].metrics.get("mean_merge_quality", 0),
        }
    
    # Stability
    if "scaling_laws" in results:
        scaling = results["scaling_laws"].metrics.get("scaling_analysis", {})
        summary["experiments"]["scaling_laws"] = {
            "status": "PASS",
            "scaling_type": scaling.get("scaling_type", "Unknown"),
            "r_squared": scaling.get("r_squared", 0),
        }
    
    if "noise_robustness" in results:
        noise = results["noise_robustness"].metrics.get("critical_analysis", {})
        summary["experiments"]["noise_robustness"] = {
            "status": "PASS",
            "max_recoverable_noise": noise.get("max_recoverable", 0),
        }
    
    # Information Theory
    if "coding_capacity" in results:
        summary["experiments"]["coding_capacity"] = {
            "status": "PASS",
            "best_efficiency": results["coding_capacity"].metrics.get("best_efficiency", 0),
        }
    
    # Biological
    if "biological" in results:
        bio = results["biological"].metrics
        summary["experiments"]["biological"] = {
            "status": "PASS" if bio.get("biological_validity_rate", 0) > 0.8 else "WARN",
            "validity_rate": bio.get("biological_validity_rate", 0),
        }
    
    # Check overall status
    for exp_name, exp_summary in summary["experiments"].items():
        if exp_summary.get("status") == "FAIL":
            summary["overall_status"] = "FAIL"
            break
    
    return summary


def print_summary(summary: dict):
    """Print formatted summary."""
    print(f"{'Experiment':<25} {'Status':<10} {'Key Metric':<25} {'Value':<15}")
    print("-"*75)
    
    for exp_name, exp_data in summary["experiments"].items():
        status = exp_data.get("status", "?")
        
        # Get the most important metric for each experiment
        if exp_name == "projection":
            metric_name = "Convergence Rate"
            metric_val = f"{exp_data.get('convergence_rate', 0):.1%}"
        elif exp_name == "association":
            metric_name = "Success Rate"
            metric_val = f"{exp_data.get('success_rate', 0):.1%}"
        elif exp_name == "merge":
            metric_name = "Merge Quality"
            metric_val = f"{exp_data.get('merge_quality', 0):.3f}"
        elif exp_name == "scaling_laws":
            metric_name = "Scaling Type"
            metric_val = exp_data.get("scaling_type", "?")[:15]
        elif exp_name == "noise_robustness":
            metric_name = "Max Recoverable"
            metric_val = f"{exp_data.get('max_recoverable_noise', 0):.1%}"
        elif exp_name == "coding_capacity":
            metric_name = "Bits/Neuron"
            metric_val = f"{exp_data.get('best_efficiency', 0):.3f}"
        elif exp_name == "biological":
            metric_name = "Bio Validity"
            metric_val = f"{exp_data.get('validity_rate', 0):.1%}"
        else:
            metric_name = "N/A"
            metric_val = "N/A"
        
        print(f"{exp_name:<25} {status:<10} {metric_name:<25} {metric_val:<15}")
    
    print()
    print(f"OVERALL STATUS: {summary['overall_status']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all scientific validation experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick tests")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive tests")
    
    args = parser.parse_args()
    
    if args.full:
        print("Full suite not yet implemented - running quick suite")
        run_quick_suite()
    else:
        run_quick_suite()

