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
import inspect
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


QUICK_EXPERIMENTS = (
    ('projection', ProjectionExperiment,
     {'n_neurons_range': [1000, 10000],
      'k_active_range': [10, 50],
      'p_connect_range': [0.05, 0.1],
      'beta_range': [0.1],
      'n_trials': 3}),
    ('association', AssociationExperiment,
     {'n_neurons_range': [1000, 5000],
      'k_active_range': [50],
      'p_connect_range': [0.1],
      'beta_range': [0.1],
      'n_trials': 3}),
    ('merge', MergeExperiment,
     {'n_neurons_range': [1000, 5000],
      'k_active_range': [50],
      'p_connect_range': [0.1],
      'beta_range': [0.1],
      'n_trials': 3}),
    ('phase_diagram', PhaseDiagramExperiment,
     {'n_neurons_range': [1000, 5000],
      'sparsity_range': [0.01, 0.05, 0.1],
      'p_connect_range': [0.1],
      'beta_range': [0.1],
      'n_trials': 3}),
    ('scaling_laws', ScalingLawsExperiment,
     {'n_neurons_range': [500, 1000, 5000, 10000, 50000],
      'fixed_sparsity': 0.05,
      'n_trials': 5}),
    ('noise_robustness', NoiseRobustnessExperiment,
     {'n_neurons': 5000,
      'k_active': 50,
      'noise_levels': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
      'n_trials': 5}),
    ('coding_capacity', CodingCapacityExperiment,
     {'n_neurons_range': [1000, 5000],
      'k_active_range': [20, 50],
      'n_assemblies_to_test': [5, 10, 20]}),
    ('biological', BiologicalParameterExperiment,
     {'test_cortical': True,
      'test_hippocampal': True,
      'test_cerebellar': False,
      'n_steps': 50}),
)


def validate_suite(configurations):
    """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#legacy-experiment-configuration"""
    if not configurations:
        raise ValueError("experiment suite must not be empty")
    errors = []
    seen = set()
    for name, factory, parameters in configurations:
        if name in seen:
            errors.append(f"{name}: duplicate experiment name")
        seen.add(name)
        signature = inspect.signature(factory.run)
        unsupported = sorted(set(parameters) - set(signature.parameters))
        if unsupported:
            errors.append(f"{name}: unsupported parameters {', '.join(unsupported)}")
        else:
            try:
                signature.bind(None, **parameters)
            except TypeError as error:
                errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("Invalid experiment suite; no experiments started:\n" + "\n".join(errors))


def run_quick_suite():
    """Validate every declared call before any experiment is constructed or run."""
    validate_suite(QUICK_EXPERIMENTS)
    print("LEGACY EXPERIMENT SUITE - QUICK - SCIENTIFIC STATUS VOID")
    all_results = {}
    for name, factory, parameters in QUICK_EXPERIMENTS:
        experiment = factory(verbose=True)
        all_results[name] = experiment.run(**parameters)
        experiment.save_result(all_results[name], "_quick")
    summary = generate_summary(all_results)
    print_summary(summary)
    summary_path = Path(__file__).parent.parent / "results" / f"master_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_new_document(summary_path, summary)
    print(f"Master summary saved to: {summary_path}")
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
