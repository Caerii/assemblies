"""
Projection Primitive Validation

Scientific Questions:
1. Does projection reliably create assemblies in downstream areas?
2. What are the conditions for convergence (n, k, p, beta)?
3. How many steps does convergence take?
4. Is the resulting assembly stable?

Expected Results:
- Projection should converge to stable assembly within O(log n) steps
- Convergence should be robust to noise
- Assembly size should stabilize at k neurons
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
    convergence_metric
)

# Import the brain module
import brain as brain_module


@dataclass
class ProjectionConfig:
    """Configuration for a single projection test."""
    n_neurons: int  # Total neurons per area
    k_active: int   # Number of active neurons (assembly size)
    p_connect: float  # Connection probability
    beta: float     # Plasticity parameter
    max_rounds: int = 50  # Maximum projection rounds


class ProjectionConvergenceExperiment(ExperimentBase):
    """
    Test: Does projection reliably create stable assemblies?
    
    Hypothesis: Given a stimulus projecting to an area, repeated projection
    will converge to a stable assembly of size k within O(log n) steps.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="projection_convergence",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
            verbose=verbose
        )
    
    def run_single_trial(
        self, 
        config: ProjectionConfig,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single projection trial.
        
        Creates a brain with:
        - Source area A with stimulus
        - Target area B receiving projection
        
        Returns metrics about convergence.
        """
        self.log(f"  Trial {trial_id}: n={config.n_neurons}, k={config.k_active}, p={config.p_connect}, beta={config.beta}")
        
        # Create brain
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Add stimulus (source of activation)
        stimulus_name = "STIM"
        b.add_stimulus(stimulus_name, config.k_active)
        
        # Add target area
        target_area = "TARGET"
        b.add_area(target_area, config.n_neurons, config.k_active, config.beta)
        
        # Track assembly evolution
        winner_history = []
        
        # Project stimulus into target area repeatedly
        for round_idx in range(config.max_rounds):
            b.project(
                areas_by_stim={stimulus_name: [target_area]},
                dst_areas_by_src_area={}
            )
            
            # Record winners
            current_winners = np.array(b.area_by_name[target_area].winners, dtype=np.uint32)
            winner_history.append(current_winners.copy())
            
            # Early stopping if converged
            if len(winner_history) >= 5:
                recent_overlaps = [
                    measure_overlap(winner_history[-i-1], winner_history[-i-2])
                    for i in range(4)
                ]
                if all(o > 0.98 for o in recent_overlaps):
                    self.log(f"    Converged at round {round_idx + 1}")
                    break
        
        # Analyze convergence
        conv_metrics = convergence_metric(winner_history)
        
        # Final assembly properties
        final_winners = winner_history[-1] if winner_history else np.array([])
        
        return {
            "trial_id": trial_id,
            "config": {
                "n_neurons": config.n_neurons,
                "k_active": config.k_active,
                "p_connect": config.p_connect,
                "beta": config.beta,
            },
            "converged": conv_metrics["converged"],
            "convergence_step": conv_metrics["convergence_step"],
            "final_stability": conv_metrics["final_stability"],
            "total_rounds": len(winner_history),
            "final_assembly_size": len(final_winners),
            "overlap_history": conv_metrics["overlap_history"],
            "mean_overlap": conv_metrics["mean_overlap"],
        }
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        k_active_range: List[int] = None,
        p_connect_range: List[float] = None,
        beta_range: List[float] = None,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """
        Run projection convergence experiment across parameter ranges.
        
        Args:
            n_neurons_range: List of neuron counts to test
            k_active_range: List of assembly sizes to test
            p_connect_range: List of connection probabilities
            beta_range: List of plasticity values
            n_trials: Number of trials per configuration
        """
        self._start_timer()
        
        # Default parameter ranges
        if n_neurons_range is None:
            n_neurons_range = [1000, 10000, 100000]
        if k_active_range is None:
            k_active_range = [10, 50, 100]
        if p_connect_range is None:
            p_connect_range = [0.01, 0.05, 0.1]
        if beta_range is None:
            beta_range = [0.05, 0.1, 0.2]
        
        self.log(f"Starting projection convergence experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  k_active: {k_active_range}")
        self.log(f"  p_connect: {p_connect_range}")
        self.log(f"  beta: {beta_range}")
        self.log(f"  n_trials: {n_trials}")
        
        all_results = []
        total_configs = len(n_neurons_range) * len(k_active_range) * len(p_connect_range) * len(beta_range)
        config_idx = 0
        
        for n in n_neurons_range:
            for k in k_active_range:
                # Skip invalid configs where k > n
                if k >= n:
                    continue
                    
                for p in p_connect_range:
                    for beta in beta_range:
                        config_idx += 1
                        self.log(f"\nConfig {config_idx}/{total_configs}: n={n}, k={k}, p={p}, beta={beta}")
                        
                        config = ProjectionConfig(
                            n_neurons=n,
                            k_active=k,
                            p_connect=p,
                            beta=beta
                        )
                        
                        trial_results = []
                        for trial in range(n_trials):
                            try:
                                result = self.run_single_trial(config, trial)
                                trial_results.append(result)
                            except Exception as e:
                                self.log(f"    Trial {trial} failed: {e}")
                                trial_results.append({
                                    "trial_id": trial,
                                    "error": str(e),
                                    "converged": False,
                                })
                        
                        # Aggregate trial results
                        successful_trials = [r for r in trial_results if "error" not in r]
                        
                        if successful_trials:
                            convergence_rate = sum(1 for r in successful_trials if r["converged"]) / len(successful_trials)
                            mean_convergence_step = np.mean([r["convergence_step"] for r in successful_trials])
                            mean_final_stability = np.mean([r["final_stability"] for r in successful_trials])
                        else:
                            convergence_rate = 0.0
                            mean_convergence_step = float('inf')
                            mean_final_stability = 0.0
                        
                        all_results.append({
                            "config": {
                                "n_neurons": n,
                                "k_active": k,
                                "p_connect": p,
                                "beta": beta,
                            },
                            "n_trials": n_trials,
                            "successful_trials": len(successful_trials),
                            "convergence_rate": convergence_rate,
                            "mean_convergence_step": mean_convergence_step,
                            "mean_final_stability": mean_final_stability,
                            "trial_details": trial_results,
                        })
        
        duration = self._stop_timer()
        
        # Compute summary statistics
        successful_configs = [r for r in all_results if r["convergence_rate"] > 0]
        
        summary = {
            "total_configurations": len(all_results),
            "successful_configurations": len(successful_configs),
            "overall_convergence_rate": np.mean([r["convergence_rate"] for r in all_results]) if all_results else 0,
            "mean_convergence_steps": np.mean([r["mean_convergence_step"] for r in successful_configs]) if successful_configs else float('inf'),
            "best_config": max(all_results, key=lambda x: x["convergence_rate"]) if all_results else None,
        }
        
        self.log(f"\n{'='*60}")
        self.log(f"SUMMARY:")
        self.log(f"  Total configurations: {summary['total_configurations']}")
        self.log(f"  Successful configs: {summary['successful_configurations']}")
        self.log(f"  Overall convergence rate: {summary['overall_convergence_rate']:.2%}")
        self.log(f"  Mean convergence steps: {summary['mean_convergence_steps']:.1f}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "k_active_range": k_active_range,
                "p_connect_range": p_connect_range,
                "beta_range": beta_range,
                "n_trials": n_trials,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )
        
        return result


def run_quick_test():
    """Run a quick test to verify the experiment works."""
    print("="*60)
    print("QUICK TEST: Projection Convergence")
    print("="*60)
    
    exp = ProjectionConvergenceExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 10000],
        k_active_range=[10, 50],
        p_connect_range=[0.05, 0.1],
        beta_range=[0.05, 0.1],
        n_trials=3,
    )
    
    # Save results
    path = exp.save_result(result, "_quick")
    
    print(f"\nResults saved to: {path}")
    print(f"Convergence rate: {result.metrics['overall_convergence_rate']:.2%}")
    
    return result


def run_full_experiment():
    """Run the full parameter sweep experiment."""
    print("="*60)
    print("FULL EXPERIMENT: Projection Convergence")
    print("="*60)
    
    exp = ProjectionConvergenceExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000, 10000, 50000, 100000],
        k_active_range=[10, 25, 50, 100, 200],
        p_connect_range=[0.01, 0.02, 0.05, 0.1, 0.2],
        beta_range=[0.01, 0.05, 0.1, 0.2],
        n_trials=5,
    )
    
    path = exp.save_result(result, "_full")
    
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Projection Convergence Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_full_experiment()

