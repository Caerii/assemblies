"""
Phase Diagram Mapping Experiment

Scientific Questions:
1. Is there a critical sparsity level where assemblies transition from unstable to stable?
2. How does the critical point depend on network parameters (n, p, beta)?
3. Are there scaling laws near the critical point?

Expected Results:
- Sharp transition in assembly stability at critical k/n ratio
- Power-law scaling near critical point
- Universal behavior across different network sizes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
    convergence_metric
)

import brain as brain_module


@dataclass
class PhaseConfig:
    """Configuration for phase diagram point."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    max_rounds: int = 50
    stability_threshold: float = 0.95


class PhaseDiagramExperiment(ExperimentBase):
    """
    Map the phase diagram of assembly formation.
    
    Hypothesis: There exists a critical sparsity level k_c/n where
    assemblies transition from unstable (random fluctuation) to
    stable (convergent attractor).
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="phase_diagram",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose
        )
    
    def measure_stability(
        self, 
        config: PhaseConfig,
        n_trials: int = 5
    ) -> Dict[str, Any]:
        """
        Measure assembly stability for a given configuration.
        
        Returns:
            Dictionary with stability metrics including:
            - convergence_rate: fraction of trials that converged
            - mean_convergence_time: average steps to converge
            - mean_final_stability: average final overlap stability
            - stability_variance: variance in stability across trials
        """
        trial_results = []
        
        for trial in range(n_trials):
            b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial)
            b.add_stimulus("STIM", config.k_active)
            b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
            
            winner_history = []
            
            for round_idx in range(config.max_rounds):
                b.project(
                    areas_by_stim={"STIM": ["TARGET"]},
                    dst_areas_by_src_area={}
                )
                winners = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
                winner_history.append(winners.copy())
                
                # Early stopping
                if len(winner_history) >= 5:
                    recent_overlaps = [
                        measure_overlap(winner_history[-i-1], winner_history[-i-2])
                        for i in range(4)
                    ]
                    if all(o > config.stability_threshold for o in recent_overlaps):
                        break
            
            conv_metrics = convergence_metric(winner_history)
            trial_results.append({
                "converged": conv_metrics["converged"],
                "convergence_step": conv_metrics["convergence_step"],
                "final_stability": conv_metrics["final_stability"],
            })
        
        converged_trials = [r for r in trial_results if r["converged"]]
        
        return {
            "convergence_rate": len(converged_trials) / n_trials,
            "mean_convergence_time": np.mean([r["convergence_step"] for r in converged_trials]) if converged_trials else float('inf'),
            "mean_final_stability": np.mean([r["final_stability"] for r in trial_results]),
            "stability_variance": np.var([r["final_stability"] for r in trial_results]),
            "n_trials": n_trials,
        }
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        sparsity_range: List[float] = None,  # k/n ratios
        p_connect_range: List[float] = None,
        beta_range: List[float] = None,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """
        Map the phase diagram across parameter space.
        
        Args:
            n_neurons_range: Network sizes to test
            sparsity_range: k/n ratios (assembly size / total neurons)
            p_connect_range: Connection probabilities
            beta_range: Plasticity values
            n_trials: Trials per configuration
        """
        self._start_timer()
        
        # Default parameter ranges
        if n_neurons_range is None:
            n_neurons_range = [1000, 5000, 10000]
        if sparsity_range is None:
            # Test from very sparse to moderately dense
            sparsity_range = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        if p_connect_range is None:
            p_connect_range = [0.05, 0.1]
        if beta_range is None:
            beta_range = [0.05, 0.1]
        
        self.log("Starting phase diagram mapping")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  sparsity (k/n): {sparsity_range}")
        self.log(f"  p_connect: {p_connect_range}")
        self.log(f"  beta: {beta_range}")
        
        all_results = []
        
        for n in n_neurons_range:
            for sparsity in sparsity_range:
                k = max(1, int(n * sparsity))  # At least 1 neuron
                
                if k >= n:
                    continue
                    
                for p in p_connect_range:
                    for beta in beta_range:
                        self.log(f"\n  n={n}, k={k} (sparsity={sparsity:.3f}), p={p}, beta={beta}")
                        
                        config = PhaseConfig(
                            n_neurons=n,
                            k_active=k,
                            p_connect=p,
                            beta=beta
                        )
                        
                        try:
                            metrics = self.measure_stability(config, n_trials)
                            
                            all_results.append({
                                "n_neurons": n,
                                "k_active": k,
                                "sparsity": sparsity,
                                "p_connect": p,
                                "beta": beta,
                                **metrics,
                            })
                            
                            self.log(f"    Conv. rate: {metrics['convergence_rate']:.2f}, "
                                   f"Mean time: {metrics['mean_convergence_time']:.1f}, "
                                   f"Stability: {metrics['mean_final_stability']:.3f}")
                            
                        except Exception as e:
                            self.log(f"    Failed: {e}")
                            all_results.append({
                                "n_neurons": n,
                                "k_active": k,
                                "sparsity": sparsity,
                                "p_connect": p,
                                "beta": beta,
                                "error": str(e),
                                "convergence_rate": 0,
                            })
        
        duration = self._stop_timer()
        
        # Analyze phase transition
        phase_analysis = self._analyze_phase_transition(all_results)
        
        summary = {
            "total_configurations": len(all_results),
            "successful_configurations": len([r for r in all_results if "error" not in r]),
            "phase_analysis": phase_analysis,
        }
        
        self.log(f"\n{'='*60}")
        self.log("PHASE DIAGRAM SUMMARY:")
        self.log(f"  Total configurations: {summary['total_configurations']}")
        self.log(f"  Critical sparsity estimate: {phase_analysis.get('critical_sparsity', 'N/A')}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "sparsity_range": sparsity_range,
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
    
    def _analyze_phase_transition(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results to find phase transition point."""
        # Group by sparsity
        sparsity_groups = {}
        for r in results:
            if "error" in r:
                continue
            s = r["sparsity"]
            if s not in sparsity_groups:
                sparsity_groups[s] = []
            sparsity_groups[s].append(r["convergence_rate"])
        
        # Find critical sparsity (where convergence rate crosses 0.5)
        sorted_sparsities = sorted(sparsity_groups.keys())
        mean_rates = [np.mean(sparsity_groups[s]) for s in sorted_sparsities]
        
        critical_sparsity = None
        for i in range(len(mean_rates) - 1):
            if mean_rates[i] < 0.5 <= mean_rates[i+1]:
                # Linear interpolation
                s1, s2 = sorted_sparsities[i], sorted_sparsities[i+1]
                r1, r2 = mean_rates[i], mean_rates[i+1]
                critical_sparsity = s1 + (0.5 - r1) * (s2 - s1) / (r2 - r1)
                break
            elif mean_rates[i] >= 0.5 > mean_rates[i+1]:
                s1, s2 = sorted_sparsities[i], sorted_sparsities[i+1]
                r1, r2 = mean_rates[i], mean_rates[i+1]
                critical_sparsity = s1 + (0.5 - r1) * (s2 - s1) / (r2 - r1)
                break
        
        return {
            "sparsities": sorted_sparsities,
            "mean_convergence_rates": mean_rates,
            "critical_sparsity": critical_sparsity,
            "all_stable": all(r >= 0.9 for r in mean_rates),
            "all_unstable": all(r <= 0.1 for r in mean_rates),
        }


def run_quick_test():
    """Run quick phase diagram test."""
    print("="*60)
    print("QUICK TEST: Phase Diagram")
    print("="*60)
    
    exp = PhaseDiagramExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000],
        sparsity_range=[0.01, 0.05, 0.1],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase Diagram Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = PhaseDiagramExperiment(verbose=True)
        result = exp.run(n_trials=5)
        exp.save_result(result, "_full")

