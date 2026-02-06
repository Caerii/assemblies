"""
Scaling Laws Experiment

Scientific Questions:
1. How does convergence time scale with network size N?
2. Is it O(log N) as theory predicts, or something else?
3. How does assembly stability scale with N?
4. Are there universal scaling exponents?

Expected Results:
- Convergence time should scale as O(log N) or O(log k)
- Assembly stability should be independent of N (for fixed k/n ratio)
- Scaling exponents should be universal across parameter ranges
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from scipy import stats

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap
)

import brain as brain_module


@dataclass
class ScalingConfig:
    """Configuration for scaling test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    max_rounds: int = 100


class ScalingLawsExperiment(ExperimentBase):
    """
    Measure how assembly properties scale with network size.
    
    Hypothesis: Convergence time scales as O(log N) or O(log k),
    making Assembly Calculus efficient even at large scales.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="scaling_laws",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose
        )
    
    def measure_convergence_time(
        self, 
        config: ScalingConfig,
        n_trials: int = 5
    ) -> Dict[str, Any]:
        """Measure convergence time for a configuration."""
        convergence_times = []
        final_stabilities = []
        
        for trial in range(n_trials):
            b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial)
            b.add_stimulus("STIM", config.k_active)
            b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
            
            winner_history = []
            converged_at = config.max_rounds
            
            for round_idx in range(config.max_rounds):
                b.project(
                    areas_by_stim={"STIM": ["TARGET"]},
                    dst_areas_by_src_area={}
                )
                winners = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
                winner_history.append(winners.copy())
                
                # Check convergence
                if len(winner_history) >= 3:
                    overlaps = [
                        measure_overlap(winner_history[-i-1], winner_history[-i-2])
                        for i in range(2)
                    ]
                    if all(o > 0.98 for o in overlaps):
                        converged_at = round_idx + 1
                        break
            
            convergence_times.append(converged_at)
            
            # Measure final stability
            if len(winner_history) >= 2:
                final_stabilities.append(
                    measure_overlap(winner_history[-1], winner_history[-2])
                )
            else:
                final_stabilities.append(0.0)
        
        return {
            "mean_convergence_time": np.mean(convergence_times),
            "std_convergence_time": np.std(convergence_times),
            "min_convergence_time": np.min(convergence_times),
            "max_convergence_time": np.max(convergence_times),
            "mean_final_stability": np.mean(final_stabilities),
            "convergence_times": convergence_times,
        }
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        fixed_sparsity: float = 0.05,  # k/n ratio
        p_connect: float = 0.1,
        beta: float = 0.1,
        n_trials: int = 10,
        **kwargs
    ) -> ExperimentResult:
        """
        Measure scaling laws across network sizes.
        
        Args:
            n_neurons_range: Network sizes to test (should span orders of magnitude)
            fixed_sparsity: Fixed k/n ratio to maintain
            p_connect: Connection probability
            beta: Plasticity parameter
            n_trials: Trials per configuration
        """
        self._start_timer()
        
        if n_neurons_range is None:
            # Span 3 orders of magnitude
            n_neurons_range = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        
        self.log("Starting scaling laws experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  fixed_sparsity: {fixed_sparsity}")
        self.log(f"  p_connect: {p_connect}")
        self.log(f"  beta: {beta}")
        
        all_results = []
        
        for n in n_neurons_range:
            k = max(1, int(n * fixed_sparsity))
            
            self.log(f"\n  Testing n={n}, k={k}")
            
            config = ScalingConfig(
                n_neurons=n,
                k_active=k,
                p_connect=p_connect,
                beta=beta
            )
            
            try:
                metrics = self.measure_convergence_time(config, n_trials)
                
                all_results.append({
                    "n_neurons": n,
                    "k_active": k,
                    "log_n": np.log10(n),
                    "log_k": np.log10(k),
                    **metrics,
                })
                
                self.log(f"    Conv. time: {metrics['mean_convergence_time']:.1f} +/- {metrics['std_convergence_time']:.1f}")
                
            except Exception as e:
                self.log(f"    Failed: {e}")
                all_results.append({
                    "n_neurons": n,
                    "k_active": k,
                    "error": str(e),
                })
        
        duration = self._stop_timer()
        
        # Fit scaling law
        scaling_analysis = self._fit_scaling_law(all_results)
        
        summary = {
            "total_configurations": len(all_results),
            "scaling_analysis": scaling_analysis,
        }
        
        self.log(f"\n{'='*60}")
        self.log("SCALING LAWS SUMMARY:")
        self.log(f"  Scaling exponent (vs log N): {scaling_analysis.get('exponent_log_n', 'N/A'):.3f}")
        self.log(f"  R-squared: {scaling_analysis.get('r_squared', 'N/A'):.3f}")
        self.log(f"  Scaling type: {scaling_analysis.get('scaling_type', 'N/A')}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "fixed_sparsity": fixed_sparsity,
                "p_connect": p_connect,
                "beta": beta,
                "n_trials": n_trials,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )
        
        return result
    
    def _fit_scaling_law(self, results: List[Dict]) -> Dict[str, Any]:
        """Fit scaling law to convergence time data."""
        valid_results = [r for r in results if "error" not in r]
        
        if len(valid_results) < 3:
            return {"error": "Not enough data points"}
        
        log_n = np.array([r["log_n"] for r in valid_results])
        conv_times = np.array([r["mean_convergence_time"] for r in valid_results])
        
        # Fit: convergence_time = a * log(N) + b
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, conv_times)
        
        # Determine scaling type
        if abs(slope) < 0.5:
            scaling_type = "O(1) - Constant time"
        elif slope < 2:
            scaling_type = "O(log N) - Logarithmic"
        elif slope < 5:
            scaling_type = "O(log^2 N) - Polylogarithmic"
        else:
            scaling_type = "O(N^alpha) - Polynomial"
        
        return {
            "exponent_log_n": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "std_err": std_err,
            "scaling_type": scaling_type,
            "fit_equation": f"T = {slope:.2f} * log10(N) + {intercept:.2f}",
        }


def run_quick_test():
    """Run quick scaling test."""
    print("="*60)
    print("QUICK TEST: Scaling Laws")
    print("="*60)
    
    exp = ScalingLawsExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[500, 1000, 5000, 10000, 50000],
        fixed_sparsity=0.05,
        n_trials=5,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scaling Laws Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = ScalingLawsExperiment(verbose=True)
        result = exp.run(n_trials=10)
        exp.save_result(result, "_full")

