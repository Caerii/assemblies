"""
Noise Robustness Experiment

Scientific Questions:
1. How much noise can assemblies tolerate before breaking down?
2. Is there a critical noise level (phase transition)?
3. How does noise robustness scale with assembly size k?
4. Does Hebbian plasticity help recover from noise?

Expected Results:
- Assemblies should show graceful degradation with noise
- Larger assemblies (higher k) should be more robust
- Plasticity should help recover from moderate noise
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
)

import brain as brain_module


@dataclass
class NoiseConfig:
    """Configuration for noise robustness test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    noise_level: float  # Fraction of neurons to randomly flip
    n_projection_rounds: int = 10
    n_recovery_rounds: int = 10


class NoiseRobustnessExperiment(ExperimentBase):
    """
    Test assembly robustness to noise perturbations.
    
    Hypothesis: Assemblies act as attractors in neural state space,
    able to recover from perturbations up to some critical noise level.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="noise_robustness",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose
        )
    
    def add_noise_to_assembly(
        self, 
        winners: np.ndarray, 
        n_neurons: int,
        noise_level: float
    ) -> np.ndarray:
        """
        Add noise to assembly by randomly replacing some winners.
        
        Args:
            winners: Original winner indices
            n_neurons: Total neurons in area
            noise_level: Fraction of winners to replace with random neurons
        
        Returns:
            Noisy winner array
        """
        k = len(winners)
        n_to_flip = int(k * noise_level)
        
        if n_to_flip == 0:
            return winners.copy()
        
        # Select indices to flip
        flip_indices = self.rng.choice(k, size=n_to_flip, replace=False)
        
        # Get non-winner neurons
        winner_set = set(winners.tolist())
        non_winners = [i for i in range(n_neurons) if i not in winner_set]
        
        if len(non_winners) < n_to_flip:
            return winners.copy()
        
        # Replace with random non-winners
        replacements = self.rng.choice(non_winners, size=n_to_flip, replace=False)
        
        noisy_winners = winners.copy()
        noisy_winners[flip_indices] = replacements
        
        return noisy_winners
    
    def run_single_trial(
        self, 
        config: NoiseConfig,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single noise robustness trial.
        
        1. Establish stable assembly
        2. Inject noise
        3. Allow recovery through continued projection
        4. Measure recovery
        """
        # Create brain and establish assembly
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        b.add_stimulus("STIM", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        # Establish stable assembly
        for _ in range(config.n_projection_rounds):
            b.project(
                areas_by_stim={"STIM": ["TARGET"]},
                dst_areas_by_src_area={}
            )
        
        original_assembly = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Inject noise
        noisy_assembly = self.add_noise_to_assembly(
            original_assembly, 
            config.n_neurons, 
            config.noise_level
        )
        
        # Measure immediate damage
        immediate_overlap = measure_overlap(original_assembly, noisy_assembly)
        
        # Set noisy assembly as current winners
        b.area_by_name["TARGET"].winners = noisy_assembly.tolist()
        
        # Allow recovery through continued projection
        recovery_history = [immediate_overlap]
        
        for _ in range(config.n_recovery_rounds):
            b.project(
                areas_by_stim={"STIM": ["TARGET"]},
                dst_areas_by_src_area={}
            )
            current = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            recovery_history.append(measure_overlap(original_assembly, current))
        
        final_overlap = recovery_history[-1]
        recovery_amount = final_overlap - immediate_overlap
        
        return {
            "trial_id": trial_id,
            "noise_level": config.noise_level,
            "immediate_overlap": immediate_overlap,
            "final_overlap": final_overlap,
            "recovery_amount": recovery_amount,
            "full_recovery": final_overlap > 0.95,
            "partial_recovery": final_overlap > immediate_overlap,
            "recovery_history": recovery_history,
        }
    
    def run(
        self,
        n_neurons: int = 10000,
        k_active: int = 100,
        p_connect: float = 0.1,
        beta: float = 0.1,
        noise_levels: List[float] = None,
        n_trials: int = 10,
        **kwargs
    ) -> ExperimentResult:
        """
        Test noise robustness across noise levels.
        
        Args:
            n_neurons: Network size
            k_active: Assembly size
            p_connect: Connection probability
            beta: Plasticity parameter
            noise_levels: List of noise levels to test (0 to 1)
            n_trials: Trials per noise level
        """
        self._start_timer()
        
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        self.log("Starting noise robustness experiment")
        self.log(f"  n_neurons: {n_neurons}")
        self.log(f"  k_active: {k_active}")
        self.log(f"  noise_levels: {noise_levels}")
        
        all_results = []
        
        for noise in noise_levels:
            self.log(f"\n  Testing noise level: {noise:.1%}")
            
            config = NoiseConfig(
                n_neurons=n_neurons,
                k_active=k_active,
                p_connect=p_connect,
                beta=beta,
                noise_level=noise
            )
            
            trial_results = []
            for trial in range(n_trials):
                try:
                    result = self.run_single_trial(config, trial)
                    trial_results.append(result)
                except Exception as e:
                    self.log(f"    Trial {trial} failed: {e}")
            
            if trial_results:
                mean_immediate = np.mean([r["immediate_overlap"] for r in trial_results])
                mean_final = np.mean([r["final_overlap"] for r in trial_results])
                recovery_rate = sum(1 for r in trial_results if r["full_recovery"]) / len(trial_results)
                
                all_results.append({
                    "noise_level": noise,
                    "mean_immediate_overlap": mean_immediate,
                    "mean_final_overlap": mean_final,
                    "mean_recovery": mean_final - mean_immediate,
                    "full_recovery_rate": recovery_rate,
                    "n_trials": len(trial_results),
                    "trial_details": trial_results,
                })
                
                self.log(f"    Immediate: {mean_immediate:.3f}, Final: {mean_final:.3f}, Recovery rate: {recovery_rate:.1%}")
        
        duration = self._stop_timer()
        
        # Find critical noise level
        critical_analysis = self._find_critical_noise(all_results)
        
        summary = {
            "total_noise_levels": len(all_results),
            "critical_analysis": critical_analysis,
        }
        
        self.log(f"\n{'='*60}")
        self.log("NOISE ROBUSTNESS SUMMARY:")
        self.log(f"  Critical noise level: {critical_analysis.get('critical_noise', 'N/A')}")
        self.log(f"  Max recoverable noise: {critical_analysis.get('max_recoverable', 'N/A')}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons": n_neurons,
                "k_active": k_active,
                "p_connect": p_connect,
                "beta": beta,
                "noise_levels": noise_levels,
                "n_trials": n_trials,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )
        
        return result
    
    def _find_critical_noise(self, results: List[Dict]) -> Dict[str, Any]:
        """Find critical noise level where recovery fails."""
        # Sort by noise level
        sorted_results = sorted(results, key=lambda x: x["noise_level"])
        
        # Find where full recovery rate drops below 50%
        critical_noise = None
        max_recoverable = 0.0
        
        for r in sorted_results:
            if r["full_recovery_rate"] >= 0.5:
                max_recoverable = r["noise_level"]
            elif critical_noise is None:
                critical_noise = r["noise_level"]
        
        # Find where final overlap drops below 0.5
        breakdown_noise = None
        for r in sorted_results:
            if r["mean_final_overlap"] < 0.5 and breakdown_noise is None:
                breakdown_noise = r["noise_level"]
        
        return {
            "critical_noise": critical_noise,
            "max_recoverable": max_recoverable,
            "breakdown_noise": breakdown_noise,
            "robustness_profile": [
                {"noise": r["noise_level"], "final_overlap": r["mean_final_overlap"]}
                for r in sorted_results
            ],
        }


def run_quick_test():
    """Run quick noise robustness test."""
    print("="*60)
    print("QUICK TEST: Noise Robustness")
    print("="*60)
    
    exp = NoiseRobustnessExperiment(verbose=True)
    
    result = exp.run(
        n_neurons=5000,
        k_active=50,
        noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        n_trials=5,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Noise Robustness Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = NoiseRobustnessExperiment(verbose=True)
        result = exp.run(n_trials=10)
        exp.save_result(result, "_full")

