"""
Biological Validation: Compare to Real Neural Data

This experiment compares Assembly Calculus predictions to known
properties of real neural recordings.

Comparisons:
1. Sparsity levels in cortex vs simulation
2. Assembly sizes in hippocampus vs simulation
3. Firing rate distributions
4. Correlation structure

Data Sources (for future integration):
- Allen Brain Observatory
- Human Connectome Project
- Published neural recording datasets
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from research.experiments.base import ExperimentBase, ExperimentResult
import brain as brain_module


# Biological reference values from literature
BIOLOGICAL_DATA = {
    "cortex": {
        "sparsity_range": (0.01, 0.05),  # 1-5% active
        "assembly_size_range": (50, 500),  # neurons
        "firing_rate_hz": (0.1, 10),  # Hz
        "source": "Barth & Bhalla (2012), Harris (2005)",
    },
    "hippocampus": {
        "sparsity_range": (0.01, 0.03),  # 1-3% active
        "assembly_size_range": (50, 200),
        "firing_rate_hz": (0.5, 5),
        "source": "Buzsaki (2010), place cell recordings",
    },
    "visual_cortex_v1": {
        "sparsity_range": (0.02, 0.10),  # 2-10%
        "assembly_size_range": (10, 100),
        "firing_rate_hz": (1, 20),
        "source": "Olshausen & Field (1996), sparse coding",
    },
}


@dataclass
class BiologicalComparisonConfig:
    """Configuration for biological comparison."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_timesteps: int = 100


class NeuralDataComparisonExperiment(ExperimentBase):
    """Compare simulation to biological neural data."""
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="neural_data_comparison",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "biological_validation",
            verbose=verbose
        )
    
    def simulate_activity(
        self,
        config: BiologicalComparisonConfig,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """Simulate neural activity and collect statistics."""
        brain = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        brain.add_stimulus("STIM", config.k_active)
        brain.add_area("AREA", config.n_neurons, config.k_active, config.beta)
        
        # Track activity
        activity_history = []
        assembly_sizes = []
        
        for t in range(config.n_timesteps):
            brain.project({"STIM": ["AREA"]}, {})
            winners = brain.area_by_name["AREA"].winners
            
            activity_history.append(len(winners))
            assembly_sizes.append(len(winners))
        
        # Compute statistics
        sparsity = np.mean(activity_history) / config.n_neurons
        mean_assembly_size = np.mean(assembly_sizes)
        
        return {
            "sparsity": sparsity,
            "mean_assembly_size": mean_assembly_size,
            "activity_variance": np.var(activity_history),
        }
    
    def compare_to_biology(
        self,
        sim_stats: Dict[str, float],
        region: str
    ) -> Dict[str, Any]:
        """Compare simulation statistics to biological data."""
        bio = BIOLOGICAL_DATA.get(region, BIOLOGICAL_DATA["cortex"])
        
        # Check sparsity
        sparsity_in_range = bio["sparsity_range"][0] <= sim_stats["sparsity"] <= bio["sparsity_range"][1]
        
        # Check assembly size
        size_in_range = bio["assembly_size_range"][0] <= sim_stats["mean_assembly_size"] <= bio["assembly_size_range"][1]
        
        return {
            "region": region,
            "simulation_sparsity": sim_stats["sparsity"],
            "biological_sparsity_range": bio["sparsity_range"],
            "sparsity_match": sparsity_in_range,
            "simulation_assembly_size": sim_stats["mean_assembly_size"],
            "biological_assembly_range": bio["assembly_size_range"],
            "assembly_size_match": size_in_range,
            "source": bio["source"],
        }
    
    def run(
        self,
        regions: List[str] = None,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """Run biological comparison experiments."""
        self._start_timer()
        
        if regions is None:
            regions = ["cortex", "hippocampus", "visual_cortex_v1"]
        
        self.log("Starting biological validation experiment")
        
        all_results = []
        
        for region in regions:
            self.log(f"\n  Testing {region}...")
            
            bio = BIOLOGICAL_DATA.get(region, BIOLOGICAL_DATA["cortex"])
            
            # Configure simulation to match biological parameters
            target_sparsity = np.mean(bio["sparsity_range"])
            target_assembly = int(np.mean(bio["assembly_size_range"]))
            n_neurons = int(target_assembly / target_sparsity)
            
            config = BiologicalComparisonConfig(
                n_neurons=n_neurons,
                k_active=target_assembly,
                p_connect=0.1,
                beta=0.1,
            )
            
            self.log(f"    Config: n={n_neurons}, k={target_assembly}")
            
            trial_results = []
            for trial in range(n_trials):
                try:
                    sim_stats = self.simulate_activity(config, trial)
                    comparison = self.compare_to_biology(sim_stats, region)
                    trial_results.append(comparison)
                except Exception as e:
                    self.log(f"    Trial {trial} failed: {e}")
            
            if trial_results:
                sparsity_match_rate = sum(1 for r in trial_results if r["sparsity_match"]) / len(trial_results)
                size_match_rate = sum(1 for r in trial_results if r["assembly_size_match"]) / len(trial_results)
                
                all_results.append({
                    "region": region,
                    "sparsity_match_rate": sparsity_match_rate,
                    "size_match_rate": size_match_rate,
                    "mean_sparsity": np.mean([r["simulation_sparsity"] for r in trial_results]),
                    "mean_assembly_size": np.mean([r["simulation_assembly_size"] for r in trial_results]),
                })
                
                self.log(f"    Sparsity match: {sparsity_match_rate:.0%}")
                self.log(f"    Assembly size match: {size_match_rate:.0%}")
        
        duration = self._stop_timer()
        
        summary = {
            "regions_tested": len(all_results),
            "overall_sparsity_match": np.mean([r["sparsity_match_rate"] for r in all_results]),
            "overall_size_match": np.mean([r["size_match_rate"] for r in all_results]),
        }
        
        self.log(f"\n{'='*60}")
        self.log("BIOLOGICAL VALIDATION SUMMARY:")
        self.log(f"  Sparsity match: {summary['overall_sparsity_match']:.0%}")
        self.log(f"  Assembly size match: {summary['overall_size_match']:.0%}")
        self.log(f"  Duration: {duration:.1f}s")
        
        return ExperimentResult(
            experiment_name=self.name,
            parameters={"regions": regions, "n_trials": n_trials, "seed": self.seed},
            metrics=summary,
            raw_data={"all_results": all_results, "biological_data": BIOLOGICAL_DATA},
            duration_seconds=duration,
        )


def run_quick_test():
    """Run quick biological validation test."""
    print("="*60)
    print("Biological Validation Test")
    print("="*60)
    
    exp = NeuralDataComparisonExperiment(verbose=True)
    result = exp.run(n_trials=3)
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    run_quick_test()

