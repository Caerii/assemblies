"""
Assembly Distinctiveness Experiment

CRITICAL FINDING: Different stimuli create assemblies with ~95% overlap!

This experiment investigates:
1. How much do assemblies from different stimuli overlap?
2. What parameters affect distinctiveness?
3. Is this a fundamental limitation or a parameter issue?

Scientific Importance:
- If assemblies aren't distinct, the brain can't represent different concepts
- This is the "superposition catastrophe" in neural coding
- Understanding this is crucial for Assembly Calculus validity
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from itertools import combinations

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
)

import brain as brain_module


@dataclass
class DistinctivenessConfig:
    """Configuration for distinctiveness test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_stimuli: int
    n_projection_rounds: int = 15


class AssemblyDistinctivenessExperiment(ExperimentBase):
    """
    Test whether different stimuli create distinct assemblies.
    
    This is a CRITICAL test for Assembly Calculus validity.
    If assemblies overlap too much, the framework can't represent
    distinct concepts.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="assembly_distinctiveness",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "stability",
            verbose=verbose
        )
    
    def create_multiple_assemblies(
        self,
        config: DistinctivenessConfig,
        trial_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """Create assemblies from multiple different stimuli."""
        assemblies = {}
        
        for i in range(config.n_stimuli):
            # Each stimulus gets its own brain to ensure independence
            b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id * 1000 + i)
            stim_name = f"STIM_{i}"
            b.add_stimulus(stim_name, config.k_active)
            b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
            
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            
            assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        return assemblies
    
    def create_assemblies_same_brain(
        self,
        config: DistinctivenessConfig,
        trial_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """Create assemblies from multiple stimuli in the SAME brain."""
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Add all stimuli
        for i in range(config.n_stimuli):
            b.add_stimulus(f"STIM_{i}", config.k_active)
        
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        assemblies = {}
        
        for i in range(config.n_stimuli):
            stim_name = f"STIM_{i}"
            
            # Project this stimulus
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            
            assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        return assemblies
    
    def compute_pairwise_overlaps(
        self,
        assemblies: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute all pairwise overlaps between assemblies."""
        names = list(assemblies.keys())
        overlaps = []
        
        for name1, name2 in combinations(names, 2):
            overlap = measure_overlap(assemblies[name1], assemblies[name2])
            overlaps.append({
                "pair": (name1, name2),
                "overlap": overlap,
            })
        
        overlap_values = [o["overlap"] for o in overlaps]
        
        return {
            "pairwise_overlaps": overlaps,
            "mean_overlap": np.mean(overlap_values),
            "std_overlap": np.std(overlap_values),
            "min_overlap": np.min(overlap_values),
            "max_overlap": np.max(overlap_values),
            "n_distinct_pairs": sum(1 for o in overlap_values if o < 0.5),
            "n_total_pairs": len(overlap_values),
        }
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        k_active_range: List[int] = None,
        p_connect_range: List[float] = None,
        n_stimuli: int = 5,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """Test assembly distinctiveness across parameters."""
        self._start_timer()
        
        if n_neurons_range is None:
            n_neurons_range = [1000, 5000, 10000]
        if k_active_range is None:
            k_active_range = [10, 50, 100]
        if p_connect_range is None:
            p_connect_range = [0.01, 0.05, 0.1]
        
        self.log("Starting assembly distinctiveness experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  k_active: {k_active_range}")
        self.log(f"  p_connect: {p_connect_range}")
        self.log(f"  n_stimuli: {n_stimuli}")
        
        all_results = []
        
        for n in n_neurons_range:
            for k in k_active_range:
                if k >= n:
                    continue
                    
                for p in p_connect_range:
                    sparsity = k / n
                    
                    self.log(f"\n  Testing n={n}, k={k} (sparsity={sparsity:.2%}), p={p}")
                    
                    config = DistinctivenessConfig(
                        n_neurons=n,
                        k_active=k,
                        p_connect=p,
                        beta=0.1,
                        n_stimuli=n_stimuli
                    )
                    
                    # Test with separate brains
                    separate_overlaps = []
                    for trial in range(n_trials):
                        try:
                            assemblies = self.create_multiple_assemblies(config, trial)
                            overlap_stats = self.compute_pairwise_overlaps(assemblies)
                            separate_overlaps.append(overlap_stats["mean_overlap"])
                        except Exception as e:
                            self.log(f"    Separate brain trial {trial} failed: {e}")
                    
                    # Test with same brain
                    same_brain_overlaps = []
                    for trial in range(n_trials):
                        try:
                            assemblies = self.create_assemblies_same_brain(config, trial)
                            overlap_stats = self.compute_pairwise_overlaps(assemblies)
                            same_brain_overlaps.append(overlap_stats["mean_overlap"])
                        except Exception as e:
                            self.log(f"    Same brain trial {trial} failed: {e}")
                    
                    mean_separate = np.mean(separate_overlaps) if separate_overlaps else 1.0
                    mean_same = np.mean(same_brain_overlaps) if same_brain_overlaps else 1.0
                    
                    result = {
                        "n_neurons": n,
                        "k_active": k,
                        "sparsity": sparsity,
                        "p_connect": p,
                        "mean_overlap_separate_brains": mean_separate,
                        "mean_overlap_same_brain": mean_same,
                        "assemblies_distinct_separate": mean_separate < 0.5,
                        "assemblies_distinct_same": mean_same < 0.5,
                    }
                    all_results.append(result)
                    
                    self.log(f"    Separate brains: overlap={mean_separate:.3f}")
                    self.log(f"    Same brain: overlap={mean_same:.3f}")
        
        duration = self._stop_timer()
        
        # Find best parameters for distinctiveness
        best_separate = min(all_results, key=lambda x: x["mean_overlap_separate_brains"])
        best_same = min(all_results, key=lambda x: x["mean_overlap_same_brain"])
        
        # Calculate theoretical expected overlap
        # For random k-subsets of n neurons, expected overlap is k/n
        theoretical_overlaps = [r["sparsity"] for r in all_results]
        
        summary = {
            "total_configurations": len(all_results),
            "any_distinct_separate": any(r["assemblies_distinct_separate"] for r in all_results),
            "any_distinct_same": any(r["assemblies_distinct_same"] for r in all_results),
            "best_overlap_separate": best_separate["mean_overlap_separate_brains"],
            "best_params_separate": {
                "n": best_separate["n_neurons"],
                "k": best_separate["k_active"],
                "p": best_separate["p_connect"],
            },
            "best_overlap_same": best_same["mean_overlap_same_brain"],
            "mean_theoretical_overlap": np.mean(theoretical_overlaps),
        }
        
        self.log(f"\n{'='*60}")
        self.log("ASSEMBLY DISTINCTIVENESS SUMMARY:")
        self.log(f"  Any distinct (separate brains): {summary['any_distinct_separate']}")
        self.log(f"  Any distinct (same brain): {summary['any_distinct_same']}")
        self.log(f"  Best overlap (separate): {summary['best_overlap_separate']:.3f}")
        self.log(f"  Best overlap (same): {summary['best_overlap_same']:.3f}")
        self.log(f"  Mean theoretical (random): {summary['mean_theoretical_overlap']:.3f}")
        self.log(f"  Duration: {duration:.1f}s")
        
        # CRITICAL INTERPRETATION
        if not summary["any_distinct_separate"]:
            self.log("\n  ** CRITICAL: No configuration produced distinct assemblies!")
            self.log("      This suggests the random seed creates same 'hub' neurons")
            self.log("      Competition within same brain is needed for distinctiveness")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "k_active_range": k_active_range,
                "p_connect_range": p_connect_range,
                "n_stimuli": n_stimuli,
                "n_trials": n_trials,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )
        
        return result


def run_quick_test():
    """Run quick distinctiveness test."""
    print("="*60)
    print("Assembly Distinctiveness Test")
    print("="*60)
    
    exp = AssemblyDistinctivenessExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[10, 50],
        p_connect_range=[0.01, 0.1],
        n_stimuli=5,
        n_trials=3,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Assembly Distinctiveness Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = AssemblyDistinctivenessExperiment(verbose=True)
        result = exp.run(n_trials=10)
        exp.save_result(result, "_full")

