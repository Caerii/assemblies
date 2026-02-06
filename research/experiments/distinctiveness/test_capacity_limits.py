"""
Assembly Capacity Limits Experiment

Key Question: How many distinct assemblies can one brain area hold?

This experiment investigates:
1. What is the maximum number of distinct assemblies?
2. How does capacity scale with n and k?
3. What happens when capacity is exceeded? (Catastrophic interference?)
4. Is there a theoretical bound we can derive?

Theoretical Prediction:
- Random k-subsets: Expected overlap = k/n
- For assemblies to be "distinct" (overlap < 0.5): need k/n < 0.5
- Maximum capacity might be related to n/k (number of non-overlapping assemblies)
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
class CapacityConfig:
    """Configuration for capacity test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_projection_rounds: int = 15


class CapacityLimitsExperiment(ExperimentBase):
    """
    Test the maximum number of distinct assemblies an area can hold.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="capacity_limits",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "distinctiveness",
            verbose=verbose
        )
    
    def create_n_assemblies(
        self,
        config: CapacityConfig,
        n_assemblies: int,
        trial_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """Create n assemblies in the same brain."""
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        for i in range(n_assemblies):
            b.add_stimulus(f"STIM_{i}", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        assemblies = {}
        for i in range(n_assemblies):
            stim_name = f"STIM_{i}"
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        return assemblies
    
    def compute_distinctiveness_metrics(
        self,
        assemblies: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute metrics about how distinct the assemblies are."""
        names = list(assemblies.keys())
        n = len(names)
        
        if n < 2:
            return {"n_assemblies": n, "all_distinct": True, "mean_overlap": 0.0}
        
        overlaps = []
        for i, j in combinations(range(n), 2):
            overlap = measure_overlap(assemblies[names[i]], assemblies[names[j]])
            overlaps.append(overlap)
        
        mean_overlap = np.mean(overlaps)
        max_overlap = np.max(overlaps)
        min_overlap = np.min(overlaps)
        
        # Count how many pairs are "distinct" (overlap < threshold)
        distinct_threshold = 0.5
        n_distinct_pairs = sum(1 for o in overlaps if o < distinct_threshold)
        total_pairs = len(overlaps)
        
        # All assemblies distinct if all pairs are distinct
        all_distinct = n_distinct_pairs == total_pairs
        
        return {
            "n_assemblies": n,
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "min_overlap": min_overlap,
            "n_distinct_pairs": n_distinct_pairs,
            "total_pairs": total_pairs,
            "distinctiveness_rate": n_distinct_pairs / total_pairs,
            "all_distinct": all_distinct,
        }
    
    def find_capacity_limit(
        self,
        config: CapacityConfig,
        max_assemblies: int = 50,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Find the capacity limit by adding assemblies until distinctiveness breaks down.
        """
        results = []
        
        for n_assemblies in range(2, max_assemblies + 1, 2):  # Step by 2 for speed
            try:
                assemblies = self.create_n_assemblies(config, n_assemblies, trial_id)
                metrics = self.compute_distinctiveness_metrics(assemblies)
                results.append(metrics)
                
                # Stop if distinctiveness has completely broken down
                if metrics["mean_overlap"] > 0.8:
                    break
                    
            except Exception as e:
                self.log(f"    Failed at n={n_assemblies}: {e}")
                break
        
        # Find capacity (last n where distinctiveness > 0.5)
        capacity = 0
        for r in results:
            if r["distinctiveness_rate"] > 0.5:
                capacity = r["n_assemblies"]
        
        return {
            "capacity": capacity,
            "max_tested": results[-1]["n_assemblies"] if results else 0,
            "progression": results,
        }
    
    def test_capacity_vs_parameters(
        self,
        n_neurons_range: List[int],
        sparsity_range: List[float],
        p_connect: float = 0.1,
        beta: float = 0.1,
        trial_id: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Test how capacity scales with n and k.
        """
        results = []
        
        for n in n_neurons_range:
            for sparsity in sparsity_range:
                k = max(1, int(n * sparsity))
                
                config = CapacityConfig(
                    n_neurons=n,
                    k_active=k,
                    p_connect=p_connect,
                    beta=beta
                )
                
                # Theoretical max (if assemblies were non-overlapping)
                theoretical_max = n // k
                
                # Test up to 2x theoretical max
                max_to_test = min(100, theoretical_max * 2)
                
                capacity_result = self.find_capacity_limit(config, max_to_test, trial_id)
                
                results.append({
                    "n_neurons": n,
                    "k_active": k,
                    "sparsity": sparsity,
                    "theoretical_max": theoretical_max,
                    "actual_capacity": capacity_result["capacity"],
                    "capacity_ratio": capacity_result["capacity"] / theoretical_max if theoretical_max > 0 else 0,
                })
        
        return results
    
    def test_catastrophic_interference(
        self,
        config: CapacityConfig,
        n_assemblies: int = 20,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Do later assemblies interfere with earlier ones?
        
        Create assemblies sequentially and check if early assemblies
        are still retrievable after creating many more.
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        for i in range(n_assemblies):
            b.add_stimulus(f"STIM_{i}", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        # Create assemblies and save them
        original_assemblies = {}
        for i in range(n_assemblies):
            stim_name = f"STIM_{i}"
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            original_assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Now re-activate each stimulus and see if we get the same assembly
        retrieval_overlaps = []
        for i in range(n_assemblies):
            stim_name = f"STIM_{i}"
            
            # Re-project this stimulus
            for _ in range(5):  # Fewer rounds for retrieval
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            
            retrieved = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            overlap = measure_overlap(original_assemblies[stim_name], retrieved)
            
            retrieval_overlaps.append({
                "stimulus": stim_name,
                "creation_order": i,
                "retrieval_overlap": overlap,
                "successfully_retrieved": overlap > 0.8,
            })
        
        # Check if early assemblies are harder to retrieve
        early_retrievals = [r["retrieval_overlap"] for r in retrieval_overlaps[:n_assemblies//3]]
        late_retrievals = [r["retrieval_overlap"] for r in retrieval_overlaps[-n_assemblies//3:]]
        
        return {
            "test_type": "catastrophic_interference",
            "n_assemblies": n_assemblies,
            "mean_retrieval_overlap": np.mean([r["retrieval_overlap"] for r in retrieval_overlaps]),
            "early_mean_retrieval": np.mean(early_retrievals) if early_retrievals else 0,
            "late_mean_retrieval": np.mean(late_retrievals) if late_retrievals else 0,
            "early_assemblies_degraded": np.mean(early_retrievals) < np.mean(late_retrievals) - 0.1 if early_retrievals and late_retrievals else False,
            "retrieval_details": retrieval_overlaps,
        }
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        sparsity_range: List[float] = None,
        p_connect: float = 0.1,
        beta: float = 0.1,
        n_trials: int = 3,
        **kwargs
    ) -> ExperimentResult:
        """Run capacity limit tests."""
        self._start_timer()
        
        if n_neurons_range is None:
            n_neurons_range = [1000, 5000, 10000]
        if sparsity_range is None:
            sparsity_range = [0.01, 0.02, 0.05, 0.1]
        
        self.log(f"Starting capacity limits experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  sparsity: {sparsity_range}")
        
        all_results = {
            "capacity_vs_parameters": [],
            "catastrophic_interference": [],
        }
        
        # Test 1: Capacity vs Parameters
        self.log(f"\n--- Test 1: Capacity vs Parameters ---")
        for trial in range(n_trials):
            try:
                results = self.test_capacity_vs_parameters(
                    n_neurons_range, sparsity_range, p_connect, beta, trial
                )
                all_results["capacity_vs_parameters"].extend(results)
                
                for r in results:
                    self.log(f"  n={r['n_neurons']}, k={r['k_active']}: "
                           f"capacity={r['actual_capacity']}/{r['theoretical_max']} "
                           f"({r['capacity_ratio']:.1%})")
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        # Test 2: Catastrophic Interference
        self.log(f"\n--- Test 2: Catastrophic Interference ---")
        for trial in range(n_trials):
            config = CapacityConfig(
                n_neurons=10000,
                k_active=100,
                p_connect=p_connect,
                beta=beta
            )
            try:
                result = self.test_catastrophic_interference(config, n_assemblies=20, trial_id=trial)
                all_results["catastrophic_interference"].append(result)
                
                self.log(f"  Trial {trial}: mean retrieval={result['mean_retrieval_overlap']:.3f}, "
                       f"early degraded={result['early_assemblies_degraded']}")
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        duration = self._stop_timer()
        
        # Summary
        if all_results["capacity_vs_parameters"]:
            mean_capacity_ratio = np.mean([r["capacity_ratio"] for r in all_results["capacity_vs_parameters"]])
        else:
            mean_capacity_ratio = 0
        
        if all_results["catastrophic_interference"]:
            interference_rate = np.mean([r["early_assemblies_degraded"] for r in all_results["catastrophic_interference"]])
        else:
            interference_rate = 0
        
        summary = {
            "mean_capacity_ratio": mean_capacity_ratio,
            "catastrophic_interference_observed": interference_rate > 0.5,
            "mean_retrieval_accuracy": np.mean([r["mean_retrieval_overlap"] for r in all_results["catastrophic_interference"]]) if all_results["catastrophic_interference"] else 0,
        }
        
        self.log(f"\n{'='*60}")
        self.log(f"CAPACITY LIMITS SUMMARY:")
        self.log(f"  Mean capacity ratio (actual/theoretical): {summary['mean_capacity_ratio']:.1%}")
        self.log(f"  Catastrophic interference: {summary['catastrophic_interference_observed']}")
        self.log(f"  Mean retrieval accuracy: {summary['mean_retrieval_accuracy']:.3f}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "sparsity_range": sparsity_range,
                "p_connect": p_connect,
                "beta": beta,
                "n_trials": n_trials,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data=all_results,
            duration_seconds=duration,
        )
        
        return result


def run_quick_test():
    """Run quick capacity limits test."""
    print("="*60)
    print("Capacity Limits Test")
    print("="*60)
    
    exp = CapacityLimitsExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000],
        sparsity_range=[0.02, 0.05],
        n_trials=2,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capacity Limits Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = CapacityLimitsExperiment(verbose=True)
        result = exp.run(n_trials=5)
        exp.save_result(result, "_full")

