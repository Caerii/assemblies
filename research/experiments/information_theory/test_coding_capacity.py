"""
Coding Capacity Experiment

Scientific Questions:
1. How many distinct assemblies can be reliably stored and retrieved?
2. What is the information capacity in bits?
3. How does capacity scale with n and k?
4. What is the theoretical maximum vs achieved capacity?

Expected Results:
- Capacity should scale as C(n,k) = n choose k combinations
- Practical capacity limited by overlap/interference
- Sparse coding (small k/n) should be more efficient per neuron
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy.special import comb
import math

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
    measure_jaccard,
)

import brain as brain_module


@dataclass
class CodingConfig:
    """Configuration for coding capacity test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_projection_rounds: int = 10


class CodingCapacityExperiment(ExperimentBase):
    """
    Measure information coding capacity of assembly representations.
    
    Hypothesis: Assembly coding has capacity proportional to log2(C(n,k)),
    but practical capacity is limited by interference between assemblies.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="coding_capacity",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "information_theory",
            verbose=verbose
        )
    
    def theoretical_capacity(self, n: int, k: int) -> Dict[str, float]:
        """Calculate theoretical capacity bounds."""
        # Number of possible assemblies: C(n,k)
        n_combinations = comb(n, k, exact=False)
        
        # Information capacity in bits
        if n_combinations > 0:
            bits = math.log2(n_combinations)
        else:
            bits = 0
        
        # Bits per neuron
        bits_per_neuron = bits / n if n > 0 else 0
        
        # Bits per active neuron
        bits_per_active = bits / k if k > 0 else 0
        
        return {
            "n_combinations": n_combinations,
            "total_bits": bits,
            "bits_per_neuron": bits_per_neuron,
            "bits_per_active_neuron": bits_per_active,
        }
    
    def measure_distinguishability(
        self,
        assemblies: List[np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Measure how distinguishable a set of assemblies are.
        
        Args:
            assemblies: List of winner arrays
            threshold: Overlap threshold for considering assemblies distinct
        
        Returns:
            Distinguishability metrics
        """
        n_assemblies = len(assemblies)
        
        if n_assemblies < 2:
            return {
                "n_assemblies": n_assemblies,
                "n_distinguishable": n_assemblies,
                "mean_pairwise_overlap": 0.0,
            }
        
        # Compute pairwise overlaps
        overlaps = []
        for i in range(n_assemblies):
            for j in range(i + 1, n_assemblies):
                overlap = measure_overlap(assemblies[i], assemblies[j])
                overlaps.append(overlap)
        
        mean_overlap = np.mean(overlaps)
        max_overlap = np.max(overlaps)
        min_overlap = np.min(overlaps)
        
        # Count distinguishable pairs (overlap below threshold)
        n_distinguishable_pairs = sum(1 for o in overlaps if o < threshold)
        total_pairs = len(overlaps)
        
        return {
            "n_assemblies": n_assemblies,
            "mean_pairwise_overlap": mean_overlap,
            "max_pairwise_overlap": max_overlap,
            "min_pairwise_overlap": min_overlap,
            "distinguishable_pair_rate": n_distinguishable_pairs / total_pairs if total_pairs > 0 else 1.0,
            "overlap_distribution": overlaps,
        }
    
    def create_assemblies(
        self,
        config: CodingConfig,
        n_assemblies: int
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Create multiple distinct assemblies using different stimuli.
        
        Returns:
            Tuple of (list of assemblies, list of success flags)
        """
        assemblies = []
        successes = []
        
        for i in range(n_assemblies):
            b = brain_module.Brain(p=config.p_connect, seed=self.seed + i * 1000)
            
            # Each assembly gets a unique stimulus
            stim_name = f"STIM_{i}"
            b.add_stimulus(stim_name, config.k_active)
            b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
            
            # Project to establish assembly
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            
            assembly = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            assemblies.append(assembly)
            successes.append(len(assembly) == config.k_active)
        
        return assemblies, successes
    
    def run(
        self,
        n_neurons_range: List[int] = None,
        k_active_range: List[int] = None,
        n_assemblies_to_test: List[int] = None,
        p_connect: float = 0.1,
        beta: float = 0.1,
        **kwargs
    ) -> ExperimentResult:
        """
        Measure coding capacity across configurations.
        
        Args:
            n_neurons_range: Network sizes to test
            k_active_range: Assembly sizes to test
            n_assemblies_to_test: Number of assemblies to create for each config
            p_connect: Connection probability
            beta: Plasticity parameter
        """
        self._start_timer()
        
        if n_neurons_range is None:
            n_neurons_range = [1000, 5000, 10000]
        if k_active_range is None:
            k_active_range = [10, 50, 100]
        if n_assemblies_to_test is None:
            n_assemblies_to_test = [5, 10, 20, 50]
        
        self.log(f"Starting coding capacity experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  k_active: {k_active_range}")
        self.log(f"  n_assemblies_to_test: {n_assemblies_to_test}")
        
        all_results = []
        
        for n in n_neurons_range:
            for k in k_active_range:
                if k >= n:
                    continue
                
                # Calculate theoretical capacity
                theory = self.theoretical_capacity(n, k)
                
                self.log(f"\n  n={n}, k={k}")
                self.log(f"    Theoretical capacity: {theory['total_bits']:.1f} bits")
                
                config = CodingConfig(
                    n_neurons=n,
                    k_active=k,
                    p_connect=p_connect,
                    beta=beta
                )
                
                capacity_results = []
                
                for n_assemblies in n_assemblies_to_test:
                    try:
                        assemblies, successes = self.create_assemblies(config, n_assemblies)
                        distinguishability = self.measure_distinguishability(assemblies)
                        
                        # Practical bits = log2(n_distinguishable_assemblies)
                        n_distinct = int(n_assemblies * distinguishability["distinguishable_pair_rate"])
                        practical_bits = math.log2(max(1, n_distinct))
                        
                        capacity_results.append({
                            "n_assemblies": n_assemblies,
                            "success_rate": sum(successes) / len(successes),
                            "mean_overlap": distinguishability["mean_pairwise_overlap"],
                            "distinguishable_rate": distinguishability["distinguishable_pair_rate"],
                            "practical_bits": practical_bits,
                        })
                        
                        self.log(f"    {n_assemblies} assemblies: overlap={distinguishability['mean_pairwise_overlap']:.3f}, "
                               f"distinct={distinguishability['distinguishable_pair_rate']:.1%}")
                        
                    except Exception as e:
                        self.log(f"    {n_assemblies} assemblies: Failed - {e}")
                
                all_results.append({
                    "n_neurons": n,
                    "k_active": k,
                    "sparsity": k / n,
                    "theoretical_capacity": theory,
                    "capacity_results": capacity_results,
                })
        
        duration = self._stop_timer()
        
        # Summary analysis
        summary = self._compute_summary(all_results)
        
        self.log(f"\n{'='*60}")
        self.log(f"CODING CAPACITY SUMMARY:")
        self.log(f"  Best efficiency: {summary.get('best_efficiency', 'N/A'):.3f} bits/neuron")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons_range": n_neurons_range,
                "k_active_range": k_active_range,
                "n_assemblies_to_test": n_assemblies_to_test,
                "p_connect": p_connect,
                "beta": beta,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )
        
        return result
    
    def _compute_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics."""
        efficiencies = []
        
        for r in results:
            theory = r["theoretical_capacity"]
            efficiencies.append(theory["bits_per_neuron"])
        
        return {
            "total_configurations": len(results),
            "best_efficiency": max(efficiencies) if efficiencies else 0,
            "mean_efficiency": np.mean(efficiencies) if efficiencies else 0,
        }


def run_quick_test():
    """Run quick coding capacity test."""
    print("="*60)
    print("QUICK TEST: Coding Capacity")
    print("="*60)
    
    exp = CodingCapacityExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[20, 50],
        n_assemblies_to_test=[5, 10, 20],
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coding Capacity Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = CodingCapacityExperiment(verbose=True)
        result = exp.run()
        exp.save_result(result, "_full")

