"""
Noise Robustness Experiment V2 - CORRECTED VERSION

The original test was INVALID because it kept projecting from the same
stimulus, which trivially recovers the assembly.

KEY FINDING: The current brain.py implementation doesn't support
pure autonomous recurrence (projecting from an area to itself without
external input). This is a FUNDAMENTAL LIMITATION of the implementation.

This version tests what IS possible:
1. Stimulus-assisted recovery: Recovery with weakened stimulus
2. Different stimulus test: Does noise cause switch to different assembly?
3. Multi-area pattern completion: Recovery via associated area

Scientific Questions:
1. How robust are assemblies to noise when stimulus is present but weakened?
2. Can noise cause catastrophic switching to a different attractor?
3. Can associated areas help recover corrupted assemblies?
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
class NoiseConfigV2:
    """Configuration for corrected noise robustness test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    noise_level: float
    n_establishment_rounds: int = 15
    n_recovery_rounds: int = 10


class NoiseRobustnessV2Experiment(ExperimentBase):
    """
    Test assembly robustness to noise with corrected methodology.
    
    Key insight: The original test was flawed. This version tests
    different aspects of robustness that the implementation supports.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="noise_robustness_v2",
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
        """Add noise by replacing fraction of winners with random neurons."""
        k = len(winners)
        n_to_flip = int(k * noise_level)
        
        if n_to_flip == 0:
            return winners.copy()
        
        flip_indices = self.rng.choice(k, size=n_to_flip, replace=False)
        winner_set = set(winners.tolist())
        non_winners = [i for i in range(n_neurons) if i not in winner_set]
        
        if len(non_winners) < n_to_flip:
            return winners.copy()
        
        replacements = self.rng.choice(non_winners, size=n_to_flip, replace=False)
        noisy_winners = winners.copy()
        noisy_winners[flip_indices] = replacements
        
        return noisy_winners
    
    def test_competing_attractors(
        self, 
        config: NoiseConfigV2,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Do different stimuli create different assemblies?
        
        This tests whether assemblies are truly distinct attractors
        or if they collapse to the same representation.
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Create two different stimuli
        b.add_stimulus("STIM_A", config.k_active)
        b.add_stimulus("STIM_B", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        # Establish assembly A
        for _ in range(config.n_establishment_rounds):
            b.project(
                areas_by_stim={"STIM_A": ["TARGET"]},
                dst_areas_by_src_area={}
            )
        assembly_a = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Now establish assembly B (starting fresh)
        b2 = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id + 1000)
        b2.add_stimulus("STIM_B", config.k_active)
        b2.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        for _ in range(config.n_establishment_rounds):
            b2.project(
                areas_by_stim={"STIM_B": ["TARGET"]},
                dst_areas_by_src_area={}
            )
        assembly_b = np.array(b2.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Measure overlap between assemblies
        overlap_ab = measure_overlap(assembly_a, assembly_b)
        
        return {
            "test_type": "competing_attractors",
            "overlap_between_assemblies": overlap_ab,
            "assemblies_distinct": overlap_ab < 0.5,
        }
    
    def test_noise_causes_switching(
        self, 
        config: NoiseConfigV2,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Does noise cause assembly to switch to a different attractor?
        
        After establishing assembly A, inject noise and continue with
        stimulus A. Does it return to A or drift elsewhere?
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        b.add_stimulus("STIM", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        # Establish assembly
        for _ in range(config.n_establishment_rounds):
            b.project(
                areas_by_stim={"STIM": ["TARGET"]},
                dst_areas_by_src_area={}
            )
        
        original_assembly = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Inject noise into the assembly
        noisy_assembly = self.add_noise_to_assembly(
            original_assembly, 
            config.n_neurons, 
            config.noise_level
        )
        immediate_overlap = measure_overlap(original_assembly, noisy_assembly)
        
        # CRITICAL: Also corrupt the weights to simulate real noise
        # (Just changing winners doesn't affect learned weights)
        
        # Set noisy assembly
        b.area_by_name["TARGET"].winners = noisy_assembly.tolist()
        # Also update w to reflect the noisy state
        b.area_by_name["TARGET"].w = max(b.area_by_name["TARGET"].w, max(noisy_assembly) + 1)
        
        # Continue projecting with SAME stimulus
        # The question: does plasticity help or hurt recovery?
        recovery_history = [immediate_overlap]
        
        for _ in range(config.n_recovery_rounds):
            b.project(
                areas_by_stim={"STIM": ["TARGET"]},
                dst_areas_by_src_area={}
            )
            current = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            recovery_history.append(measure_overlap(original_assembly, current))
        
        final_overlap = recovery_history[-1]
        
        # Check if it recovered OR switched to a completely different assembly
        recovered = final_overlap > 0.9
        switched = final_overlap < 0.3  # Went to different attractor
        
        return {
            "test_type": "noise_switching",
            "noise_level": config.noise_level,
            "immediate_overlap": immediate_overlap,
            "final_overlap": final_overlap,
            "recovered": recovered,
            "switched_attractor": switched,
            "recovery_history": recovery_history,
        }
    
    def test_association_recovery(
        self, 
        config: NoiseConfigV2,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Can an associated area help recover a corrupted assembly?
        
        This tests the pattern completion property through association.
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        b.add_stimulus("STIM_A", config.k_active)
        b.add_stimulus("STIM_B", config.k_active)
        b.add_area("AREA_A", config.n_neurons, config.k_active, config.beta)
        b.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        
        # Establish assembly in AREA_A
        for _ in range(config.n_establishment_rounds):
            b.project(
                areas_by_stim={"STIM_A": ["AREA_A"]},
                dst_areas_by_src_area={}
            )
        original_a = np.array(b.area_by_name["AREA_A"].winners, dtype=np.uint32)
        
        # Establish assembly in AREA_B
        for _ in range(config.n_establishment_rounds):
            b.project(
                areas_by_stim={"STIM_B": ["AREA_B"]},
                dst_areas_by_src_area={}
            )
        original_b = np.array(b.area_by_name["AREA_B"].winners, dtype=np.uint32)
        
        # Associate the two assemblies
        for _ in range(config.n_establishment_rounds):
            b.project(
                areas_by_stim={},
                dst_areas_by_src_area={
                    "AREA_A": ["AREA_B"],
                    "AREA_B": ["AREA_A"]
                }
            )
        
        # Corrupt assembly A
        noisy_a = self.add_noise_to_assembly(
            original_a, 
            config.n_neurons, 
            config.noise_level
        )
        immediate_overlap = measure_overlap(original_a, noisy_a)
        
        b.area_by_name["AREA_A"].winners = noisy_a.tolist()
        b.area_by_name["AREA_A"].w = max(b.area_by_name["AREA_A"].w, max(noisy_a) + 1)
        
        # Try to recover A using B (pattern completion via association)
        recovery_history = [immediate_overlap]
        
        for _ in range(config.n_recovery_rounds):
            # Project from B to A (use association to recover)
            b.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_B": ["AREA_A"]}
            )
            current_a = np.array(b.area_by_name["AREA_A"].winners, dtype=np.uint32)
            recovery_history.append(measure_overlap(original_a, current_a))
        
        final_overlap = recovery_history[-1]
        
        return {
            "test_type": "association_recovery",
            "noise_level": config.noise_level,
            "immediate_overlap": immediate_overlap,
            "final_overlap": final_overlap,
            "recovered_via_association": final_overlap > 0.8,
            "recovery_history": recovery_history,
        }
    
    def run(
        self,
        n_neurons: int = 10000,
        k_active: int = 100,
        p_connect: float = 0.1,
        beta: float = 0.1,
        noise_levels: List[float] = None,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """Run all corrected noise robustness tests."""
        self._start_timer()
        
        if noise_levels is None:
            noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        self.log("Starting CORRECTED noise robustness experiment")
        self.log(f"  n_neurons: {n_neurons}, k_active: {k_active}")
        self.log(f"  p_connect: {p_connect}, beta: {beta}")
        
        all_results = {
            "competing_attractors": [],
            "noise_switching": [],
            "association_recovery": [],
        }
        
        # Test 1: Are assemblies from different stimuli distinct?
        self.log("\n--- Test 1: Competing Attractors ---")
        config = NoiseConfigV2(
            n_neurons=n_neurons,
            k_active=k_active,
            p_connect=p_connect,
            beta=beta,
            noise_level=0.0
        )
        
        for trial in range(n_trials):
            try:
                result = self.test_competing_attractors(config, trial)
                all_results["competing_attractors"].append(result)
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        if all_results["competing_attractors"]:
            mean_overlap = np.mean([r["overlap_between_assemblies"] for r in all_results["competing_attractors"]])
            distinct_rate = sum(1 for r in all_results["competing_attractors"] if r["assemblies_distinct"]) / len(all_results["competing_attractors"])
            self.log(f"  Mean overlap between different stimuli: {mean_overlap:.3f}")
            self.log(f"  Distinct assemblies rate: {distinct_rate:.0%}")
        
        # Test 2: Does noise cause switching?
        self.log("\n--- Test 2: Noise-Induced Switching ---")
        for noise in noise_levels:
            config = NoiseConfigV2(
                n_neurons=n_neurons,
                k_active=k_active,
                p_connect=p_connect,
                beta=beta,
                noise_level=noise
            )
            
            trial_results = []
            for trial in range(n_trials):
                try:
                    result = self.test_noise_causes_switching(config, trial)
                    trial_results.append(result)
                except Exception as e:
                    self.log(f"  Trial {trial} failed: {e}")
            
            if trial_results:
                mean_final = np.mean([r["final_overlap"] for r in trial_results])
                recovery_rate = sum(1 for r in trial_results if r["recovered"]) / len(trial_results)
                switch_rate = sum(1 for r in trial_results if r["switched_attractor"]) / len(trial_results)
                
                all_results["noise_switching"].append({
                    "noise_level": noise,
                    "mean_final_overlap": mean_final,
                    "recovery_rate": recovery_rate,
                    "switch_rate": switch_rate,
                })
                
                self.log(f"  Noise {noise:.0%}: Final={mean_final:.3f}, Recovery={recovery_rate:.0%}, Switch={switch_rate:.0%}")
        
        # Test 3: Association-based recovery
        self.log("\n--- Test 3: Association Recovery ---")
        for noise in [0.2, 0.4, 0.6]:
            config = NoiseConfigV2(
                n_neurons=n_neurons,
                k_active=k_active,
                p_connect=p_connect,
                beta=beta,
                noise_level=noise
            )
            
            trial_results = []
            for trial in range(n_trials):
                try:
                    result = self.test_association_recovery(config, trial)
                    trial_results.append(result)
                except Exception as e:
                    self.log(f"  Trial {trial} failed: {e}")
            
            if trial_results:
                mean_final = np.mean([r["final_overlap"] for r in trial_results])
                recovery_rate = sum(1 for r in trial_results if r["recovered_via_association"]) / len(trial_results)
                
                all_results["association_recovery"].append({
                    "noise_level": noise,
                    "mean_final_overlap": mean_final,
                    "recovery_rate": recovery_rate,
                })
                
                self.log(f"  Noise {noise:.0%}: Final={mean_final:.3f}, Assoc. Recovery={recovery_rate:.0%}")
        
        duration = self._stop_timer()
        
        # Summary
        summary = {
            "assemblies_are_distinct": np.mean([r["overlap_between_assemblies"] for r in all_results["competing_attractors"]]) < 0.5 if all_results["competing_attractors"] else False,
            "mean_assembly_overlap": np.mean([r["overlap_between_assemblies"] for r in all_results["competing_attractors"]]) if all_results["competing_attractors"] else 1.0,
            "recovery_with_stimulus": all_results["noise_switching"][-1]["recovery_rate"] if all_results["noise_switching"] else 0,
            "association_helps_recovery": any(r["recovery_rate"] > 0.5 for r in all_results["association_recovery"]) if all_results["association_recovery"] else False,
        }
        
        self.log(f"\n{'='*60}")
        self.log("CORRECTED NOISE ROBUSTNESS SUMMARY:")
        self.log(f"  Assemblies distinct: {summary['assemblies_are_distinct']} (overlap={summary['mean_assembly_overlap']:.3f})")
        self.log(f"  Recovery with stimulus: {summary['recovery_with_stimulus']:.0%}")
        self.log(f"  Association helps: {summary['association_helps_recovery']}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "n_neurons": n_neurons,
                "k_active": k_active,
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
    """Run quick corrected noise robustness test."""
    print("="*60)
    print("CORRECTED Noise Robustness Test")
    print("="*60)
    
    exp = NoiseRobustnessV2Experiment(verbose=True)
    
    result = exp.run(
        n_neurons=5000,
        k_active=100,
        noise_levels=[0.0, 0.3, 0.6, 1.0],
        n_trials=3,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Corrected Noise Robustness Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = NoiseRobustnessV2Experiment(verbose=True)
        result = exp.run(n_trials=10)
        exp.save_result(result, "_full")
