"""
Competition Mechanisms for Assembly Distinctiveness

Key Discovery: Assemblies are only distinct when stimuli compete in the same brain.

This experiment investigates:
1. WHY does competition create distinctiveness?
2. What is the mechanism of competition?
3. Can we quantify the "competition strength"?
4. How does order of presentation affect distinctiveness?

Hypotheses:
H1: Plasticity creates "ownership" - first stimulus strengthens its neurons
H2: Winner-take-all creates exclusion - once neurons are claimed, they can't be reused
H3: Support growth (w) tracks "used" neurons, preventing reuse
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from research.experiments.base import (
    ExperimentBase, 
    ExperimentResult, 
    measure_overlap,
)

import brain as brain_module


@dataclass
class CompetitionConfig:
    """Configuration for competition test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_projection_rounds: int = 15


class CompetitionMechanismsExperiment(ExperimentBase):
    """
    Investigate the mechanisms that create distinct assemblies.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="competition_mechanisms",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "distinctiveness",
            verbose=verbose
        )
    
    def test_sequential_vs_interleaved(
        self,
        config: CompetitionConfig,
        n_stimuli: int = 3,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Does order of presentation matter?
        
        Sequential: AAAA BBBB CCCC
        Interleaved: ABC ABC ABC ABC
        
        Hypothesis: Interleaved should create MORE distinct assemblies
        because competition happens at each step.
        """
        # Sequential presentation
        b_seq = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        for i in range(n_stimuli):
            b_seq.add_stimulus(f"STIM_{i}", config.k_active)
        b_seq.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        seq_assemblies = {}
        for i in range(n_stimuli):
            stim_name = f"STIM_{i}"
            for _ in range(config.n_projection_rounds):
                b_seq.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            seq_assemblies[stim_name] = np.array(b_seq.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Interleaved presentation
        b_int = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        for i in range(n_stimuli):
            b_int.add_stimulus(f"STIM_{i}", config.k_active)
        b_int.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        int_assemblies = {}
        for round_idx in range(config.n_projection_rounds):
            for i in range(n_stimuli):
                stim_name = f"STIM_{i}"
                b_int.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
                if round_idx == config.n_projection_rounds - 1:
                    int_assemblies[stim_name] = np.array(b_int.area_by_name["TARGET"].winners, dtype=np.uint32)
        
        # Compute overlaps
        seq_overlaps = []
        int_overlaps = []
        
        for i in range(n_stimuli):
            for j in range(i + 1, n_stimuli):
                seq_overlaps.append(measure_overlap(
                    seq_assemblies[f"STIM_{i}"], 
                    seq_assemblies[f"STIM_{j}"]
                ))
                int_overlaps.append(measure_overlap(
                    int_assemblies[f"STIM_{i}"], 
                    int_assemblies[f"STIM_{j}"]
                ))
        
        return {
            "test_type": "sequential_vs_interleaved",
            "sequential_mean_overlap": np.mean(seq_overlaps),
            "interleaved_mean_overlap": np.mean(int_overlaps),
            "interleaved_better": np.mean(int_overlaps) < np.mean(seq_overlaps),
            "sequential_overlaps": seq_overlaps,
            "interleaved_overlaps": int_overlaps,
        }
    
    def test_support_growth_effect(
        self,
        config: CompetitionConfig,
        n_stimuli: int = 5,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: How does support (w) grow and affect distinctiveness?
        
        The support w tracks how many neurons have "ever fired".
        Hypothesis: As w grows, new stimuli are forced to use new neurons.
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        for i in range(n_stimuli):
            b.add_stimulus(f"STIM_{i}", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        assemblies = {}
        support_history = []
        
        for i in range(n_stimuli):
            stim_name = f"STIM_{i}"
            
            # Record support before this stimulus
            support_before = b.area_by_name["TARGET"].w
            
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            
            # Record support after
            support_after = b.area_by_name["TARGET"].w
            
            assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            support_history.append({
                "stimulus": stim_name,
                "support_before": support_before,
                "support_after": support_after,
                "support_growth": support_after - support_before,
            })
        
        # Compute overlaps
        overlaps = []
        for i in range(n_stimuli):
            for j in range(i + 1, n_stimuli):
                overlaps.append({
                    "pair": (f"STIM_{i}", f"STIM_{j}"),
                    "overlap": measure_overlap(assemblies[f"STIM_{i}"], assemblies[f"STIM_{j}"]),
                    "order_distance": j - i,  # How far apart in presentation order
                })
        
        # Does overlap decrease with order distance?
        order_distances = [o["order_distance"] for o in overlaps]
        overlap_values = [o["overlap"] for o in overlaps]
        
        # Simple correlation
        if len(set(order_distances)) > 1:
            correlation = np.corrcoef(order_distances, overlap_values)[0, 1]
        else:
            correlation = 0.0
        
        return {
            "test_type": "support_growth",
            "support_history": support_history,
            "final_support": support_history[-1]["support_after"],
            "total_support_growth": sum(s["support_growth"] for s in support_history),
            "mean_overlap": np.mean(overlap_values),
            "overlap_vs_order_correlation": correlation,
            "later_stimuli_more_distinct": correlation < -0.3,
        }
    
    def test_plasticity_effect(
        self,
        config: CompetitionConfig,
        n_stimuli: int = 3,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Does plasticity (beta) affect distinctiveness?
        
        Hypothesis: Higher beta creates stronger "ownership" of neurons,
        leading to more distinct assemblies.
        """
        beta_values = [0.0, 0.05, 0.1, 0.2, 0.5]
        results_by_beta = {}
        
        for beta in beta_values:
            b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
            for i in range(n_stimuli):
                b.add_stimulus(f"STIM_{i}", config.k_active)
            b.add_area("TARGET", config.n_neurons, config.k_active, beta)
            
            assemblies = {}
            for i in range(n_stimuli):
                stim_name = f"STIM_{i}"
                for _ in range(config.n_projection_rounds):
                    b.project(
                        areas_by_stim={stim_name: ["TARGET"]},
                        dst_areas_by_src_area={}
                    )
                assemblies[stim_name] = np.array(b.area_by_name["TARGET"].winners, dtype=np.uint32)
            
            # Compute mean overlap
            overlaps = []
            for i in range(n_stimuli):
                for j in range(i + 1, n_stimuli):
                    overlaps.append(measure_overlap(
                        assemblies[f"STIM_{i}"], 
                        assemblies[f"STIM_{j}"]
                    ))
            
            results_by_beta[beta] = {
                "mean_overlap": np.mean(overlaps),
                "std_overlap": np.std(overlaps),
            }
        
        # Find optimal beta
        best_beta = min(results_by_beta.keys(), key=lambda b: results_by_beta[b]["mean_overlap"])
        
        return {
            "test_type": "plasticity_effect",
            "results_by_beta": results_by_beta,
            "best_beta": best_beta,
            "best_overlap": results_by_beta[best_beta]["mean_overlap"],
            "plasticity_helps": results_by_beta[0.0]["mean_overlap"] > results_by_beta[best_beta]["mean_overlap"],
        }
    
    def test_neuron_reuse_tracking(
        self,
        config: CompetitionConfig,
        n_stimuli: int = 5,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Test: Which neurons get reused across assemblies?
        
        Track which specific neurons appear in multiple assemblies.
        """
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        for i in range(n_stimuli):
            b.add_stimulus(f"STIM_{i}", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta)
        
        assemblies = {}
        for i in range(n_stimuli):
            stim_name = f"STIM_{i}"
            for _ in range(config.n_projection_rounds):
                b.project(
                    areas_by_stim={stim_name: ["TARGET"]},
                    dst_areas_by_src_area={}
                )
            assemblies[stim_name] = set(b.area_by_name["TARGET"].winners.tolist())
        
        # Count how many assemblies each neuron appears in
        all_neurons = set()
        for assembly in assemblies.values():
            all_neurons.update(assembly)
        
        neuron_counts = {}
        for neuron in all_neurons:
            count = sum(1 for assembly in assemblies.values() if neuron in assembly)
            neuron_counts[neuron] = count
        
        # Distribution of reuse
        count_distribution = {}
        for count in range(1, n_stimuli + 1):
            count_distribution[count] = sum(1 for c in neuron_counts.values() if c == count)
        
        # "Hub" neurons that appear in many assemblies
        hub_threshold = max(2, n_stimuli // 2)
        hub_neurons = [n for n, c in neuron_counts.items() if c >= hub_threshold]
        
        return {
            "test_type": "neuron_reuse",
            "total_unique_neurons": len(all_neurons),
            "neurons_in_single_assembly": count_distribution.get(1, 0),
            "neurons_in_multiple": sum(count_distribution.get(i, 0) for i in range(2, n_stimuli + 1)),
            "hub_neurons_count": len(hub_neurons),
            "count_distribution": count_distribution,
            "reuse_rate": len(hub_neurons) / len(all_neurons) if all_neurons else 0,
        }
    
    def run(
        self,
        n_neurons: int = 10000,
        k_active: int = 100,
        p_connect: float = 0.1,
        beta: float = 0.1,
        n_trials: int = 5,
        **kwargs
    ) -> ExperimentResult:
        """Run all competition mechanism tests."""
        self._start_timer()
        
        self.log("Starting competition mechanisms experiment")
        self.log(f"  n_neurons: {n_neurons}, k_active: {k_active}")
        
        config = CompetitionConfig(
            n_neurons=n_neurons,
            k_active=k_active,
            p_connect=p_connect,
            beta=beta
        )
        
        all_results = {
            "sequential_vs_interleaved": [],
            "support_growth": [],
            "plasticity_effect": [],
            "neuron_reuse": [],
        }
        
        # Test 1: Sequential vs Interleaved
        self.log("\n--- Test 1: Sequential vs Interleaved ---")
        for trial in range(n_trials):
            try:
                result = self.test_sequential_vs_interleaved(config, trial_id=trial)
                all_results["sequential_vs_interleaved"].append(result)
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        if all_results["sequential_vs_interleaved"]:
            seq_mean = np.mean([r["sequential_mean_overlap"] for r in all_results["sequential_vs_interleaved"]])
            int_mean = np.mean([r["interleaved_mean_overlap"] for r in all_results["sequential_vs_interleaved"]])
            self.log(f"  Sequential overlap: {seq_mean:.3f}")
            self.log(f"  Interleaved overlap: {int_mean:.3f}")
        
        # Test 2: Support Growth
        self.log("\n--- Test 2: Support Growth Effect ---")
        for trial in range(n_trials):
            try:
                result = self.test_support_growth_effect(config, trial_id=trial)
                all_results["support_growth"].append(result)
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        if all_results["support_growth"]:
            mean_overlap = np.mean([r["mean_overlap"] for r in all_results["support_growth"]])
            mean_corr = np.mean([r["overlap_vs_order_correlation"] for r in all_results["support_growth"]])
            self.log(f"  Mean overlap: {mean_overlap:.3f}")
            self.log(f"  Order-overlap correlation: {mean_corr:.3f}")
        
        # Test 3: Plasticity Effect
        self.log("\n--- Test 3: Plasticity Effect ---")
        for trial in range(n_trials):
            try:
                result = self.test_plasticity_effect(config, trial_id=trial)
                all_results["plasticity_effect"].append(result)
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        if all_results["plasticity_effect"]:
            best_betas = [r["best_beta"] for r in all_results["plasticity_effect"]]
            self.log(f"  Best beta values: {best_betas}")
        
        # Test 4: Neuron Reuse
        self.log("\n--- Test 4: Neuron Reuse Tracking ---")
        for trial in range(n_trials):
            try:
                result = self.test_neuron_reuse_tracking(config, trial_id=trial)
                all_results["neuron_reuse"].append(result)
            except Exception as e:
                self.log(f"  Trial {trial} failed: {e}")
        
        if all_results["neuron_reuse"]:
            mean_reuse = np.mean([r["reuse_rate"] for r in all_results["neuron_reuse"]])
            self.log(f"  Mean neuron reuse rate: {mean_reuse:.1%}")
        
        duration = self._stop_timer()
        
        # Summary
        summary = {
            "interleaved_creates_more_distinct": np.mean([r["interleaved_better"] for r in all_results["sequential_vs_interleaved"]]) > 0.5 if all_results["sequential_vs_interleaved"] else False,
            "later_stimuli_more_distinct": np.mean([r["later_stimuli_more_distinct"] for r in all_results["support_growth"]]) > 0.5 if all_results["support_growth"] else False,
            "plasticity_helps_distinctiveness": np.mean([r["plasticity_helps"] for r in all_results["plasticity_effect"]]) > 0.5 if all_results["plasticity_effect"] else False,
            "mean_neuron_reuse_rate": np.mean([r["reuse_rate"] for r in all_results["neuron_reuse"]]) if all_results["neuron_reuse"] else 0,
        }
        
        self.log(f"\n{'='*60}")
        self.log("COMPETITION MECHANISMS SUMMARY:")
        self.log(f"  Interleaved better: {summary['interleaved_creates_more_distinct']}")
        self.log(f"  Later stimuli more distinct: {summary['later_stimuli_more_distinct']}")
        self.log(f"  Plasticity helps: {summary['plasticity_helps_distinctiveness']}")
        self.log(f"  Neuron reuse rate: {summary['mean_neuron_reuse_rate']:.1%}")
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
    """Run quick competition mechanisms test."""
    print("="*60)
    print("Competition Mechanisms Test")
    print("="*60)
    
    exp = CompetitionMechanismsExperiment(verbose=True)
    
    result = exp.run(
        n_neurons=5000,
        k_active=50,
        n_trials=3,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Competition Mechanisms Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = CompetitionMechanismsExperiment(verbose=True)
        result = exp.run(n_trials=10)
        exp.save_result(result, "_full")

