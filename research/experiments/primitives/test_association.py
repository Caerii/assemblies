"""
Association Primitive Validation

Scientific Questions:
1. Does association reliably increase overlap between assemblies?
2. How many rounds are needed for significant overlap increase?
3. Is the association symmetric (A->B same as B->A)?
4. Does association preserve individual assembly identity?

Expected Results:
- Overlap should increase monotonically with association rounds
- Assemblies should remain distinguishable after association
- Association should be approximately symmetric
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
    measure_jaccard,
)

import brain as brain_module


@dataclass
class AssociationConfig:
    """Configuration for association test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_association_rounds: int = 20
    n_projection_rounds: int = 10  # To establish assemblies first


class AssociationBindingExperiment(ExperimentBase):
    """
    Test: Does association increase overlap between assemblies?
    
    Hypothesis: Repeatedly co-activating two assemblies increases their
    overlap through Hebbian plasticity.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="association_binding",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
            verbose=verbose
        )
    
    def _establish_assembly(
        self, 
        brain: brain_module.Brain, 
        stim_name: str, 
        area_name: str,
        n_rounds: int
    ) -> np.ndarray:
        """Project stimulus into area to establish stable assembly."""
        for _ in range(n_rounds):
            brain.project(
                areas_by_stim={stim_name: [area_name]},
                dst_areas_by_src_area={}
            )
        return np.array(brain.area_by_name[area_name].winners, dtype=np.uint32)
    
    def run_single_trial(
        self, 
        config: AssociationConfig,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single association trial.
        
        Tests association by:
        1. Creating two assemblies in separate areas (A and B)
        2. Projecting both to a shared area C
        3. Measuring if repeated co-projection increases overlap
        
        This tests the core association mechanism: when two assemblies
        repeatedly project to the same area, their representations should
        become more similar (increased overlap).
        """
        self.log(f"  Trial {trial_id}: n={config.n_neurons}, k={config.k_active}")
        
        # Create brain with two source areas and one target area
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Add two stimuli for the source areas
        b.add_stimulus("STIM_A", config.k_active)
        b.add_stimulus("STIM_B", config.k_active)
        
        # Add source areas A and B
        b.add_area("AREA_A", config.n_neurons, config.k_active, config.beta)
        b.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        
        # Add target area C where association happens
        b.add_area("AREA_C", config.n_neurons, config.k_active, config.beta)
        
        # Phase 1: Establish assembly in AREA_A
        self.log(f"    Establishing assembly A...")
        for _ in range(config.n_projection_rounds):
            b.project(areas_by_stim={"STIM_A": ["AREA_A"]}, dst_areas_by_src_area={})
        assembly_a = np.array(b.area_by_name["AREA_A"].winners, dtype=np.uint32)
        
        # Phase 2: Establish assembly in AREA_B
        self.log(f"    Establishing assembly B...")
        for _ in range(config.n_projection_rounds):
            b.project(areas_by_stim={"STIM_B": ["AREA_B"]}, dst_areas_by_src_area={})
        assembly_b = np.array(b.area_by_name["AREA_B"].winners, dtype=np.uint32)
        
        # Phase 3: Project A alone to C, record assembly
        self.log(f"    Projecting A -> C...")
        for _ in range(config.n_projection_rounds):
            b.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_A": ["AREA_C"]}
            )
        assembly_c_from_a = np.array(b.area_by_name["AREA_C"].winners, dtype=np.uint32)
        
        # Phase 4: Project B alone to C (fresh brain for fair comparison)
        b2 = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id + 1000)
        b2.add_stimulus("STIM_B", config.k_active)
        b2.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        b2.add_area("AREA_C", config.n_neurons, config.k_active, config.beta)
        
        for _ in range(config.n_projection_rounds):
            b2.project(areas_by_stim={"STIM_B": ["AREA_B"]}, dst_areas_by_src_area={})
        
        for _ in range(config.n_projection_rounds):
            b2.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_B": ["AREA_C"]}
            )
        assembly_c_from_b = np.array(b2.area_by_name["AREA_C"].winners, dtype=np.uint32)
        
        # Measure initial overlap between C assemblies from A vs B
        initial_overlap = measure_overlap(assembly_c_from_a, assembly_c_from_b)
        self.log(f"    Initial overlap (A->C vs B->C): {initial_overlap:.3f}")
        
        # Phase 5: Association - repeatedly co-project A and B to C
        # This should strengthen connections and increase overlap
        self.log(f"    Running {config.n_association_rounds} association rounds...")
        
        overlap_history = [initial_overlap]
        
        for round_idx in range(config.n_association_rounds):
            # Co-project both areas to C
            b.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_A": ["AREA_C"], "AREA_B": ["AREA_C"]}
            )
            
            # Measure overlap with original assemblies
            current_c = np.array(b.area_by_name["AREA_C"].winners, dtype=np.uint32)
            overlap_a = measure_overlap(current_c, assembly_c_from_a)
            overlap_b = measure_overlap(current_c, assembly_c_from_b)
            
            # Track combined overlap
            combined_overlap = (overlap_a + overlap_b) / 2
            overlap_history.append(combined_overlap)
        
        final_overlap = overlap_history[-1]
        overlap_increase = final_overlap - initial_overlap
        
        self.log(f"    Final overlap: {final_overlap:.3f} (increase: {overlap_increase:+.3f})")
        
        return {
            "trial_id": trial_id,
            "config": {
                "n_neurons": config.n_neurons,
                "k_active": config.k_active,
                "p_connect": config.p_connect,
                "beta": config.beta,
            },
            "initial_overlap": initial_overlap,
            "final_overlap": final_overlap,
            "overlap_increase": overlap_increase,
            "overlap_history": overlap_history,
            "association_successful": final_overlap > 0.5,  # Assembly captures both sources
            "assembly_a_size": len(assembly_a),
            "assembly_b_size": len(assembly_b),
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
        """Run association experiment across parameter ranges."""
        self._start_timer()
        
        # Default parameter ranges
        if n_neurons_range is None:
            n_neurons_range = [1000, 10000]
        if k_active_range is None:
            k_active_range = [50, 100]
        if p_connect_range is None:
            p_connect_range = [0.05, 0.1]
        if beta_range is None:
            beta_range = [0.05, 0.1]
        
        self.log(f"Starting association binding experiment")
        self.log(f"  n_neurons: {n_neurons_range}")
        self.log(f"  k_active: {k_active_range}")
        
        all_results = []
        
        for n in n_neurons_range:
            for k in k_active_range:
                if k >= n:
                    continue
                    
                for p in p_connect_range:
                    for beta in beta_range:
                        self.log(f"\nConfig: n={n}, k={k}, p={p}, beta={beta}")
                        
                        config = AssociationConfig(
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
                                    "association_successful": False,
                                })
                        
                        successful_trials = [r for r in trial_results if "error" not in r]
                        
                        if successful_trials:
                            success_rate = sum(1 for r in successful_trials if r["association_successful"]) / len(successful_trials)
                            mean_overlap_increase = np.mean([r["overlap_increase"] for r in successful_trials])
                            mean_final_overlap = np.mean([r["final_overlap"] for r in successful_trials])
                        else:
                            success_rate = 0.0
                            mean_overlap_increase = 0.0
                            mean_final_overlap = 0.0
                        
                        all_results.append({
                            "config": {
                                "n_neurons": n,
                                "k_active": k,
                                "p_connect": p,
                                "beta": beta,
                            },
                            "success_rate": success_rate,
                            "mean_overlap_increase": mean_overlap_increase,
                            "mean_final_overlap": mean_final_overlap,
                            "trial_details": trial_results,
                        })
        
        duration = self._stop_timer()
        
        summary = {
            "total_configurations": len(all_results),
            "overall_success_rate": np.mean([r["success_rate"] for r in all_results]) if all_results else 0,
            "mean_overlap_increase": np.mean([r["mean_overlap_increase"] for r in all_results]) if all_results else 0,
        }
        
        self.log(f"\n{'='*60}")
        self.log(f"SUMMARY:")
        self.log(f"  Overall success rate: {summary['overall_success_rate']:.2%}")
        self.log(f"  Mean overlap increase: {summary['mean_overlap_increase']:.3f}")
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
    """Run a quick test."""
    print("="*60)
    print("QUICK TEST: Association Binding")
    print("="*60)
    
    exp = AssociationBindingExperiment(verbose=True)
    
    result = exp.run(
        n_neurons_range=[1000, 5000],
        k_active_range=[50],
        p_connect_range=[0.1],
        beta_range=[0.1],
        n_trials=3,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Association Binding Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = AssociationBindingExperiment(verbose=True)
        result = exp.run(n_trials=5)
        exp.save_result(result, "_full")

