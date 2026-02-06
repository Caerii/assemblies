"""
Merge Primitive Validation

Scientific Questions:
1. Does merge create a new assembly representing both parent assemblies?
2. Is the merged assembly distinguishable from both parents?
3. Does the merged assembly maintain connections to both parents?
4. Is merge compositional (can we decode both components)?

Expected Results:
- Merged assembly should have significant overlap with both parents
- Merged assembly should be a distinct entity (not just union)
- Projecting from merged assembly should activate both parent concepts
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
class MergeConfig:
    """Configuration for merge test."""
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    n_projection_rounds: int = 10  # To establish assemblies
    n_merge_rounds: int = 10       # Rounds for merge operation


class MergeCompositionExperiment(ExperimentBase):
    """
    Test: Does merge create composite assemblies?
    
    Hypothesis: When two assemblies A and B project simultaneously to 
    area C, the resulting assembly C should:
    1. Have overlap with both A and B projections individually
    2. Be distinct from either A or B alone
    3. Preserve information about both components
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="merge_composition",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "primitives",
            verbose=verbose
        )
    
    def run_single_trial(
        self, 
        config: MergeConfig,
        trial_id: int = 0
    ) -> Dict[str, Any]:
        """
        Run a single merge trial.
        
        Tests merge by:
        1. Creating assemblies A and B in separate areas
        2. Projecting A alone to C -> record C_A
        3. Projecting B alone to C -> record C_B  
        4. Projecting A and B together to C -> record C_AB (merged)
        5. Verify C_AB has properties of both C_A and C_B
        """
        self.log(f"  Trial {trial_id}: n={config.n_neurons}, k={config.k_active}")
        
        # Create brain
        b = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Add stimuli
        b.add_stimulus("STIM_A", config.k_active)
        b.add_stimulus("STIM_B", config.k_active)
        
        # Add source areas
        b.add_area("AREA_A", config.n_neurons, config.k_active, config.beta)
        b.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        
        # Add merge target area
        b.add_area("MERGE", config.n_neurons, config.k_active, config.beta)
        
        # Phase 1: Establish assembly A
        self.log(f"    Establishing assembly A...")
        for _ in range(config.n_projection_rounds):
            b.project(areas_by_stim={"STIM_A": ["AREA_A"]}, dst_areas_by_src_area={})
        assembly_a = np.array(b.area_by_name["AREA_A"].winners, dtype=np.uint32)
        
        # Phase 2: Establish assembly B
        self.log(f"    Establishing assembly B...")
        for _ in range(config.n_projection_rounds):
            b.project(areas_by_stim={"STIM_B": ["AREA_B"]}, dst_areas_by_src_area={})
        assembly_b = np.array(b.area_by_name["AREA_B"].winners, dtype=np.uint32)
        
        # Phase 3: Project A alone to MERGE -> C_A
        self.log(f"    Projecting A -> MERGE...")
        for _ in range(config.n_projection_rounds):
            b.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_A": ["MERGE"]}
            )
        c_a = np.array(b.area_by_name["MERGE"].winners, dtype=np.uint32)
        
        # Phase 4: Fresh brain for B -> MERGE projection
        b2 = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id + 1000)
        b2.add_stimulus("STIM_B", config.k_active)
        b2.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        b2.add_area("MERGE", config.n_neurons, config.k_active, config.beta)
        
        self.log(f"    Projecting B -> MERGE...")
        for _ in range(config.n_projection_rounds):
            b2.project(areas_by_stim={"STIM_B": ["AREA_B"]}, dst_areas_by_src_area={})
        
        for _ in range(config.n_projection_rounds):
            b2.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_B": ["MERGE"]}
            )
        c_b = np.array(b2.area_by_name["MERGE"].winners, dtype=np.uint32)
        
        # Phase 5: Merge - project A and B together to MERGE
        # Use a fresh brain with both assemblies established for clean merge
        self.log(f"    Performing MERGE (A + B -> MERGE)...")
        
        b3 = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id + 2000)
        b3.add_stimulus("STIM_A", config.k_active)
        b3.add_stimulus("STIM_B", config.k_active)
        b3.add_area("AREA_A", config.n_neurons, config.k_active, config.beta)
        b3.add_area("AREA_B", config.n_neurons, config.k_active, config.beta)
        b3.add_area("MERGE", config.n_neurons, config.k_active, config.beta)
        
        # Establish both assemblies
        for _ in range(config.n_projection_rounds):
            b3.project(areas_by_stim={"STIM_A": ["AREA_A"]}, dst_areas_by_src_area={})
        for _ in range(config.n_projection_rounds):
            b3.project(areas_by_stim={"STIM_B": ["AREA_B"]}, dst_areas_by_src_area={})
        
        # Now do the merge: project both simultaneously
        merge_history = []
        for round_idx in range(config.n_merge_rounds):
            b3.project(
                areas_by_stim={},
                dst_areas_by_src_area={"AREA_A": ["MERGE"], "AREA_B": ["MERGE"]}
            )
            current = np.array(b3.area_by_name["MERGE"].winners, dtype=np.uint32)
            merge_history.append(current.copy())
        
        c_ab = np.array(b3.area_by_name["MERGE"].winners, dtype=np.uint32)
        
        # Compute metrics
        overlap_ca_cb = measure_overlap(c_a, c_b)  # How similar are A->C and B->C?
        overlap_cab_ca = measure_overlap(c_ab, c_a)  # How much of A is in merge?
        overlap_cab_cb = measure_overlap(c_ab, c_b)  # How much of B is in merge?
        
        # Jaccard similarities
        jaccard_ca_cb = measure_jaccard(c_a, c_b)
        jaccard_cab_ca = measure_jaccard(c_ab, c_a)
        jaccard_cab_cb = measure_jaccard(c_ab, c_b)
        
        # Merge quality: should have significant overlap with BOTH parents
        merge_quality = min(overlap_cab_ca, overlap_cab_cb)
        
        # Composition score: average overlap with both parents
        composition_score = (overlap_cab_ca + overlap_cab_cb) / 2
        
        # Distinctness: merged should not be identical to either parent
        distinctness = 1.0 - max(jaccard_cab_ca, jaccard_cab_cb)
        
        self.log(f"    Overlap C_A vs C_B: {overlap_ca_cb:.3f}")
        self.log(f"    Overlap C_AB vs C_A: {overlap_cab_ca:.3f}")
        self.log(f"    Overlap C_AB vs C_B: {overlap_cab_cb:.3f}")
        self.log(f"    Merge quality: {merge_quality:.3f}")
        self.log(f"    Composition score: {composition_score:.3f}")
        
        # Success criteria:
        # 1. Merged assembly captures both parents (min overlap > 0.3)
        # 2. Merged assembly is compositional (composition_score > 0.5)
        merge_successful = merge_quality > 0.3 and composition_score > 0.5
        
        return {
            "trial_id": trial_id,
            "config": {
                "n_neurons": config.n_neurons,
                "k_active": config.k_active,
                "p_connect": config.p_connect,
                "beta": config.beta,
            },
            "overlap_ca_cb": overlap_ca_cb,
            "overlap_cab_ca": overlap_cab_ca,
            "overlap_cab_cb": overlap_cab_cb,
            "jaccard_ca_cb": jaccard_ca_cb,
            "jaccard_cab_ca": jaccard_cab_ca,
            "jaccard_cab_cb": jaccard_cab_cb,
            "merge_quality": merge_quality,
            "composition_score": composition_score,
            "distinctness": distinctness,
            "merge_successful": merge_successful,
            "assembly_sizes": {
                "a": len(assembly_a),
                "b": len(assembly_b),
                "c_a": len(c_a),
                "c_b": len(c_b),
                "c_ab": len(c_ab),
            },
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
        """Run merge experiment across parameter ranges."""
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
        
        self.log(f"Starting merge composition experiment")
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
                        
                        config = MergeConfig(
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
                                import traceback
                                traceback.print_exc()
                                trial_results.append({
                                    "trial_id": trial,
                                    "error": str(e),
                                    "merge_successful": False,
                                })
                        
                        successful_trials = [r for r in trial_results if "error" not in r]
                        
                        if successful_trials:
                            success_rate = sum(1 for r in successful_trials if r["merge_successful"]) / len(successful_trials)
                            mean_merge_quality = np.mean([r["merge_quality"] for r in successful_trials])
                            mean_composition = np.mean([r["composition_score"] for r in successful_trials])
                        else:
                            success_rate = 0.0
                            mean_merge_quality = 0.0
                            mean_composition = 0.0
                        
                        all_results.append({
                            "config": {
                                "n_neurons": n,
                                "k_active": k,
                                "p_connect": p,
                                "beta": beta,
                            },
                            "success_rate": success_rate,
                            "mean_merge_quality": mean_merge_quality,
                            "mean_composition_score": mean_composition,
                            "trial_details": trial_results,
                        })
        
        duration = self._stop_timer()
        
        summary = {
            "total_configurations": len(all_results),
            "overall_success_rate": np.mean([r["success_rate"] for r in all_results]) if all_results else 0,
            "mean_merge_quality": np.mean([r["mean_merge_quality"] for r in all_results]) if all_results else 0,
            "mean_composition_score": np.mean([r["mean_composition_score"] for r in all_results]) if all_results else 0,
        }
        
        self.log(f"\n{'='*60}")
        self.log(f"SUMMARY:")
        self.log(f"  Overall success rate: {summary['overall_success_rate']:.2%}")
        self.log(f"  Mean merge quality: {summary['mean_merge_quality']:.3f}")
        self.log(f"  Mean composition score: {summary['mean_composition_score']:.3f}")
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
    print("QUICK TEST: Merge Composition")
    print("="*60)
    
    exp = MergeCompositionExperiment(verbose=True)
    
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
    
    parser = argparse.ArgumentParser(description="Merge Composition Experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = MergeCompositionExperiment(verbose=True)
        result = exp.run(n_trials=5)
        exp.save_result(result, "_full")

