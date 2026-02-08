"""
Biological Parameter Validation

Scientific Questions:
1. Do our simulation parameters match biological measurements?
2. What sparsity levels are observed in real cortex?
3. What firing rates correspond to our model dynamics?
4. How do our assembly sizes compare to biological estimates?

Reference Values (from literature):
- Cortical sparsity: 1-5% of neurons active at any time
- Firing rates: 0.1-10 Hz for most cortical neurons
- Assembly sizes: 50-500 neurons (estimates vary widely)
- Connection probability: ~10% local, <1% long-range
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

from src.core.brain import Brain


# Biological reference values from literature
BIOLOGICAL_REFERENCES = {
    "cortical_sparsity": {
        "min": 0.01,  # 1%
        "max": 0.05,  # 5%
        "typical": 0.02,  # 2%
        "source": "Barth & Bhalla (2012), various cortical recordings",
    },
    "hippocampal_sparsity": {
        "min": 0.01,
        "max": 0.03,
        "typical": 0.02,
        "source": "Sparse coding in hippocampus",
    },
    "cerebellar_sparsity": {
        "min": 0.001,  # Granule cells very sparse
        "max": 0.01,
        "typical": 0.005,
        "source": "Cerebellar granule cell activity",
    },
    "cortical_firing_rate_hz": {
        "min": 0.1,
        "max": 10.0,
        "typical": 1.0,
        "source": "Various cortical recordings",
    },
    "assembly_size_neurons": {
        "min": 50,
        "max": 500,
        "typical": 100,
        "source": "Harris (2005), Buzsaki (2010)",
    },
    "local_connection_probability": {
        "min": 0.05,
        "max": 0.20,
        "typical": 0.10,
        "source": "Cortical connectivity studies",
    },
    "plasticity_timescale_ms": {
        "min": 10,
        "max": 100,
        "typical": 20,
        "source": "STDP window measurements",
    },
}


@dataclass
class BiologicalConfig:
    """Configuration matching biological parameters."""
    brain_region: str
    n_neurons: int
    k_active: int
    p_connect: float
    beta: float
    timestep_ms: float = 1.0


class BiologicalParameterExperiment(ExperimentBase):
    """
    Validate simulation parameters against biological measurements.
    
    Tests whether our model operates in biologically plausible regimes.
    """
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="biological_parameters",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "biological_validation",
            verbose=verbose
        )
    
    def check_parameter_validity(
        self,
        param_name: str,
        value: float,
    ) -> Dict[str, Any]:
        """Check if a parameter value is within biological range."""
        if param_name not in BIOLOGICAL_REFERENCES:
            return {"valid": None, "message": f"No reference for {param_name}"}
        
        ref = BIOLOGICAL_REFERENCES[param_name]
        
        in_range = ref["min"] <= value <= ref["max"]
        near_typical = abs(value - ref["typical"]) / ref["typical"] < 0.5
        
        return {
            "param_name": param_name,
            "value": value,
            "reference_min": ref["min"],
            "reference_max": ref["max"],
            "reference_typical": ref["typical"],
            "in_biological_range": in_range,
            "near_typical": near_typical,
            "source": ref["source"],
        }
    
    def simulate_with_biological_params(
        self,
        config: BiologicalConfig,
        n_steps: int = 100
    ) -> Dict[str, Any]:
        """Run simulation with biologically-inspired parameters."""
        b = Brain(p=config.p_connect, seed=self.seed, w_max=20.0)
        b.add_stimulus("STIM", config.k_active)
        b.add_area("TARGET", config.n_neurons, config.k_active, config.beta, explicit=True)
        
        # Track activity over time
        activity_history = []
        winner_history = []
        
        for step in range(n_steps):
            b.project(
                areas_by_stim={"STIM": ["TARGET"]},
                dst_areas_by_src_area={}
            )
            
            winners = np.array(b.areas["TARGET"].winners, dtype=np.uint32)
            winner_history.append(winners.copy())
            
            # Calculate instantaneous sparsity
            sparsity = len(winners) / config.n_neurons
            activity_history.append(sparsity)
        
        # Calculate firing rate (assuming timestep_ms)
        # Firing rate = (active fraction) / timestep_ms * 1000
        mean_sparsity = np.mean(activity_history)
        implied_firing_rate = mean_sparsity / config.timestep_ms * 1000
        
        # Measure assembly stability
        if len(winner_history) >= 2:
            final_stability = measure_overlap(winner_history[-1], winner_history[-2])
        else:
            final_stability = 0.0
        
        return {
            "brain_region": config.brain_region,
            "n_neurons": config.n_neurons,
            "k_active": config.k_active,
            "actual_sparsity": mean_sparsity,
            "implied_firing_rate_hz": implied_firing_rate,
            "assembly_stability": final_stability,
            "activity_variance": np.var(activity_history),
        }
    
    def run(
        self,
        test_cortical: bool = True,
        test_hippocampal: bool = True,
        test_cerebellar: bool = True,
        n_steps: int = 100,
        **kwargs
    ) -> ExperimentResult:
        """
        Run biological parameter validation.
        
        Tests simulation behavior with parameters matching different brain regions.
        """
        self._start_timer()
        
        self.log("Starting biological parameter validation")
        
        all_results = []
        parameter_checks = []
        
        # Define biologically-inspired configurations
        configs = []
        
        if test_cortical:
            # Cortical column parameters
            configs.append(BiologicalConfig(
                brain_region="cortex",
                n_neurons=10000,  # Mini-column
                k_active=200,     # 2% sparsity
                p_connect=0.1,    # 10% local connectivity
                beta=0.1,
                timestep_ms=1.0,
            ))
            
            # Check cortical parameters
            parameter_checks.append(self.check_parameter_validity(
                "cortical_sparsity", 200/10000
            ))
            parameter_checks.append(self.check_parameter_validity(
                "local_connection_probability", 0.1
            ))
        
        if test_hippocampal:
            # Hippocampal parameters
            configs.append(BiologicalConfig(
                brain_region="hippocampus",
                n_neurons=10000,
                k_active=200,     # 2% sparsity
                p_connect=0.05,   # Sparser connectivity
                beta=0.15,        # Higher plasticity
                timestep_ms=1.0,
            ))
            
            parameter_checks.append(self.check_parameter_validity(
                "hippocampal_sparsity", 200/10000
            ))
        
        if test_cerebellar:
            # Cerebellar granule cell parameters
            configs.append(BiologicalConfig(
                brain_region="cerebellum",
                n_neurons=100000,  # Many granule cells
                k_active=500,      # 0.5% sparsity
                p_connect=0.01,    # Very sparse
                beta=0.05,
                timestep_ms=1.0,
            ))
            
            parameter_checks.append(self.check_parameter_validity(
                "cerebellar_sparsity", 500/100000
            ))
        
        # Run simulations
        for config in configs:
            self.log(f"\n  Testing {config.brain_region}...")
            self.log(f"    n={config.n_neurons}, k={config.k_active}, p={config.p_connect}")
            
            try:
                result = self.simulate_with_biological_params(config, n_steps)
                all_results.append(result)
                
                self.log(f"    Sparsity: {result['actual_sparsity']:.4f}")
                self.log(f"    Implied firing rate: {result['implied_firing_rate_hz']:.2f} Hz")
                self.log(f"    Assembly stability: {result['assembly_stability']:.3f}")
                
            except Exception as e:
                self.log(f"    Failed: {e}")
                all_results.append({
                    "brain_region": config.brain_region,
                    "error": str(e),
                })
        
        duration = self._stop_timer()
        
        # Summarize parameter validity
        valid_params = sum(1 for p in parameter_checks if p.get("in_biological_range", False))
        total_params = len(parameter_checks)
        
        summary = {
            "regions_tested": len(configs),
            "parameter_checks": parameter_checks,
            "valid_parameters": valid_params,
            "total_parameters": total_params,
            "biological_validity_rate": valid_params / total_params if total_params > 0 else 0,
        }
        
        self.log(f"\n{'='*60}")
        self.log("BIOLOGICAL VALIDATION SUMMARY:")
        self.log(f"  Regions tested: {summary['regions_tested']}")
        self.log(f"  Parameters in biological range: {valid_params}/{total_params}")
        self.log(f"  Duration: {duration:.1f}s")
        
        result = ExperimentResult(
            experiment_name=self.name,
            parameters={
                "test_cortical": test_cortical,
                "test_hippocampal": test_hippocampal,
                "test_cerebellar": test_cerebellar,
                "n_steps": n_steps,
                "seed": self.seed,
            },
            metrics=summary,
            raw_data={
                "simulation_results": all_results,
                "biological_references": BIOLOGICAL_REFERENCES,
            },
            duration_seconds=duration,
        )
        
        return result


def run_quick_test():
    """Run quick biological validation test."""
    print("="*60)
    print("QUICK TEST: Biological Parameter Validation")
    print("="*60)
    
    exp = BiologicalParameterExperiment(verbose=True)
    
    result = exp.run(
        test_cortical=True,
        test_hippocampal=True,
        test_cerebellar=False,  # Skip for speed
        n_steps=50,
    )
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Biological Parameter Validation")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exp = BiologicalParameterExperiment(verbose=True)
        result = exp.run()
        exp.save_result(result, "_full")

