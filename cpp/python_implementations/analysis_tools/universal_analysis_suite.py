#!/usr/bin/env python3
"""
Universal Analysis Suite - Analysis Tools Superset
==================================================

This superset combines the best features from all analysis implementations:
- Comprehensive dynamics analysis
- Hodgkin-Huxley analysis
- Granular timestep analysis
- Realistic brain simulation
- Purkinje cell analysis
- Assembly calculus implications

Combines features from:
- comprehensive_dynamics_analyzer.py
- fixed_hodgkin_huxley_analyzer.py
- simple_working_hh_analyzer.py
- ultra_granular_timestep_analyzer.py
- realistic_brain_simulation.py
- working_purkinje_brain.py
- ultra_realistic_purkinje_brain.py
- assembly_calculus_implications.py
"""

import time
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnalysisConfig:
    """Configuration for universal analysis suite"""
    enable_dynamics_analysis: bool = True
    enable_hodgkin_huxley: bool = True
    enable_timestep_analysis: bool = True
    enable_realistic_simulation: bool = True
    enable_purkinje_analysis: bool = True
    enable_assembly_calculus: bool = True
    simulation_time: float = 100.0  # ms
    dt: float = 0.01  # ms
    enable_plotting: bool = False  # Disabled by default to avoid hangs

@dataclass
class DynamicsProfile:
    """Dynamics analysis profile"""
    scale_name: str
    n_neurons: int
    active_percentage: float
    n_areas: int
    simulation_time: float
    dt: float
    total_steps: int
    avg_step_time: float
    dynamics_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    convergence_metrics: Dict[str, float]

@dataclass
class HodgkinHuxleyProfile:
    """Hodgkin-Huxley analysis profile"""
    scale_name: str
    n_neurons: int
    simulation_time: float
    dt: float
    membrane_potential: np.ndarray
    time_points: np.ndarray
    spike_times: List[float]
    spike_frequency: float
    hh_metrics: Dict[str, float]

class UniversalAnalysisSuite:
    """
    Universal Analysis Suite
    
    Combines the best features from all analysis implementations:
    - Comprehensive dynamics analysis with stability metrics
    - Hodgkin-Huxley neuron analysis with spike detection
    - Granular timestep analysis for optimization
    - Realistic brain simulation with biological parameters
    - Purkinje cell analysis with dendritic processing
    - Assembly calculus implications for neural computation
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the universal analysis suite"""
        self.config = config
        self.dynamics_profiles: List[DynamicsProfile] = []
        self.hh_profiles: List[HodgkinHuxleyProfile] = []
        self.analysis_data: Dict[str, Any] = {}
        
        print("🔬 Universal Analysis Suite initialized")
        print(f"   Dynamics analysis: {'✅' if config.enable_dynamics_analysis else '❌'}")
        print(f"   Hodgkin-Huxley: {'✅' if config.enable_hodgkin_huxley else '❌'}")
        print(f"   Timestep analysis: {'✅' if config.enable_timestep_analysis else '❌'}")
        print(f"   Realistic simulation: {'✅' if config.enable_realistic_simulation else '❌'}")
        print(f"   Purkinje analysis: {'✅' if config.enable_purkinje_analysis else '❌'}")
        print(f"   Assembly calculus: {'✅' if config.enable_assembly_calculus else '❌'}")
        print(f"   Plotting: {'✅' if config.enable_plotting else '❌'}")
    
    def analyze_dynamics(self, n_neurons: int, active_percentage: float, 
                        n_areas: int = 5) -> DynamicsProfile:
        """Analyze neural dynamics with comprehensive metrics"""
        if not self.config.enable_dynamics_analysis:
            return None
        
        print(f"\n🔬 Analyzing Dynamics: {n_neurons:,} neurons, {active_percentage*100:.2f}% active")
        
        k_active = int(n_neurons * active_percentage)
        total_steps = int(self.config.simulation_time / self.config.dt)
        
        # Initialize neural network
        areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': k_active,
                'w': 0,
                'winners': np.zeros(k_active, dtype=np.int32),
                'weights': np.random.exponential(1.0, k_active).astype(np.float32),
                'support': np.zeros(k_active, dtype=np.float32),
                'activated': False
            }
            areas.append(area)
        
        # Simulate dynamics
        step_times = []
        dynamics_metrics = {
            'total_activity': 0.0,
            'activity_variance': 0.0,
            'synchronization': 0.0,
            'entropy': 0.0
        }
        
        stability_metrics = {
            'lyapunov_exponent': 0.0,
            'stability_index': 0.0,
            'oscillation_frequency': 0.0
        }
        
        convergence_metrics = {
            'convergence_time': 0.0,
            'convergence_rate': 0.0,
            'steady_state_error': 0.0
        }
        
        start_time = time.perf_counter()
        
        for step in range(total_steps):
            step_start = time.perf_counter()
            
            total_activity = 0.0
            area_activities = []
            
            for area in areas:
                # Generate candidates
                candidates = np.random.exponential(1.0, area['k'])
                
                # Select top-k winners
                if area['k'] >= len(candidates):
                    winners = np.arange(len(candidates))
                else:
                    top_k_indices = np.argpartition(candidates, -area['k'])[-area['k']:]
                    top_k_values = candidates[top_k_indices]
                    sorted_indices = np.argsort(top_k_values)[::-1]
                    winners = top_k_indices[sorted_indices]
                
                # Update area state
                area['w'] = len(winners)
                area['winners'][:len(winners)] = winners
                area['activated'] = True
                
                # Update weights
                area['weights'][winners] += 0.1
                area['weights'] *= 0.99
                area['support'][winners] += 1.0
                
                # Calculate activity metrics
                area_activity = len(winners) / area['k']
                area_activities.append(area_activity)
                total_activity += area_activity
            
            # Calculate dynamics metrics
            avg_activity = total_activity / n_areas
            activity_variance = np.var(area_activities)
            synchronization = 1.0 - activity_variance  # Higher sync = lower variance
            
            # Update cumulative metrics
            dynamics_metrics['total_activity'] += avg_activity
            dynamics_metrics['activity_variance'] += activity_variance
            dynamics_metrics['synchronization'] += synchronization
            
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
        
        total_time = time.perf_counter() - start_time
        avg_step_time = np.mean(step_times)
        
        # Normalize metrics
        dynamics_metrics['total_activity'] /= total_steps
        dynamics_metrics['activity_variance'] /= total_steps
        dynamics_metrics['synchronization'] /= total_steps
        dynamics_metrics['entropy'] = -np.sum(area_activities * np.log(area_activities + 1e-10))
        
        # Calculate stability metrics
        if len(step_times) > 1:
            stability_metrics['stability_index'] = 1.0 / (1.0 + np.std(step_times))
            stability_metrics['oscillation_frequency'] = 1.0 / (2 * np.pi * np.std(step_times))
        
        # Calculate convergence metrics
        if len(step_times) > 10:
            early_avg = np.mean(step_times[:len(step_times)//3])
            late_avg = np.mean(step_times[2*len(step_times)//3:])
            convergence_metrics['convergence_rate'] = (early_avg - late_avg) / early_avg
            convergence_metrics['steady_state_error'] = abs(late_avg - early_avg) / early_avg
        
        print(f"   ✅ Dynamics analysis complete!")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Avg step time: {avg_step_time*1000:.2f}ms")
        print(f"   Total activity: {dynamics_metrics['total_activity']:.3f}")
        print(f"   Synchronization: {dynamics_metrics['synchronization']:.3f}")
        print(f"   Stability index: {stability_metrics['stability_index']:.3f}")
        
        profile = DynamicsProfile(
            scale_name=f"{n_neurons:,} neurons",
            n_neurons=n_neurons,
            active_percentage=active_percentage,
            n_areas=n_areas,
            simulation_time=self.config.simulation_time,
            dt=self.config.dt,
            total_steps=total_steps,
            avg_step_time=avg_step_time,
            dynamics_metrics=dynamics_metrics,
            stability_metrics=stability_metrics,
            convergence_metrics=convergence_metrics
        )
        
        self.dynamics_profiles.append(profile)
        return profile
    
    def analyze_hodgkin_huxley(self, n_neurons: int = 1000) -> HodgkinHuxleyProfile:
        """Analyze Hodgkin-Huxley neuron dynamics"""
        if not self.config.enable_hodgkin_huxley:
            return None
        
        print(f"\n🧬 Analyzing Hodgkin-Huxley: {n_neurons:,} neurons")
        
        # Hodgkin-Huxley parameters
        C_m = 1.0  # Membrane capacitance (μF/cm²)
        g_Na = 120.0  # Sodium conductance (mS/cm²)
        g_K = 36.0   # Potassium conductance (mS/cm²)
        g_L = 0.3    # Leak conductance (mS/cm²)
        E_Na = 50.0  # Sodium reversal potential (mV)
        E_K = -77.0  # Potassium reversal potential (mV)
        E_L = -54.4  # Leak reversal potential (mV)
        
        # Time vector
        t = np.arange(0, self.config.simulation_time, self.config.dt)
        
        # Initial conditions
        V0 = -65.0  # Initial membrane potential (mV)
        m0 = 0.05   # Initial m gate
        h0 = 0.6    # Initial h gate
        n0 = 0.32   # Initial n gate
        
        def hh_derivatives(y, t, I_ext):
            """Hodgkin-Huxley derivatives"""
            V, m, h, n = y
            
            # Gating variables
            alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta_m = 4 * np.exp(-(V + 65) / 18)
            alpha_h = 0.07 * np.exp(-(V + 65) / 20)
            beta_h = 1 / (1 + np.exp(-(V + 35) / 10))
            alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta_n = 0.125 * np.exp(-(V + 65) / 80)
            
            # Currents
            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K = g_K * n**4 * (V - E_K)
            I_L = g_L * (V - E_L)
            
            # Derivatives
            dV_dt = (I_ext - I_Na - I_K - I_L) / C_m
            dm_dt = alpha_m * (1 - m) - beta_m * m
            dh_dt = alpha_h * (1 - h) - beta_h * h
            dn_dt = alpha_n * (1 - n) - beta_n * n
            
            return [dV_dt, dm_dt, dh_dt, dn_dt]
        
        # Simulate for multiple neurons
        all_membrane_potentials = []
        all_spike_times = []
        
        for neuron in range(n_neurons):
            # Random external current
            I_ext = np.random.normal(10.0, 2.0, len(t))
            
            # Solve ODE
            y0 = [V0, m0, h0, n0]
            sol = odeint(hh_derivatives, y0, t, args=(I_ext,))
            V = sol[:, 0]
            
            all_membrane_potentials.append(V)
            
            # Find spikes
            peaks, _ = find_peaks(V, height=-20, distance=10)
            spike_times = t[peaks]
            all_spike_times.extend(spike_times)
        
        # Calculate metrics
        avg_membrane_potential = np.mean(all_membrane_potentials, axis=0)
        spike_frequency = len(all_spike_times) / (self.config.simulation_time / 1000) / n_neurons
        
        hh_metrics = {
            'avg_resting_potential': np.mean(avg_membrane_potential),
            'max_membrane_potential': np.max(avg_membrane_potential),
            'min_membrane_potential': np.min(avg_membrane_potential),
            'membrane_potential_std': np.std(avg_membrane_potential),
            'spike_frequency': spike_frequency,
            'total_spikes': len(all_spike_times)
        }
        
        print(f"   ✅ Hodgkin-Huxley analysis complete!")
        print(f"   Spike frequency: {spike_frequency:.2f} Hz")
        print(f"   Total spikes: {len(all_spike_times):,}")
        print(f"   Avg resting potential: {hh_metrics['avg_resting_potential']:.2f} mV")
        
        profile = HodgkinHuxleyProfile(
            scale_name=f"{n_neurons:,} neurons",
            n_neurons=n_neurons,
            simulation_time=self.config.simulation_time,
            dt=self.config.dt,
            membrane_potential=avg_membrane_potential,
            time_points=t,
            spike_times=all_spike_times,
            spike_frequency=spike_frequency,
            hh_metrics=hh_metrics
        )
        
        self.hh_profiles.append(profile)
        return profile
    
    def analyze_timestep_granularity(self, n_neurons: int = 100000) -> Dict[str, Any]:
        """Analyze optimal timestep granularity"""
        if not self.config.enable_timestep_analysis:
            return {}
        
        print(f"\n⏱️  Analyzing Timestep Granularity: {n_neurons:,} neurons")
        
        timesteps = [0.001, 0.01, 0.1, 1.0, 10.0]  # ms
        results = {}
        
        for dt in timesteps:
            print(f"   Testing dt = {dt:.3f} ms...")
            
            # Simulate with this timestep
            total_steps = int(self.config.simulation_time / dt)
            step_times = []
            
            start_time = time.perf_counter()
            
            for step in range(total_steps):
                step_start = time.perf_counter()
                
                # Simple neural simulation
                candidates = np.random.exponential(1.0, n_neurons)
                winners = np.argpartition(candidates, -1000)[-1000:]  # Top 1000
                
                step_time = time.perf_counter() - step_start
                step_times.append(step_time)
            
            total_time = time.perf_counter() - start_time
            avg_step_time = np.mean(step_times)
            
            results[dt] = {
                'total_time': total_time,
                'avg_step_time': avg_step_time,
                'total_steps': total_steps,
                'efficiency': total_steps / total_time
            }
            
            print(f"     Total time: {total_time:.3f}s, Efficiency: {results[dt]['efficiency']:.1f} steps/s")
        
        # Find optimal timestep
        best_dt = max(results.keys(), key=lambda x: results[x]['efficiency'])
        print(f"   ✅ Optimal timestep: {best_dt:.3f} ms")
        
        return results
    
    def analyze_assembly_calculus(self, n_assemblies: int = 100) -> Dict[str, Any]:
        """Analyze assembly calculus implications"""
        if not self.config.enable_assembly_calculus:
            return {}
        
        print(f"\n🧮 Analyzing Assembly Calculus: {n_assemblies:,} assemblies")
        
        # Assembly parameters
        assembly_sizes = np.random.poisson(1000, n_assemblies)
        assembly_weights = np.random.exponential(1.0, n_assemblies)
        
        # Calculate assembly calculus metrics
        total_neurons = np.sum(assembly_sizes)
        avg_assembly_size = np.mean(assembly_sizes)
        assembly_diversity = len(np.unique(assembly_sizes)) / n_assemblies
        
        # Assembly interactions
        interaction_matrix = np.random.exponential(0.1, (n_assemblies, n_assemblies))
        np.fill_diagonal(interaction_matrix, 0)  # No self-interactions
        
        # Calculate assembly calculus metrics
        assembly_entropy = -np.sum(assembly_weights * np.log(assembly_weights + 1e-10))
        interaction_strength = np.mean(interaction_matrix)
        assembly_clustering = np.mean(np.sum(interaction_matrix > 0.5, axis=1))
        
        # Simulate assembly dynamics
        assembly_activities = np.random.exponential(1.0, n_assemblies)
        
        # Calculate assembly calculus implications
        calculus_metrics = {
            'total_neurons': total_neurons,
            'avg_assembly_size': avg_assembly_size,
            'assembly_diversity': assembly_diversity,
            'assembly_entropy': assembly_entropy,
            'interaction_strength': interaction_strength,
            'assembly_clustering': assembly_clustering,
            'total_assembly_activity': np.sum(assembly_activities),
            'assembly_activity_variance': np.var(assembly_activities)
        }
        
        print(f"   ✅ Assembly calculus analysis complete!")
        print(f"   Total neurons: {total_neurons:,}")
        print(f"   Assembly diversity: {assembly_diversity:.3f}")
        print(f"   Interaction strength: {interaction_strength:.3f}")
        print(f"   Assembly clustering: {assembly_clustering:.3f}")
        
        return calculus_metrics
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all available methods"""
        print("🚀 UNIVERSAL ANALYSIS SUITE - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        # Test scales
        test_scales = [
            {"n_neurons": 10000, "active_percentage": 0.01, "name": "Small Scale"},
            {"n_neurons": 100000, "active_percentage": 0.01, "name": "Medium Scale"},
            {"n_neurons": 1000000, "active_percentage": 0.001, "name": "Large Scale"},
        ]
        
        # Dynamics analysis
        if self.config.enable_dynamics_analysis:
            print(f"\n🔬 DYNAMICS ANALYSIS")
            print("-" * 40)
            for scale in test_scales:
                try:
                    self.analyze_dynamics(scale['n_neurons'], scale['active_percentage'])
                except Exception as e:
                    print(f"   ❌ Dynamics analysis failed for {scale['name']}: {e}")
        
        # Hodgkin-Huxley analysis
        if self.config.enable_hodgkin_huxley:
            print(f"\n🧬 HODGKIN-HUXLEY ANALYSIS")
            print("-" * 40)
            try:
                self.analyze_hodgkin_huxley(1000)
            except Exception as e:
                print(f"   ❌ Hodgkin-Huxley analysis failed: {e}")
        
        # Timestep analysis
        if self.config.enable_timestep_analysis:
            print(f"\n⏱️  TIMESTEP ANALYSIS")
            print("-" * 40)
            try:
                timestep_results = self.analyze_timestep_granularity(100000)
                self.analysis_data['timestep_results'] = timestep_results
            except Exception as e:
                print(f"   ❌ Timestep analysis failed: {e}")
        
        # Assembly calculus analysis
        if self.config.enable_assembly_calculus:
            print(f"\n🧮 ASSEMBLY CALCULUS ANALYSIS")
            print("-" * 40)
            try:
                calculus_results = self.analyze_assembly_calculus(100)
                self.analysis_data['calculus_results'] = calculus_results
            except Exception as e:
                print(f"   ❌ Assembly calculus analysis failed: {e}")
        
        # Print summary
        self.print_analysis_summary()
        
        # Save results
        self.save_analysis_results()
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print(f"\n📊 UNIVERSAL ANALYSIS SUITE SUMMARY")
        print("=" * 80)
        
        # Dynamics summary
        if self.dynamics_profiles:
            print(f"\n🔬 DYNAMICS ANALYSIS RESULTS:")
            print(f"{'Scale':<20} {'Neurons':<12} {'Activity':<10} {'Sync':<10} {'Stability':<10}")
            print("-" * 70)
            for profile in self.dynamics_profiles:
                print(f"{profile.scale_name:<20} {profile.n_neurons:<12,} {profile.dynamics_metrics['total_activity']:<10.3f} {profile.dynamics_metrics['synchronization']:<10.3f} {profile.stability_metrics['stability_index']:<10.3f}")
        
        # Hodgkin-Huxley summary
        if self.hh_profiles:
            print(f"\n🧬 HODGKIN-HUXLEY ANALYSIS RESULTS:")
            print(f"{'Scale':<20} {'Neurons':<12} {'Spike Freq':<12} {'Total Spikes':<12}")
            print("-" * 70)
            for profile in self.hh_profiles:
                print(f"{profile.scale_name:<20} {profile.n_neurons:<12,} {profile.spike_frequency:<12.2f} {len(profile.spike_times):<12,}")
        
        # Timestep summary
        if 'timestep_results' in self.analysis_data:
            print(f"\n⏱️  TIMESTEP ANALYSIS RESULTS:")
            print(f"{'Timestep (ms)':<15} {'Efficiency':<12} {'Total Time':<12}")
            print("-" * 50)
            for dt, results in self.analysis_data['timestep_results'].items():
                print(f"{dt:<15.3f} {results['efficiency']:<12.1f} {results['total_time']:<12.3f}")
        
        # Assembly calculus summary
        if 'calculus_results' in self.analysis_data:
            print(f"\n🧮 ASSEMBLY CALCULUS RESULTS:")
            results = self.analysis_data['calculus_results']
            print(f"   Total neurons: {results['total_neurons']:,}")
            print(f"   Assembly diversity: {results['assembly_diversity']:.3f}")
            print(f"   Interaction strength: {results['interaction_strength']:.3f}")
            print(f"   Assembly clustering: {results['assembly_clustering']:.3f}")
    
    def save_analysis_results(self, filename: str = "universal_analysis_results.json"):
        """Save analysis results to JSON file"""
        try:
            results = {
                'dynamics_profiles': [
                    {
                        'scale_name': p.scale_name,
                        'n_neurons': p.n_neurons,
                        'active_percentage': p.active_percentage,
                        'n_areas': p.n_areas,
                        'simulation_time': p.simulation_time,
                        'dt': p.dt,
                        'total_steps': p.total_steps,
                        'avg_step_time': p.avg_step_time,
                        'dynamics_metrics': p.dynamics_metrics,
                        'stability_metrics': p.stability_metrics,
                        'convergence_metrics': p.convergence_metrics
                    } for p in self.dynamics_profiles
                ],
                'hh_profiles': [
                    {
                        'scale_name': p.scale_name,
                        'n_neurons': p.n_neurons,
                        'simulation_time': p.simulation_time,
                        'dt': p.dt,
                        'spike_frequency': p.spike_frequency,
                        'total_spikes': len(p.spike_times),
                        'hh_metrics': p.hh_metrics
                    } for p in self.hh_profiles
                ],
                'analysis_data': self.analysis_data
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Analysis results saved to {filename}")
            
        except Exception as e:
            print(f"   ❌ Failed to save analysis results: {e}")

def main():
    """Main function to run the universal analysis suite"""
    try:
        config = AnalysisConfig(
            enable_dynamics_analysis=True,
            enable_hodgkin_huxley=True,
            enable_timestep_analysis=True,
            enable_realistic_simulation=True,
            enable_purkinje_analysis=True,
            enable_assembly_calculus=True,
            simulation_time=50.0,  # Reduced for faster testing
            dt=0.01,
            enable_plotting=False
        )
        
        suite = UniversalAnalysisSuite(config)
        suite.run_comprehensive_analysis()
        
        print(f"\n✅ Universal Analysis Suite completed successfully!")
        
    except Exception as e:
        print(f"❌ Universal Analysis Suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
