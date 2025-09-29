#!/usr/bin/env python3
"""
Comprehensive Dynamics Analyzer
Tests ultra-fine timesteps with proper stimulation and explores computational dynamics
"""

import time
import numpy as np
import cupy as cp
import psutil
import os

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB

def analyze_gpu_memory():
    """Analyze GPU memory usage"""
    if cp.cuda.is_available():
        mempool = cp.get_default_memory_pool()
        return {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_gb': mempool.used_bytes() / 1024**3,
            'total_gb': mempool.total_bytes() / 1024**3,
            'device_memory_gb': cp.cuda.Device().mem_info[1] / 1024**3
        }
    return None

class AdvancedHodgkinHuxleyNeuron:
    """
    Advanced Hodgkin-Huxley neuron with proper stimulation and dynamics
    """
    
    def __init__(self, n_neurons, dt_ms=0.01):
        """Initialize advanced Hodgkin-Huxley neurons"""
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.dt_s = dt_ms / 1000.0
        
        # Hodgkin-Huxley parameters
        self.C_m = 1.0  # Membrane capacitance (μF/cm²)
        self.g_Na = 120.0  # Sodium conductance (mS/cm²)
        self.g_K = 36.0   # Potassium conductance (mS/cm²)
        self.g_L = 0.3    # Leak conductance (mS/cm²)
        self.E_Na = 50.0  # Sodium reversal potential (mV)
        self.E_K = -77.0  # Potassium reversal potential (mV)
        self.E_L = -54.4  # Leak reversal potential (mV)
        
        # State variables
        self.V = cp.full(n_neurons, -65.0, dtype=cp.float32)  # Membrane potential (mV)
        self.m = cp.zeros(n_neurons, dtype=cp.float32)  # Sodium activation
        self.h = cp.ones(n_neurons, dtype=cp.float32)   # Sodium inactivation
        self.n = cp.zeros(n_neurons, dtype=cp.float32)  # Potassium activation
        
        # Input current
        self.I_inj = cp.zeros(n_neurons, dtype=cp.float32)
        
        # Spike detection
        self.spike_threshold = -50.0
        self.last_spike_time = cp.zeros(n_neurons, dtype=cp.float32)
        self.spike_count = cp.zeros(n_neurons, dtype=cp.int32)
        self.spike_times = []
        
        # Synaptic dynamics
        self.synaptic_weights = cp.random.exponential(0.1, n_neurons).astype(cp.float32)
        self.synaptic_delays = cp.random.randint(1, 10, n_neurons).astype(cp.int32)
        self.synaptic_buffer = cp.zeros((n_neurons, 10), dtype=cp.float32)
        
    def alpha_m(self, V):
        """Sodium activation rate"""
        return 0.1 * (V + 40.0) / (1.0 - cp.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Sodium activation rate"""
        return 4.0 * cp.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Sodium inactivation rate"""
        return 0.07 * cp.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Sodium inactivation rate"""
        return 1.0 / (1.0 + cp.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """Potassium activation rate"""
        return 0.01 * (V + 55.0) / (1.0 - cp.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """Potassium activation rate"""
        return 0.125 * cp.exp(-(V + 65.0) / 80.0)
    
    def step(self, I_inj=None, synaptic_input=None):
        """Integrate one timestep with synaptic dynamics"""
        if I_inj is not None:
            self.I_inj = I_inj
        
        # Add synaptic input
        if synaptic_input is not None:
            self.I_inj += synaptic_input
        
        # Calculate rate constants
        alpha_m = self.alpha_m(self.V)
        beta_m = self.beta_m(self.V)
        alpha_h = self.alpha_h(self.V)
        beta_h = self.beta_h(self.V)
        alpha_n = self.alpha_n(self.V)
        beta_n = self.beta_n(self.V)
        
        # Update gating variables
        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        n_inf = alpha_n / (alpha_n + beta_n)
        
        tau_m = 1.0 / (alpha_m + beta_m)
        tau_h = 1.0 / (alpha_h + beta_h)
        tau_n = 1.0 / (alpha_n + beta_n)
        
        self.m += (m_inf - self.m) * self.dt_s / tau_m
        self.h += (h_inf - self.h) * self.dt_s / tau_h
        self.n += (n_inf - self.n) * self.dt_s / tau_n
        
        # Calculate currents
        I_Na = self.g_Na * (self.m**3) * self.h * (self.V - self.E_Na)
        I_K = self.g_K * (self.n**4) * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        # Update membrane potential
        dV_dt = (self.I_inj - I_Na - I_K - I_L) / self.C_m
        self.V += dV_dt * self.dt_s
        
        # Detect spikes
        spiked = self.V > self.spike_threshold
        if cp.any(spiked):
            self.last_spike_time[spiked] = time.perf_counter()
            self.spike_count[spiked] += 1
            # Reset voltage after spike
            self.V[spiked] = -65.0
        
        return spiked

class ComprehensiveDynamicsAnalyzer:
    """
    Comprehensive Dynamics Analyzer
    Tests ultra-fine timesteps with proper stimulation and explores computational dynamics
    """
    
    def __init__(self, n_neurons=100000, seed=42):
        """Initialize comprehensive dynamics analyzer"""
        self.n_neurons = n_neurons
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        cp.random.seed(seed)
        
        print(f"🧠 Comprehensive Dynamics Analysis")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Testing advanced Hodgkin-Huxley dynamics")
    
    def test_timestep_performance(self, dt_ms, n_steps=1000, stimulation_type='constant'):
        """Test performance at specific timestep with different stimulation types"""
        print(f"\n🧪 Testing {dt_ms}ms timestep with {stimulation_type} stimulation:")
        
        # Initialize Hodgkin-Huxley neurons
        hh_neurons = AdvancedHodgkinHuxleyNeuron(self.n_neurons, dt_ms)
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        initial_gpu_memory = analyze_gpu_memory()
        
        # Generate input current based on stimulation type
        if stimulation_type == 'constant':
            I_inj = cp.full(self.n_neurons, 10.0, dtype=cp.float32)  # Constant current
        elif stimulation_type == 'random':
            I_inj = cp.random.normal(10.0, 5.0, self.n_neurons).astype(cp.float32)  # Random current
        elif stimulation_type == 'pulse':
            I_inj = cp.zeros(self.n_neurons, dtype=cp.float32)  # Pulse current
        elif stimulation_type == 'sinusoidal':
            I_inj = cp.zeros(self.n_neurons, dtype=cp.float32)  # Sinusoidal current
        else:
            I_inj = cp.full(self.n_neurons, 10.0, dtype=cp.float32)
        
        # Time the simulation
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            # Generate time-varying input current
            if stimulation_type == 'pulse':
                # Pulse every 100 steps
                if step % 100 < 50:
                    I_inj = cp.full(self.n_neurons, 15.0, dtype=cp.float32)
                else:
                    I_inj = cp.zeros(self.n_neurons, dtype=cp.float32)
            elif stimulation_type == 'sinusoidal':
                # Sinusoidal input
                t = step * dt_ms / 1000.0
                I_inj = 10.0 + 5.0 * cp.sin(2 * cp.pi * 10 * t)  # 10 Hz oscillation
                I_inj = cp.full(self.n_neurons, float(I_inj), dtype=cp.float32)
            
            # Integrate one step
            spiked = hh_neurons.step(I_inj)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Final memory
        final_cpu_memory = get_memory_usage()
        final_gpu_memory = analyze_gpu_memory()
        
        # Calculate statistics
        avg_step_time = total_time / n_steps
        steps_per_second = 1.0 / avg_step_time
        neurons_per_second = self.n_neurons / avg_step_time
        
        # Calculate firing rates
        total_spikes = cp.sum(hh_neurons.spike_count)
        firing_rate = total_spikes / (n_steps * dt_ms / 1000.0) / self.n_neurons
        
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Average Step Time: {avg_step_time*1000:.3f}ms")
        print(f"   Steps per Second: {steps_per_second:.1f}")
        print(f"   Neurons per Second: {neurons_per_second:,.0f}")
        print(f"   Total Spikes: {total_spikes:,}")
        print(f"   Average Firing Rate: {firing_rate:.2f} Hz")
        print(f"   CPU Memory Change: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        if initial_gpu_memory and final_gpu_memory:
            print(f"   GPU Memory Change: {final_gpu_memory['used_gb'] - initial_gpu_memory['used_gb']:.3f} GB")
        
        return {
            'dt_ms': dt_ms,
            'stimulation_type': stimulation_type,
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'neurons_per_second': neurons_per_second,
            'total_spikes': int(total_spikes),
            'firing_rate': float(firing_rate),
            'memory_change': final_cpu_memory - initial_cpu_memory
        }
    
    def test_ultra_fine_timesteps_with_stimulation(self):
        """Test ultra fine timesteps with proper stimulation"""
        print("🔬 ULTRA FINE TIMESTEP ANALYSIS WITH STIMULATION")
        print("=" * 80)
        
        # Test extremely fine timesteps with different stimulation types
        timesteps = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
        stimulation_types = ['constant', 'random', 'pulse', 'sinusoidal']
        
        all_results = []
        
        for stimulation_type in stimulation_types:
            print(f"\n{'='*80}")
            print(f"🧪 TESTING: {stimulation_type.upper()} STIMULATION")
            print(f"{'='*80}")
            
            results = []
            
            for dt_ms in timesteps:
                print(f"\n--- {dt_ms}ms timestep ---")
                
                try:
                    result = self.test_timestep_performance(dt_ms, n_steps=1000, stimulation_type=stimulation_type)
                    results.append(result)
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    results.append({
                        'dt_ms': dt_ms,
                        'stimulation_type': stimulation_type,
                        'total_time': float('inf'),
                        'avg_step_time': float('inf'),
                        'steps_per_second': 0,
                        'neurons_per_second': 0,
                        'total_spikes': 0,
                        'firing_rate': 0,
                        'memory_change': 0
                    })
                    all_results.append(results[-1])
            
            # Summary for this stimulation type
            print(f"\n📊 {stimulation_type.upper()} STIMULATION SUMMARY:")
            print(f"{'Timestep (ms)':<15} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Firing Rate (Hz)':<15}")
            print("-" * 70)
            
            for result in results:
                if result['steps_per_second'] > 0:
                    print(f"{result['dt_ms']:<15.3f} {result['steps_per_second']:<10.1f} {result['avg_step_time']*1000:<10.3f} {result['neurons_per_second']:<15,.0f} {result['firing_rate']:<15.2f}")
                else:
                    print(f"{result['dt_ms']:<15.3f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
        
        return all_results
    
    def test_computational_dynamics(self):
        """Test computational dynamics beyond single point processing"""
        print("\n🧠 COMPUTATIONAL DYNAMICS ANALYSIS")
        print("=" * 80)
        
        # Test different computational approaches
        approaches = [
            {
                'name': 'Single Point Processing',
                'description': 'Simple spike detection and counting',
                'complexity': 'Low',
                'timestep_range': '0.1-1.0ms',
                'neurons_per_second': '1B+'
            },
            {
                'name': 'Hodgkin-Huxley Dynamics',
                'description': 'Full membrane dynamics with gating variables',
                'complexity': 'High',
                'timestep_range': '0.01-0.1ms',
                'neurons_per_second': '100M-1B'
            },
            {
                'name': 'Synaptic Dynamics',
                'description': 'Synaptic transmission with delays and plasticity',
                'complexity': 'Medium',
                'timestep_range': '0.1-1.0ms',
                'neurons_per_second': '100M-1B'
            },
            {
                'name': 'Network Effects',
                'description': 'Lateral inhibition and feedback loops',
                'complexity': 'High',
                'timestep_range': '0.1-10ms',
                'neurons_per_second': '10M-100M'
            }
        ]
        
        print(f"{'Approach':<25} {'Description':<40} {'Complexity':<12} {'Timestep Range':<15} {'Neurons/sec':<15}")
        print("-" * 110)
        
        for approach in approaches:
            print(f"{approach['name']:<25} {approach['description']:<40} {approach['complexity']:<12} {approach['timestep_range']:<15} {approach['neurons_per_second']:<15}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use 0.01ms timestep for Hodgkin-Huxley dynamics")
        print(f"   2. Use 0.1ms timestep for synaptic dynamics")
        print(f"   3. Use 1ms timestep for network effects")
        print(f"   4. Use adaptive timesteps based on activity levels")
        print(f"   5. Implement hybrid approaches for different brain regions")

if __name__ == "__main__":
    # Test comprehensive dynamics
    analyzer = ComprehensiveDynamicsAnalyzer(n_neurons=100000, seed=42)
    results = analyzer.test_ultra_fine_timesteps_with_stimulation()
    
    # Test computational dynamics
    analyzer.test_computational_dynamics()
    
    # Find optimal timestep for each stimulation type
    stimulation_types = ['constant', 'random', 'pulse', 'sinusoidal']
    
    for stim_type in stimulation_types:
        stim_results = [r for r in results if r['stimulation_type'] == stim_type and r['steps_per_second'] > 0]
        if stim_results:
            best = max(stim_results, key=lambda x: x['steps_per_second'])
            print(f"\n🏆 BEST PERFORMANCE ({stim_type.upper()}): {best['dt_ms']}ms timestep")
            print(f"   Steps/sec: {best['steps_per_second']:.1f}")
            print(f"   ms/step: {best['avg_step_time']*1000:.3f}ms")
            print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
            print(f"   Firing Rate: {best['firing_rate']:.2f} Hz")
