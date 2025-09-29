#!/usr/bin/env python3
"""
Simple Working Hodgkin-Huxley Analyzer
Focus on getting proper firing rates and voltage monitoring
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

class SimpleHodgkinHuxleyNeuron:
    """
    Simple working Hodgkin-Huxley neuron with proper stimulation
    """
    
    def __init__(self, n_neurons, dt_ms=0.01):
        """Initialize simple Hodgkin-Huxley neurons"""
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
        self.spike_threshold = -55.0  # Lower threshold
        self.spike_count = cp.zeros(n_neurons, dtype=cp.int32)
        
        # Voltage monitoring
        self.voltage_history = []
        self.max_history = 100  # Keep last 100 voltage values
        
        # Initialize gating variables to steady state
        self._initialize_steady_state()
    
    def _initialize_steady_state(self):
        """Initialize gating variables to steady state at rest potential"""
        V_rest = -65.0
        
        # Calculate steady state values
        alpha_m = self.alpha_m(V_rest)
        beta_m = self.beta_m(V_rest)
        alpha_h = self.alpha_h(V_rest)
        beta_h = self.beta_h(V_rest)
        alpha_n = self.alpha_n(V_rest)
        beta_n = self.beta_n(V_rest)
        
        self.m = alpha_m / (alpha_m + beta_m)
        self.h = alpha_h / (alpha_h + beta_h)
        self.n = alpha_n / (alpha_n + beta_n)
    
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
    
    def step(self, I_inj=None):
        """Integrate one timestep"""
        if I_inj is not None:
            self.I_inj = I_inj
        
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
        
        # Update gating variables
        self.m = m_inf + (self.m - m_inf) * cp.exp(-self.dt_s / tau_m)
        self.h = h_inf + (self.h - h_inf) * cp.exp(-self.dt_s / tau_h)
        self.n = n_inf + (self.n - n_inf) * cp.exp(-self.dt_s / tau_n)
        
        # Calculate currents
        I_Na = self.g_Na * (self.m**3) * self.h * (self.V - self.E_Na)
        I_K = self.g_K * (self.n**4) * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)
        
        # Update membrane potential
        dV_dt = (self.I_inj - I_Na - I_K - I_L) / self.C_m
        self.V += dV_dt * self.dt_s
        
        # Monitor voltage (keep last few values)
        if len(self.voltage_history) < self.max_history:
            self.voltage_history.append(self.V.copy())
        else:
            self.voltage_history.pop(0)
            self.voltage_history.append(self.V.copy())
        
        # Detect spikes
        spiked = self.V > self.spike_threshold
        if cp.any(spiked):
            self.spike_count[spiked] += 1
            # Reset voltage after spike
            self.V[spiked] = -65.0
        
        return spiked
    
    def get_voltage_stats(self):
        """Get voltage statistics"""
        if not self.voltage_history:
            return {
                'mean_voltage': 0.0,
                'std_voltage': 0.0,
                'min_voltage': 0.0,
                'max_voltage': 0.0,
                'voltage_range': 0.0
            }
        
        # Get current voltage values
        current_V = self.V.get() if hasattr(self.V, 'get') else self.V
        
        return {
            'mean_voltage': float(cp.mean(self.V)),
            'std_voltage': float(cp.std(self.V)),
            'min_voltage': float(cp.min(self.V)),
            'max_voltage': float(cp.max(self.V)),
            'voltage_range': float(cp.max(self.V) - cp.min(self.V))
        }

class SimpleWorkingHHAnalyzer:
    """
    Simple working Hodgkin-Huxley analyzer
    """
    
    def __init__(self, n_neurons=1000, seed=42):
        """Initialize simple working analyzer"""
        self.n_neurons = n_neurons
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        cp.random.seed(seed)
        
        print(f"🧠 Simple Working Hodgkin-Huxley Analysis")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Testing with proper stimulation currents")
    
    def test_voltage_monitoring(self, dt_ms=0.01, n_steps=1000):
        """Test voltage monitoring capabilities"""
        print(f"\n🔬 Testing voltage monitoring at {dt_ms}ms timestep:")
        
        # Initialize neurons
        hh_neurons = SimpleHodgkinHuxleyNeuron(self.n_neurons, dt_ms)
        
        # Test with different stimulation currents
        currents = [0, 10, 20, 30, 40, 50]  # μA/cm²
        
        for current in currents:
            print(f"\n--- Testing with {current} μA/cm² current ---")
            
            # Reset neurons
            hh_neurons.V.fill(-65.0)
            hh_neurons.spike_count.fill(0)
            hh_neurons.voltage_history.clear()
            
            # Apply current
            I_inj = cp.full(self.n_neurons, current, dtype=cp.float32)
            
            # Run simulation
            for step in range(n_steps):
                spiked = hh_neurons.step(I_inj)
            
            # Get results
            total_spikes = cp.sum(hh_neurons.spike_count)
            firing_rate = total_spikes / (n_steps * dt_ms / 1000.0) / self.n_neurons
            voltage_stats = hh_neurons.get_voltage_stats()
            
            print(f"   Current: {current} μA/cm²")
            print(f"   Firing Rate: {firing_rate:.2f} Hz")
            print(f"   Mean Voltage: {voltage_stats['mean_voltage']:.2f} mV")
            print(f"   Voltage Range: {voltage_stats['min_voltage']:.2f} to {voltage_stats['max_voltage']:.2f} mV")
            print(f"   Voltage Std: {voltage_stats['std_voltage']:.2f} mV")
    
    def test_timestep_performance(self, dt_ms, n_steps=1000, stimulation_type='constant'):
        """Test timestep performance with proper stimulation"""
        print(f"\n🧪 Testing {dt_ms}ms timestep with {stimulation_type} stimulation:")
        
        # Initialize neurons
        hh_neurons = SimpleHodgkinHuxleyNeuron(self.n_neurons, dt_ms)
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        
        # Generate proper input current
        if stimulation_type == 'constant':
            I_inj = cp.full(self.n_neurons, 30.0, dtype=cp.float32)  # Higher current
        elif stimulation_type == 'random':
            I_inj = cp.random.normal(30.0, 10.0, self.n_neurons).astype(cp.float32)
        elif stimulation_type == 'pulse':
            I_inj = cp.zeros(self.n_neurons, dtype=cp.float32)
        else:
            I_inj = cp.full(self.n_neurons, 30.0, dtype=cp.float32)
        
        # Time the simulation
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            # Generate time-varying input current
            if stimulation_type == 'pulse':
                if step % 100 < 50:
                    I_inj = cp.full(self.n_neurons, 40.0, dtype=cp.float32)
                else:
                    I_inj = cp.zeros(self.n_neurons, dtype=cp.float32)
            
            # Integrate one step
            spiked = hh_neurons.step(I_inj)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Final memory
        final_cpu_memory = get_memory_usage()
        
        # Calculate statistics
        avg_step_time = total_time / n_steps
        steps_per_second = 1.0 / avg_step_time
        neurons_per_second = self.n_neurons / avg_step_time
        
        # Calculate firing rates
        total_spikes = cp.sum(hh_neurons.spike_count)
        firing_rate = total_spikes / (n_steps * dt_ms / 1000.0) / self.n_neurons
        
        # Get voltage statistics
        voltage_stats = hh_neurons.get_voltage_stats()
        
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Average Step Time: {avg_step_time*1000:.3f}ms")
        print(f"   Steps per Second: {steps_per_second:.1f}")
        print(f"   Neurons per Second: {neurons_per_second:,.0f}")
        print(f"   Total Spikes: {total_spikes:,}")
        print(f"   Average Firing Rate: {firing_rate:.2f} Hz")
        print(f"   Mean Voltage: {voltage_stats['mean_voltage']:.2f} mV")
        print(f"   Voltage Range: {voltage_stats['min_voltage']:.2f} to {voltage_stats['max_voltage']:.2f} mV")
        print(f"   CPU Memory Change: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        
        return {
            'dt_ms': dt_ms,
            'stimulation_type': stimulation_type,
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'neurons_per_second': neurons_per_second,
            'total_spikes': int(total_spikes),
            'firing_rate': float(firing_rate),
            'voltage_stats': voltage_stats,
            'memory_change': final_cpu_memory - initial_cpu_memory
        }
    
    def test_ultra_fine_timesteps(self):
        """Test ultra fine timesteps with proper stimulation"""
        print("🔬 ULTRA FINE TIMESTEP ANALYSIS (WORKING)")
        print("=" * 80)
        
        # Test timesteps
        timesteps = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
        stimulation_types = ['constant', 'random', 'pulse']
        
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
                        'voltage_stats': {'mean_voltage': 0, 'std_voltage': 0, 'min_voltage': 0, 'max_voltage': 0},
                        'memory_change': 0
                    })
                    all_results.append(results[-1])
            
            # Summary for this stimulation type
            print(f"\n📊 {stimulation_type.upper()} STIMULATION SUMMARY:")
            print(f"{'Timestep (ms)':<15} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Firing Rate (Hz)':<15} {'Mean V (mV)':<12}")
            print("-" * 85)
            
            for result in results:
                if result['steps_per_second'] > 0:
                    print(f"{result['dt_ms']:<15.3f} {result['steps_per_second']:<10.1f} {result['avg_step_time']*1000:<10.3f} {result['neurons_per_second']:<15,.0f} {result['firing_rate']:<15.2f} {result['voltage_stats']['mean_voltage']:<12.2f}")
                else:
                    print(f"{result['dt_ms']:<15.3f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15} {'FAILED':<12}")
        
        return all_results

if __name__ == "__main__":
    # Test simple working Hodgkin-Huxley
    analyzer = SimpleWorkingHHAnalyzer(n_neurons=1000, seed=42)
    
    # Test voltage monitoring
    analyzer.test_voltage_monitoring()
    
    # Test ultra fine timesteps
    results = analyzer.test_ultra_fine_timesteps()
    
    # Find optimal timestep for each stimulation type
    stimulation_types = ['constant', 'random', 'pulse']
    
    for stim_type in stimulation_types:
        stim_results = [r for r in results if r['stimulation_type'] == stim_type and r['steps_per_second'] > 0]
        if stim_results:
            best = max(stim_results, key=lambda x: x['steps_per_second'])
            print(f"\n🏆 BEST PERFORMANCE ({stim_type.upper()}): {best['dt_ms']}ms timestep")
            print(f"   Steps/sec: {best['steps_per_second']:.1f}")
            print(f"   ms/step: {best['avg_step_time']*1000:.3f}ms")
            print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
            print(f"   Firing Rate: {best['firing_rate']:.2f} Hz")
            print(f"   Mean Voltage: {best['voltage_stats']['mean_voltage']:.2f} mV")
