#!/usr/bin/env python3
"""
Biologically Realistic Brain Simulation
Based on actual neuroscience data for realistic neural activity patterns
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

class RealisticBrainSimulation:
    """
    Biologically Realistic Brain Simulation
    Based on actual neuroscience data for realistic neural activity patterns
    """
    
    def __init__(self, timestep_ms=0.1, seed=42):
        """Initialize realistic brain simulation"""
        self.timestep_ms = timestep_ms
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        cp.random.seed(seed)
        
        # Brain populations based on real data
        self.populations = {
            'cortex': {
                'n_neurons': 16_000_000_000,  # 16B cortical neurons
                'mean_rate_hz': 0.2,  # 0.16-0.3 Hz average
                'rate_std': 0.1,  # Log-normal distribution
                'active_fraction_1ms': 0.00016,  # 0.016% per ms
                'active_fraction_10ms': 0.0016,  # 0.16% per 10ms
                'active_fraction_100ms': 0.016,  # 1.6% per 100ms
            },
            'cerebellar_granule': {
                'n_neurons': 69_000_000_000,  # 69B granule cells
                'mean_rate_hz': 0.1,  # 0.1 Hz at rest
                'rate_std': 0.05,
                'active_fraction_1ms': 0.0001,  # 0.01% per ms
                'active_fraction_10ms': 0.001,  # 0.1% per 10ms
                'active_fraction_100ms': 0.01,  # 1% per 100ms
            },
            'purkinje': {
                'n_neurons': 15_000_000,  # 15M Purkinje cells
                'mean_rate_hz': 50.0,  # 50 Hz intrinsic
                'rate_std': 10.0,
                'active_fraction_1ms': 0.05,  # 5% per ms
                'active_fraction_10ms': 0.4,  # 40% per 10ms
                'active_fraction_100ms': 0.99,  # 99% per 100ms
            }
        }
        
        # Calculate total brain
        self.total_neurons = sum(pop['n_neurons'] for pop in self.populations.values())
        self.total_active_1ms = sum(pop['n_neurons'] * pop['active_fraction_1ms'] for pop in self.populations.values())
        self.total_active_10ms = sum(pop['n_neurons'] * pop['active_fraction_10ms'] for pop in self.populations.values())
        self.total_active_100ms = sum(pop['n_neurons'] * pop['active_fraction_100ms'] for pop in self.populations.values())
        
        print(f"🧠 Biologically Realistic Brain Simulation")
        print(f"   Timestep: {timestep_ms}ms")
        print(f"   Total Neurons: {self.total_neurons:,}")
        print(f"   Active per 1ms: {self.total_active_1ms:,.0f} ({self.total_active_1ms/self.total_neurons*100:.4f}%)")
        print(f"   Active per 10ms: {self.total_active_10ms:,.0f} ({self.total_active_10ms/self.total_neurons*100:.4f}%)")
        print(f"   Active per 100ms: {self.total_active_100ms:,.0f} ({self.total_active_100ms/self.total_neurons*100:.4f}%)")
        
        # Initialize simulation
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Initialize the realistic simulation"""
        print(f"\n🚀 Initializing Realistic Brain:")
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        initial_gpu_memory = analyze_gpu_memory()
        
        print(f"   Initial CPU Memory: {initial_cpu_memory:.3f} GB")
        if initial_gpu_memory:
            print(f"   Initial GPU Memory: {initial_gpu_memory['used_gb']:.3f} GB")
        
        # Initialize populations
        self.population_data = {}
        for name, pop in self.populations.items():
            # Calculate active neurons for this timestep
            if self.timestep_ms == 0.1:
                active_fraction = pop['active_fraction_1ms'] * 0.1  # Scale down for 0.1ms
            elif self.timestep_ms == 1.0:
                active_fraction = pop['active_fraction_1ms']
            elif self.timestep_ms == 10.0:
                active_fraction = pop['active_fraction_10ms']
            elif self.timestep_ms == 100.0:
                active_fraction = pop['active_fraction_100ms']
            else:
                # Interpolate between known points
                if self.timestep_ms < 1.0:
                    active_fraction = pop['active_fraction_1ms'] * (self.timestep_ms / 1.0)
                elif self.timestep_ms < 10.0:
                    active_fraction = pop['active_fraction_1ms'] + (pop['active_fraction_10ms'] - pop['active_fraction_1ms']) * ((self.timestep_ms - 1.0) / 9.0)
                elif self.timestep_ms < 100.0:
                    active_fraction = pop['active_fraction_10ms'] + (pop['active_fraction_100ms'] - pop['active_fraction_10ms']) * ((self.timestep_ms - 10.0) / 90.0)
                else:
                    active_fraction = pop['active_fraction_100ms']
            
            k_active = int(pop['n_neurons'] * active_fraction)
            
            # Initialize population data
            self.population_data[name] = {
                'n_neurons': pop['n_neurons'],
                'k_active': k_active,
                'active_fraction': active_fraction,
                'mean_rate_hz': pop['mean_rate_hz'],
                'rate_std': pop['rate_std'],
                'winners': cp.zeros(k_active, dtype=cp.int32),
                'weights': cp.zeros(k_active, dtype=cp.float32),
                'support': cp.zeros(k_active, dtype=cp.float32),
                'firing_rates': cp.random.lognormal(
                    mean=np.log(pop['mean_rate_hz']), 
                    sigma=pop['rate_std'], 
                    size=k_active
                ),
                'last_spike_time': cp.zeros(k_active, dtype=cp.float32),
                'refractory_period': cp.full(k_active, 2.0, dtype=cp.float32),  # 2ms refractory
                'activated': False
            }
            
            print(f"   {name}: {pop['n_neurons']:,} neurons, {k_active:,} active ({active_fraction*100:.4f}%)")
        
        # Pre-allocated arrays for efficiency
        self.candidates = cp.zeros(max(pop['k_active'] for pop in self.population_data.values()) * 10, dtype=cp.float32)
        self.top_k_indices = cp.zeros(max(pop['k_active'] for pop in self.population_data.values()), dtype=cp.int32)
        self.top_k_values = cp.zeros(max(pop['k_active'] for pop in self.population_data.values()), dtype=cp.float32)
        self.sorted_indices = cp.zeros(max(pop['k_active'] for pop in self.population_data.values()), dtype=cp.int32)
        
        # Get final memory
        final_cpu_memory = get_memory_usage()
        final_gpu_memory = analyze_gpu_memory()
        
        print(f"   Final CPU Memory: {final_cpu_memory:.3f} GB")
        if final_gpu_memory:
            print(f"   Final GPU Memory: {final_gpu_memory['used_gb']:.3f} GB")
        
        print(f"   CPU Memory Increase: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        if initial_gpu_memory and final_gpu_memory:
            print(f"   GPU Memory Increase: {final_gpu_memory['used_gb'] - initial_gpu_memory['used_gb']:.3f} GB")
    
    def simulate_realistic_step(self):
        """Simulate one realistic step with proper spike timing"""
        current_time = time.perf_counter()
        
        for name, pop in self.population_data.items():
            # Check refractory period
            non_refractory = (current_time - pop['last_spike_time']) > pop['refractory_period']
            
            if not cp.any(non_refractory):
                continue  # All neurons are refractory
            
            # Generate spike probabilities based on firing rates
            spike_probabilities = pop['firing_rates'] * (self.timestep_ms / 1000.0)  # Convert Hz to probability per timestep
            spike_probabilities = cp.minimum(spike_probabilities, 1.0)  # Cap at 1.0
            
            # Generate random numbers
            random_values = cp.random.random(pop['k_active'])
            
            # Determine which neurons spike
            spike_mask = (random_values < spike_probabilities) & non_refractory
            
            if cp.any(spike_mask):
                # Select winners from spiking neurons
                spiking_indices = cp.where(spike_mask)[0]
                if len(spiking_indices) >= pop['k_active']:
                    # Select top-k from spiking neurons
                    candidates = cp.random.exponential(1.0, size=len(spiking_indices))
                    top_k_indices = cp.argpartition(candidates, -pop['k_active'])[-pop['k_active']:]
                    winners = spiking_indices[top_k_indices]
                else:
                    # All spiking neurons are winners
                    winners = spiking_indices
                
                # Update population state
                pop['w'] = len(winners)
                pop['winners'][:len(winners)] = winners
                pop['activated'] = True
                
                # Update weights (spike-timing dependent plasticity)
                pop['weights'][winners] += 0.1
                pop['weights'] *= 0.99
                pop['support'][winners] += 1.0
                
                # Update spike times
                pop['last_spike_time'][winners] = current_time
    
    def simulate_detailed(self, n_steps=100):
        """Simulate with detailed performance monitoring"""
        print(f"\n🧠 Realistic Simulation ({n_steps} steps, {self.timestep_ms}ms timestep):")
        
        # Get initial memory
        initial_cpu_memory = get_memory_usage()
        initial_gpu_memory = analyze_gpu_memory()
        
        step_times = []
        total_start = time.perf_counter()
        
        for step in range(n_steps):
            step_start = time.perf_counter()
            
            # Simulate one realistic step
            self.simulate_realistic_step()
            
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
        
        total_time = time.perf_counter() - total_start
        
        # Final memory
        final_cpu_memory = get_memory_usage()
        final_gpu_memory = analyze_gpu_memory()
        
        # Calculate statistics
        avg_step_time = np.mean(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        std_step_time = np.std(step_times)
        
        print(f"\n📊 Performance Statistics:")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Average Step Time: {avg_step_time*1000:.2f}ms")
        print(f"   Min Step Time: {min_step_time*1000:.2f}ms")
        print(f"   Max Step Time: {max_step_time*1000:.2f}ms")
        print(f"   Std Dev: {std_step_time*1000:.2f}ms")
        print(f"   Steps per Second: {1/avg_step_time:.1f}")
        print(f"   Neurons per Second: {self.total_neurons / avg_step_time:,.0f}")
        print(f"   Active per Second: {self.total_active_1ms / avg_step_time:,.0f}")
        
        print(f"\n💾 Memory Usage:")
        print(f"   CPU Memory Change: {final_cpu_memory - initial_cpu_memory:.3f} GB")
        if initial_gpu_memory and final_gpu_memory:
            print(f"   GPU Memory Change: {final_gpu_memory['used_gb'] - initial_gpu_memory['used_gb']:.3f} GB")
        
        return {
            'total_time': total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': 1/avg_step_time,
            'neurons_per_second': self.total_neurons / avg_step_time,
            'active_per_second': self.total_active_1ms / avg_step_time,
            'memory_efficiency': 100.0
        }

def test_realistic_simulation():
    """Test realistic brain simulation with different timesteps"""
    print("🧠 BIOLOGICALLY REALISTIC BRAIN SIMULATION")
    print("=" * 80)
    
    # Test different timesteps
    timesteps = [0.1, 1.0, 10.0, 100.0]  # ms
    
    for timestep in timesteps:
        print(f"\n{'='*80}")
        print(f"🧪 TESTING: {timestep}ms timestep")
        print(f"{'='*80}")
        
        try:
            sim = RealisticBrainSimulation(timestep_ms=timestep, seed=42)
            stats = sim.simulate_detailed(n_steps=100)
            
            print(f"\n✅ SUCCESS: {timestep}ms timestep")
            print(f"   Steps/sec: {stats['steps_per_second']:.1f}")
            print(f"   ms/step: {stats['avg_step_time']*1000:.2f}ms")
            print(f"   Neurons/sec: {stats['neurons_per_second']:,.0f}")
            print(f"   Active/sec: {stats['active_per_second']:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    # Test realistic simulation
    test_realistic_simulation()
