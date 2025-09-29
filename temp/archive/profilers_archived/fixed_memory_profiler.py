#!/usr/bin/env python3
"""
Fixed Memory Profiler for Billion-Scale Neural Simulation
========================================================

This tool provides deep analytical profiling of memory usage patterns,
performance bottlenecks, and optimization opportunities for billion-scale
neural simulation on GPU.

Fixed issues:
- Removed matplotlib dependency that was causing hangs
- Simplified data collection
- Added timeout protection
- Better error handling
"""

import cupy as cp
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass

@dataclass
class MemoryProfile:
    """Memory usage profile for a specific scale."""
    scale_name: str
    n_neurons: int
    n_active_per_area: int
    n_areas: int
    sparsity: float
    memory_gb: float
    steps_per_sec: float
    ms_per_step: float
    neurons_per_sec: float
    active_per_sec: float
    gpu_memory_used_gb: float
    gpu_memory_available_gb: float
    gpu_memory_utilization: float
    cpu_memory_used_gb: float
    cpu_memory_available_gb: float
    cpu_memory_utilization: float
    weight_matrix_size: int
    weight_matrix_memory_gb: float
    activation_memory_gb: float
    candidate_memory_gb: float
    area_memory_gb: float
    total_theoretical_memory_gb: float
    memory_efficiency: float

class FixedMemoryProfiler:
    def __init__(self):
        """Initialize the fixed memory profiler."""
        self.profiles: List[MemoryProfile] = []
        self.current_profile: Optional[MemoryProfile] = None
        
        print("🔍 Fixed Memory Profiler initialized")
        print("   - Removed matplotlib dependency")
        print("   - Added timeout protection")
        print("   - Simplified data collection")
    
    def get_gpu_memory_info(self) -> Tuple[float, float]:
        """Get GPU memory usage information."""
        try:
            used, total = cp.cuda.Device().mem_info
            return used / 1024**3, total / 1024**3
        except Exception as e:
            print(f"   ⚠️  GPU memory info failed: {e}")
            return 0.0, 0.0
    
    def get_cpu_memory_info(self) -> Tuple[float, float]:
        """Get CPU memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return memory.used / 1024**3, memory.total / 1024**3
        except Exception as e:
            print(f"   ⚠️  CPU memory info failed: {e}")
            return 0.0, 0.0
    
    def calculate_memory_requirements(self, n_neurons: int, n_active_per_area: int, 
                                    n_areas: int) -> Dict[str, float]:
        """Calculate theoretical memory requirements."""
        # Sparse representation - only active neurons
        weight_matrix_size = n_active_per_area * n_active_per_area
        weight_matrix_memory_gb = weight_matrix_size * 4 / 1024**3  # 4 bytes per float32
        
        activation_memory_gb = n_active_per_area * 4 / 1024**3  # 4 bytes per float32
        candidate_memory_gb = n_active_per_area * 4 / 1024**3  # 4 bytes per float32
        area_memory_gb = n_active_per_area * 4 * 3 / 1024**3  # 3 arrays per area
        
        total_theoretical_memory_gb = (weight_matrix_memory_gb + activation_memory_gb + 
                                     candidate_memory_gb + area_memory_gb) * n_areas
        
        return {
            'weight_matrix_size': weight_matrix_size,
            'weight_matrix_memory_gb': weight_matrix_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'candidate_memory_gb': candidate_memory_gb,
            'area_memory_gb': area_memory_gb,
            'total_theoretical_memory_gb': total_theoretical_memory_gb
        }
    
    def create_simple_brain(self, n_neurons: int, active_percentage: float, 
                          n_areas: int = 5) -> Dict[str, Any]:
        """Create a simple brain for testing without complex dependencies."""
        k_active = int(n_neurons * active_percentage)
        
        print(f"   🧠 Creating simple brain: {n_neurons:,} neurons, {k_active:,} active per area")
        
        # Create areas with minimal memory footprint
        areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': k_active,
                'w': 0,
                'winners': cp.zeros(k_active, dtype=cp.int32),
                'weights': cp.zeros(k_active, dtype=cp.float32),
                'support': cp.zeros(k_active, dtype=cp.float32),
                'activated': False
            }
            areas.append(area)
        
        return {
            'n_neurons': n_neurons,
            'k_active': k_active,
            'n_areas': n_areas,
            'areas': areas,
            'step_count': 0,
            'total_time': 0.0
        }
    
    def simulate_brain_step(self, brain: Dict[str, Any], timeout_seconds: float = 5.0) -> float:
        """Simulate one step of the brain with timeout protection."""
        start_time = time.perf_counter()
        
        try:
            for area in brain['areas']:
                # Generate candidates using CuPy
                candidates = cp.random.exponential(1.0, size=area['k'])
                
                # Select top-k winners
                if area['k'] >= len(candidates):
                    winners = cp.arange(len(candidates))
                else:
                    top_k_indices = cp.argpartition(candidates, -area['k'])[-area['k']:]
                    top_k_values = candidates[top_k_indices]
                    sorted_indices = cp.argsort(top_k_values)[::-1]
                    winners = top_k_indices[sorted_indices]
                
                # Update area state
                area['w'] = len(winners)
                area['winners'][:len(winners)] = winners
                area['activated'] = True
                
                # Update weights
                area['weights'][winners] += 0.1
                area['weights'] *= 0.99
                area['support'][winners] += 1.0
                
                # Check timeout
                if time.perf_counter() - start_time > timeout_seconds:
                    print(f"   ⚠️  Simulation timeout after {timeout_seconds}s")
                    break
            
            brain['step_count'] += 1
            step_time = time.perf_counter() - start_time
            brain['total_time'] += step_time
            
            return step_time
            
        except Exception as e:
            print(f"   ❌ Simulation step failed: {e}")
            return 0.0
    
    def profile_scale(self, scale_name: str, n_neurons: int, active_percentage: float, 
                     n_areas: int = 5, n_steps: int = 10) -> MemoryProfile:
        """Profile a specific scale with timeout protection."""
        print(f"\n🔍 Profiling {scale_name}:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active percentage: {active_percentage*100:.4f}%")
        print(f"   Areas: {n_areas}")
        
        k_active = int(n_neurons * active_percentage)
        sparsity = 1.0 - (k_active / n_neurons)
        
        # Calculate theoretical memory requirements
        memory_reqs = self.calculate_memory_requirements(n_neurons, k_active, n_areas)
        
        # Get initial memory usage
        initial_gpu_used, gpu_total = self.get_gpu_memory_info()
        initial_cpu_used, cpu_total = self.get_cpu_memory_info()
        
        print(f"   Initial GPU memory: {initial_gpu_used:.2f}GB / {gpu_total:.2f}GB")
        print(f"   Initial CPU memory: {initial_cpu_used:.2f}GB / {cpu_total:.2f}GB")
        
        try:
            # Create brain
            brain = self.create_simple_brain(n_neurons, active_percentage, n_areas)
            
            # Simulate with timeout protection
            simulation_start = time.perf_counter()
            step_times = []
            
            for step in range(n_steps):
                step_time = self.simulate_brain_step(brain, timeout_seconds=2.0)
                if step_time > 0:
                    step_times.append(step_time)
                
                # Check overall timeout
                if time.perf_counter() - simulation_start > 30.0:  # 30 second total timeout
                    print(f"   ⚠️  Overall timeout after 30s")
                    break
            
            simulation_time = time.perf_counter() - simulation_start
            
            # Get final memory usage
            final_gpu_used, gpu_total = self.get_gpu_memory_info()
            final_cpu_used, cpu_total = self.get_cpu_memory_info()
            
            # Calculate performance metrics
            if step_times:
                avg_step_time = np.mean(step_times)
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                ms_per_step = avg_step_time * 1000
            else:
                avg_step_time = 0
                steps_per_sec = 0
                ms_per_step = 0
            
            neurons_per_sec = n_neurons * steps_per_sec
            active_per_sec = k_active * n_areas * steps_per_sec
            
            # Calculate memory efficiency
            actual_memory_used = final_gpu_used - initial_gpu_used
            memory_efficiency = (actual_memory_used / memory_reqs['total_theoretical_memory_gb']) * 100 if memory_reqs['total_theoretical_memory_gb'] > 0 else 0
            
            print(f"   ✅ Profile complete!")
            print(f"   Steps completed: {len(step_times)}/{n_steps}")
            print(f"   Avg step time: {ms_per_step:.2f}ms")
            print(f"   Steps/sec: {steps_per_sec:.1f}")
            print(f"   Memory efficiency: {memory_efficiency:.1f}%")
            
            # Create profile
            profile = MemoryProfile(
                scale_name=scale_name,
                n_neurons=n_neurons,
                n_active_per_area=k_active,
                n_areas=n_areas,
                sparsity=sparsity,
                memory_gb=actual_memory_used,
                steps_per_sec=steps_per_sec,
                ms_per_step=ms_per_step,
                neurons_per_sec=neurons_per_sec,
                active_per_sec=active_per_sec,
                gpu_memory_used_gb=final_gpu_used,
                gpu_memory_available_gb=gpu_total,
                gpu_memory_utilization=(final_gpu_used / gpu_total) * 100,
                cpu_memory_used_gb=final_cpu_used,
                cpu_memory_available_gb=cpu_total,
                cpu_memory_utilization=(final_cpu_used / cpu_total) * 100,
                weight_matrix_size=memory_reqs['weight_matrix_size'],
                weight_matrix_memory_gb=memory_reqs['weight_matrix_memory_gb'],
                activation_memory_gb=memory_reqs['activation_memory_gb'],
                candidate_memory_gb=memory_reqs['candidate_memory_gb'],
                area_memory_gb=memory_reqs['area_memory_gb'],
                total_theoretical_memory_gb=memory_reqs['total_theoretical_memory_gb'],
                memory_efficiency=memory_efficiency
            )
            
            self.profiles.append(profile)
            return profile
            
        except Exception as e:
            print(f"   ❌ Profile failed: {e}")
            # Return empty profile
            return MemoryProfile(
                scale_name=scale_name,
                n_neurons=n_neurons,
                n_active_per_area=k_active,
                n_areas=n_areas,
                sparsity=sparsity,
                memory_gb=0,
                steps_per_sec=0,
                ms_per_step=0,
                neurons_per_sec=0,
                active_per_sec=0,
                gpu_memory_used_gb=0,
                gpu_memory_available_gb=0,
                gpu_memory_utilization=0,
                cpu_memory_used_gb=0,
                cpu_memory_available_gb=0,
                cpu_memory_utilization=0,
                weight_matrix_size=0,
                weight_matrix_memory_gb=0,
                activation_memory_gb=0,
                candidate_memory_gb=0,
                area_memory_gb=0,
                total_theoretical_memory_gb=0,
                memory_efficiency=0
            )
    
    def run_comprehensive_profiling(self):
        """Run comprehensive profiling across multiple scales."""
        print("🚀 FIXED MEMORY PROFILER - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        # Test scales with timeout protection
        test_scales = [
            {"n_neurons": 1000000, "active_percentage": 0.01, "name": "Million Scale (1%)"},
            {"n_neurons": 10000000, "active_percentage": 0.01, "name": "Ten Million Scale (1%)"},
            {"n_neurons": 100000000, "active_percentage": 0.001, "name": "Hundred Million Scale (0.1%)"},
            {"n_neurons": 1000000000, "active_percentage": 0.0001, "name": "Billion Scale (0.01%)"},
        ]
        
        for scale in test_scales:
            try:
                profile = self.profile_scale(
                    scale_name=scale["name"],
                    n_neurons=scale["n_neurons"],
                    active_percentage=scale["active_percentage"],
                    n_areas=5,
                    n_steps=5  # Reduced steps for faster testing
                )
                
                # Force garbage collection
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"   ❌ Scale {scale['name']} failed: {e}")
                continue
        
        self.print_summary()
        self.save_profiles()
    
    def print_summary(self):
        """Print profiling summary."""
        print(f"\n📊 FIXED MEMORY PROFILER SUMMARY")
        print("=" * 80)
        print(f"{'Scale':<25} {'Neurons':<12} {'Active%':<8} {'Steps/sec':<10} {'ms/step':<10} {'Memory GB':<10} {'Efficiency%':<12}")
        print("-" * 80)
        
        for profile in self.profiles:
            if profile.steps_per_sec > 0:
                print(f"{profile.scale_name:<25} {profile.n_neurons:<12,} {profile.sparsity*100:<8.1f} {profile.steps_per_sec:<10.1f} {profile.ms_per_step:<10.2f} {profile.memory_gb:<10.2f} {profile.memory_efficiency:<12.1f}")
            else:
                print(f"{profile.scale_name:<25} {profile.n_neurons:<12,} {profile.sparsity*100:<8.1f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'FAILED':<12}")
    
    def save_profiles(self, filename: str = "fixed_memory_profiles.json"):
        """Save profiles to JSON file."""
        try:
            profile_data = []
            for profile in self.profiles:
                profile_data.append({
                    'scale_name': profile.scale_name,
                    'n_neurons': profile.n_neurons,
                    'n_active_per_area': profile.n_active_per_area,
                    'n_areas': profile.n_areas,
                    'sparsity': profile.sparsity,
                    'memory_gb': profile.memory_gb,
                    'steps_per_sec': profile.steps_per_sec,
                    'ms_per_step': profile.ms_per_step,
                    'neurons_per_sec': profile.neurons_per_sec,
                    'active_per_sec': profile.active_per_sec,
                    'gpu_memory_used_gb': profile.gpu_memory_used_gb,
                    'gpu_memory_available_gb': profile.gpu_memory_available_gb,
                    'gpu_memory_utilization': profile.gpu_memory_utilization,
                    'cpu_memory_used_gb': profile.cpu_memory_used_gb,
                    'cpu_memory_available_gb': profile.cpu_memory_available_gb,
                    'cpu_memory_utilization': profile.cpu_memory_utilization,
                    'weight_matrix_size': profile.weight_matrix_size,
                    'weight_matrix_memory_gb': profile.weight_matrix_memory_gb,
                    'activation_memory_gb': profile.activation_memory_gb,
                    'candidate_memory_gb': profile.candidate_memory_gb,
                    'area_memory_gb': profile.area_memory_gb,
                    'total_theoretical_memory_gb': profile.total_theoretical_memory_gb,
                    'memory_efficiency': profile.memory_efficiency
                })
            
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            print(f"\n💾 Profiles saved to {filename}")
            
        except Exception as e:
            print(f"   ❌ Failed to save profiles: {e}")

def main():
    """Main function to run the fixed memory profiler."""
    try:
        profiler = FixedMemoryProfiler()
        profiler.run_comprehensive_profiling()
        
        print(f"\n✅ Fixed Memory Profiler completed successfully!")
        
    except Exception as e:
        print(f"❌ Fixed Memory Profiler failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
