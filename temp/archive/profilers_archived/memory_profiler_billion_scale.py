#!/usr/bin/env python3
"""
Memory Profiler for Billion-Scale Neural Simulation
==================================================

This tool provides deep analytical profiling of memory usage patterns,
performance bottlenecks, and optimization opportunities for billion-scale
neural simulation on GPU.
"""

import cupy as cp
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
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

class MemoryProfiler:
    def __init__(self):
        """Initialize the memory profiler."""
        self.profiles = []
        self.gpu_memory_info = self._get_gpu_memory_info()
        
    def _get_gpu_memory_info(self) -> Dict:
        """Get GPU memory information."""
        try:
            mempool = cp.get_default_memory_pool()
            return {
                'total_gb': mempool.total_bytes() / 1024**3,
                'used_gb': mempool.used_bytes() / 1024**3,
                'available_gb': (mempool.total_bytes() - mempool.used_bytes()) / 1024**3
            }
        except:
            return {'total_gb': 16.0, 'used_gb': 0.0, 'available_gb': 16.0}
    
    def _get_cpu_memory_info(self) -> Dict:
        """Get CPU memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'utilization': memory.percent
        }
    
    def _calculate_memory_requirements(self, n_neurons: int, n_areas: int, 
                                     active_percent: float, sparsity: float) -> Dict:
        """Calculate detailed memory requirements."""
        n_active_per_area = int(n_neurons * active_percent)
        
        # Weight matrix memory (sparse)
        n_weights = int(n_active_per_area * n_active_per_area * sparsity)
        weight_memory_gb = n_weights * 4 / 1024**3  # 4 bytes per float32
        
        # Sparse matrix overhead (indices, etc.)
        sparse_overhead = 4.0  # 4x overhead for ultra-sparse
        weight_memory_gb *= sparse_overhead
        
        # Other arrays memory
        activation_memory_gb = n_active_per_area * 4 / 1024**3
        candidate_memory_gb = n_active_per_area * 4 / 1024**3
        area_memory_gb = n_active_per_area * 4 / 1024**3
        
        # Total per area
        per_area_memory_gb = weight_memory_gb + activation_memory_gb + candidate_memory_gb + area_memory_gb
        
        # Total memory
        total_memory_gb = per_area_memory_gb * n_areas
        
        return {
            'n_active_per_area': n_active_per_area,
            'n_weights': n_weights,
            'weight_memory_gb': weight_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'candidate_memory_gb': candidate_memory_gb,
            'area_memory_gb': area_memory_gb,
            'per_area_memory_gb': per_area_memory_gb,
            'total_memory_gb': total_memory_gb,
            'memory_efficiency': total_memory_gb / (n_neurons * 4 / 1024**3) if n_neurons > 0 else 0
        }
    
    def profile_scale(self, n_neurons: int, n_areas: int = 5, 
                     active_percent: float = 0.001, sparsity: float = 0.0001,
                     n_steps: int = 10) -> MemoryProfile:
        """Profile memory usage for a specific scale."""
        print(f"\n🔍 Profiling {n_neurons:,} neurons...")
        
        # Calculate memory requirements
        memory_req = self._calculate_memory_requirements(n_neurons, n_areas, active_percent, sparsity)
        
        # Get initial memory state
        initial_gpu_memory = self._get_gpu_memory_info()
        initial_cpu_memory = self._get_cpu_memory_info()
        
        try:
            # Initialize arrays
            areas = []
            weights = []
            activations = []
            candidates = []
            
            for i in range(n_areas):
                # Create area
                area = cp.zeros(memory_req['n_active_per_area'], dtype=cp.float32)
                areas.append(area)
                
                # Create sparse weight matrix
                if memory_req['n_weights'] > 0:
                    row_indices = cp.random.randint(0, memory_req['n_active_per_area'], 
                                                  size=memory_req['n_weights'], dtype=cp.int32)
                    col_indices = cp.random.randint(0, memory_req['n_active_per_area'], 
                                                  size=memory_req['n_weights'], dtype=cp.int32)
                    values = cp.random.exponential(1.0, size=memory_req['n_weights'], dtype=cp.float32)
                    
                    coo_matrix = cp.sparse.coo_matrix(
                        (values, (row_indices, col_indices)),
                        shape=(memory_req['n_active_per_area'], memory_req['n_active_per_area']),
                        dtype=cp.float32
                    )
                    weights_sparse = coo_matrix.tocsr()
                    weights.append(weights_sparse)
                    del coo_matrix
                else:
                    weights.append(cp.sparse.csr_matrix(
                        (memory_req['n_active_per_area'], memory_req['n_active_per_area']),
                        dtype=cp.float32
                    ))
                
                # Create other arrays
                activations.append(cp.zeros(memory_req['n_active_per_area'], dtype=cp.float32))
                candidates.append(cp.zeros(memory_req['n_active_per_area'], dtype=cp.float32))
            
            # Get memory state after allocation
            allocated_gpu_memory = self._get_gpu_memory_info()
            allocated_cpu_memory = self._get_cpu_memory_info()
            
            # Run benchmark
            times = []
            for step in range(n_steps):
                step_start = time.time()
                
                for area_idx in range(n_areas):
                    area = areas[area_idx]
                    weights = weights[area_idx]
                    activations = activations[area_idx]
                    candidates = candidates[area_idx]
                    
                    # Generate random candidates
                    candidates[:] = cp.random.exponential(1.0, size=len(candidates))
                    
                    # Compute activations
                    activations[:] = weights.dot(area)
                    
                    # Apply threshold
                    threshold = cp.percentile(activations, 90)
                    area[:] = cp.where(activations > threshold, candidates, 0.0)
                
                step_time = time.time() - step_start
                times.append(step_time)
            
            # Calculate performance metrics
            avg_time = np.mean(times)
            steps_per_sec = 1.0 / avg_time
            neurons_per_sec = n_neurons * steps_per_sec
            active_per_sec = memory_req['n_active_per_area'] * n_areas * steps_per_sec
            
            # Get final memory state
            final_gpu_memory = self._get_gpu_memory_info()
            final_cpu_memory = self._get_cpu_memory_info()
            
            # Create profile
            profile = MemoryProfile(
                scale_name=f"{n_neurons:,} neurons",
                n_neurons=n_neurons,
                n_active_per_area=memory_req['n_active_per_area'],
                n_areas=n_areas,
                sparsity=sparsity,
                memory_gb=memory_req['total_memory_gb'],
                steps_per_sec=steps_per_sec,
                ms_per_step=avg_time * 1000,
                neurons_per_sec=neurons_per_sec,
                active_per_sec=active_per_sec,
                gpu_memory_used_gb=final_gpu_memory['used_gb'],
                gpu_memory_available_gb=final_gpu_memory['available_gb'],
                gpu_memory_utilization=(final_gpu_memory['used_gb'] / final_gpu_memory['total_gb']) * 100,
                cpu_memory_used_gb=final_cpu_memory['used_gb'],
                cpu_memory_available_gb=final_cpu_memory['available_gb'],
                cpu_memory_utilization=final_cpu_memory['utilization'],
                weight_matrix_size=memory_req['n_weights'],
                weight_matrix_memory_gb=memory_req['weight_memory_gb'],
                activation_memory_gb=memory_req['activation_memory_gb'],
                candidate_memory_gb=memory_req['candidate_memory_gb'],
                area_memory_gb=memory_req['area_memory_gb'],
                total_theoretical_memory_gb=memory_req['total_memory_gb'],
                memory_efficiency=memory_req['memory_efficiency']
            )
            
            # Cleanup
            del areas, weights, activations, candidates
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            print(f"   ✅ Profiling complete!")
            print(f"   Memory: {profile.memory_gb:.3f} GB")
            print(f"   Performance: {profile.steps_per_sec:.1f} steps/sec")
            print(f"   GPU Utilization: {profile.gpu_memory_utilization:.1f}%")
            
            return profile
            
        except Exception as e:
            print(f"   ❌ Profiling failed: {e}")
            return None
    
    def run_comprehensive_profiling(self):
        """Run comprehensive profiling across multiple scales."""
        print("🔍 COMPREHENSIVE MEMORY PROFILING")
        print("=" * 60)
        
        # Test scales
        test_scales = [
            (1_000_000, "Million Scale"),
            (10_000_000, "Ten Million Scale"),
            (100_000_000, "Hundred Million Scale"),
            (1_000_000_000, "BILLION SCALE")
        ]
        
        # Test different sparsity levels
        sparsity_levels = [0.001, 0.0001, 0.00001]
        
        for n_neurons, scale_name in test_scales:
            print(f"\n🧪 Testing {scale_name}:")
            
            for sparsity in sparsity_levels:
                print(f"   Sparsity: {sparsity*100:.3f}%")
                
                try:
                    profile = self.profile_scale(n_neurons, sparsity=sparsity, n_steps=5)
                    if profile:
                        self.profiles.append(profile)
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
        
        # Analyze results
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze profiling results and provide insights."""
        print(f"\n📊 MEMORY PROFILING ANALYSIS")
        print("=" * 80)
        
        if not self.profiles:
            print("No profiles to analyze.")
            return
        
        # Group by scale
        scales = {}
        for profile in self.profiles:
            if profile.n_neurons not in scales:
                scales[profile.n_neurons] = []
            scales[profile.n_neurons].append(profile)
        
        # Print detailed analysis
        for n_neurons in sorted(scales.keys()):
            profiles = scales[n_neurons]
            print(f"\n🔍 Scale: {n_neurons:,} neurons")
            print("-" * 50)
            
            for profile in profiles:
                print(f"Sparsity: {profile.sparsity*100:.3f}%")
                print(f"  Memory: {profile.memory_gb:.3f} GB")
                print(f"  Performance: {profile.steps_per_sec:.1f} steps/sec")
                print(f"  GPU Utilization: {profile.gpu_memory_utilization:.1f}%")
                print(f"  Memory Efficiency: {profile.memory_efficiency:.3f}")
                print()
        
        # Find optimal configurations
        self._find_optimal_configurations()
    
    def _find_optimal_configurations(self):
        """Find optimal configurations for each scale."""
        print(f"\n🎯 OPTIMAL CONFIGURATIONS")
        print("=" * 60)
        
        # Group by scale
        scales = {}
        for profile in self.profiles:
            if profile.n_neurons not in scales:
                scales[profile.n_neurons] = []
            scales[profile.n_neurons].append(profile)
        
        for n_neurons in sorted(scales.keys()):
            profiles = scales[n_neurons]
            
            # Find best performance
            best_performance = max(profiles, key=lambda p: p.steps_per_sec)
            
            # Find most memory efficient
            most_efficient = min(profiles, key=lambda p: p.memory_gb)
            
            # Find best balance
            best_balance = max(profiles, key=lambda p: p.steps_per_sec / p.memory_gb)
            
            print(f"\nScale: {n_neurons:,} neurons")
            print(f"  Best Performance: {best_performance.sparsity*100:.3f}% sparsity")
            print(f"    {best_performance.steps_per_sec:.1f} steps/sec, {best_performance.memory_gb:.3f} GB")
            print(f"  Most Memory Efficient: {most_efficient.sparsity*100:.3f}% sparsity")
            print(f"    {most_efficient.steps_per_sec:.1f} steps/sec, {most_efficient.memory_gb:.3f} GB")
            print(f"  Best Balance: {best_balance.sparsity*100:.3f}% sparsity")
            print(f"    {best_balance.steps_per_sec:.1f} steps/sec, {best_balance.memory_gb:.3f} GB")
    
    def generate_memory_report(self):
        """Generate a comprehensive memory report."""
        print(f"\n📋 MEMORY REPORT")
        print("=" * 60)
        
        if not self.profiles:
            print("No profiles available for report.")
            return
        
        # Calculate statistics
        total_memory = sum(p.memory_gb for p in self.profiles)
        avg_performance = np.mean([p.steps_per_sec for p in self.profiles])
        max_performance = max(p.steps_per_sec for p in self.profiles)
        
        print(f"Total Memory Tested: {total_memory:.3f} GB")
        print(f"Average Performance: {avg_performance:.1f} steps/sec")
        print(f"Maximum Performance: {max_performance:.1f} steps/sec")
        
        # Memory efficiency analysis
        memory_efficiencies = [p.memory_efficiency for p in self.profiles]
        print(f"Memory Efficiency Range: {min(memory_efficiencies):.3f} - {max(memory_efficiencies):.3f}")
        
        # GPU utilization analysis
        gpu_utilizations = [p.gpu_memory_utilization for p in self.profiles]
        print(f"GPU Utilization Range: {min(gpu_utilizations):.1f}% - {max(gpu_utilizations):.1f}%")

def main():
    """Main function to run memory profiling."""
    profiler = MemoryProfiler()
    profiler.run_comprehensive_profiling()
    profiler.generate_memory_report()

if __name__ == "__main__":
    main()

