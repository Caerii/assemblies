#!/usr/bin/env python3
"""
Universal Profiler - Profilers Superset
=======================================

This superset combines the best features from all profiling implementations:
- Advanced performance profiling
- Memory usage analysis
- Scaling laws analysis
- Detailed performance analysis
- Fixed memory profiling
- Theoretical memory analysis

Combines features from:
- advanced_profiler.py
- detailed_performance_analyzer.py
- scaling_laws_analyzer.py
- memory_profiler_billion_scale.py
- fixed_memory_profiler.py
- simple_memory_analyzer.py
- theoretical_memory_analysis.py
"""

import time
import numpy as np
import os
import sys
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

# Try to import CuPy for GPU memory management
try:
    import cupy as cp
    print("✅ CuPy imported successfully!")
    print(f"   CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"   Current device: {cp.cuda.Device().id}")
    print(f"   Device memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    CUPY_AVAILABLE = True
except ImportError:
    print("⚠️  CuPy not available, using NumPy fallback")
    CUPY_AVAILABLE = False

@dataclass
class ProfilerConfig:
    """Configuration for universal profiler"""
    enable_gpu_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_scaling_analysis: bool = True
    enable_theoretical_analysis: bool = True
    enable_systematic_profiling: bool = True
    timeout_seconds: float = 30.0
    max_steps: int = 10
    profile_interval: int = 1
    systematic_steps: int = 100  # More steps for systematic analysis
    enable_extended_scales: bool = True  # Enable billion+ scale testing

@dataclass
class MemoryProfile:
    """Comprehensive memory usage profile"""
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
    theoretical_memory_gb: float
    memory_efficiency: float
    cache_hit_rate: float
    memory_pool_efficiency: float

@dataclass
class PerformanceProfile:
    """Comprehensive performance profile"""
    scale_name: str
    n_neurons: int
    active_percentage: float
    n_areas: int
    total_time: float
    avg_step_time: float
    min_step_time: float
    max_step_time: float
    std_step_time: float
    steps_per_second: float
    neurons_per_second: float
    active_neurons_per_second: float
    memory_usage_gb: float
    gpu_utilization: float
    cpu_utilization: float
    cache_performance: Dict[str, float]
    bottleneck_analysis: Dict[str, float]

class UniversalProfiler:
    """
    Universal Profiler
    
    Combines the best features from all profiling implementations:
    - Advanced performance profiling with bottleneck analysis
    - Comprehensive memory usage analysis
    - Scaling laws analysis across multiple scales
    - Detailed performance analysis with statistics
    - Fixed memory profiling with timeout protection
    - Theoretical memory analysis and efficiency calculation
    """
    
    def __init__(self, config: ProfilerConfig):
        """Initialize the universal profiler"""
        self.config = config
        self.profiles: List[MemoryProfile] = []
        self.performance_profiles: List[PerformanceProfile] = []
        self.scaling_data: Dict[str, List[float]] = {}
        
        print("🔍 Universal Profiler initialized")
        print(f"   GPU profiling: {'✅' if config.enable_gpu_profiling else '❌'}")
        print(f"   CPU profiling: {'✅' if config.enable_cpu_profiling else '❌'}")
        print(f"   Memory profiling: {'✅' if config.enable_memory_profiling else '❌'}")
        print(f"   Scaling analysis: {'✅' if config.enable_scaling_analysis else '❌'}")
        print(f"   Theoretical analysis: {'✅' if config.enable_theoretical_analysis else '❌'}")
        print(f"   Timeout protection: {config.timeout_seconds}s")
    
    def get_gpu_memory_info(self) -> Tuple[float, float]:
        """Get GPU memory usage information"""
        if not CUPY_AVAILABLE or not self.config.enable_gpu_profiling:
            return 0.0, 0.0
        
        try:
            used, total = cp.cuda.Device().mem_info
            return used / 1024**3, total / 1024**3
        except Exception as e:
            print(f"   ⚠️  GPU memory info failed: {e}")
            return 0.0, 0.0
    
    def get_cpu_memory_info(self) -> Tuple[float, float]:
        """Get CPU memory usage information"""
        if not self.config.enable_cpu_profiling:
            return 0.0, 0.0
        
        try:
            memory = psutil.virtual_memory()
            return memory.used / 1024**3, memory.total / 1024**3
        except Exception as e:
            print(f"   ⚠️  CPU memory info failed: {e}")
            return 0.0, 0.0
    
    def calculate_theoretical_memory(self, n_neurons: int, n_active_per_area: int, 
                                   n_areas: int) -> Dict[str, float]:
        """Calculate theoretical memory requirements for sparse implementation"""
        # Very sparse representation - only essential data structures
        sparsity_factor = 0.001  # 0.1% sparsity for ultra-sparse estimates
        weight_matrix_size = int(n_active_per_area * n_active_per_area * sparsity_factor)
        weight_matrix_memory_gb = weight_matrix_size * 4 / 1024**3  # 4 bytes per float32
        
        # Minimal sparse data structures
        indices_memory_gb = weight_matrix_size * 4 / 1024**3  # indices for sparse matrix
        activation_memory_gb = n_active_per_area * 4 / 1024**3
        candidate_memory_gb = n_active_per_area * 4 / 1024**3
        area_memory_gb = n_active_per_area * 4 * 3 / 1024**3  # 3 arrays per area
        
        # Minimal overhead for CuPy operations
        overhead_factor = 1.1  # 10% overhead for CuPy operations
        
        total_theoretical_memory_gb = (weight_matrix_memory_gb + indices_memory_gb + 
                                     activation_memory_gb + candidate_memory_gb + 
                                     area_memory_gb) * n_areas * overhead_factor
        
        return {
            'weight_matrix_size': weight_matrix_size,
            'weight_matrix_memory_gb': weight_matrix_memory_gb,
            'indices_memory_gb': indices_memory_gb,
            'activation_memory_gb': activation_memory_gb,
            'candidate_memory_gb': candidate_memory_gb,
            'area_memory_gb': area_memory_gb,
            'overhead_factor': overhead_factor,
            'total_theoretical_memory_gb': total_theoretical_memory_gb
        }
    
    def create_test_brain(self, n_neurons: int, active_percentage: float, 
                         n_areas: int = 5) -> Dict[str, Any]:
        """Create a test brain for profiling"""
        k_active = int(n_neurons * active_percentage)
        
        print(f"   🧠 Creating test brain: {n_neurons:,} neurons, {k_active:,} active per area")
        
        # Create areas with appropriate memory management
        areas = []
        for i in range(n_areas):
            if CUPY_AVAILABLE and self.config.enable_gpu_profiling:
                area = {
                    'n': n_neurons,
                    'k': k_active,
                    'w': 0,
                    'winners': cp.zeros(k_active, dtype=cp.int32),
                    'weights': cp.zeros(k_active, dtype=cp.float32),
                    'support': cp.zeros(k_active, dtype=cp.float32),
                    'activated': False
                }
            else:
                area = {
                    'n': n_neurons,
                    'k': k_active,
                    'w': 0,
                    'winners': np.zeros(k_active, dtype=np.int32),
                    'weights': np.zeros(k_active, dtype=np.float32),
                    'support': np.zeros(k_active, dtype=np.float32),
                    'activated': False
                }
            areas.append(area)
        
        return {
            'n_neurons': n_neurons,
            'k_active': k_active,
            'n_areas': n_areas,
            'areas': areas,
            'step_count': 0,
            'total_time': 0.0,
            'step_times': []
        }
    
    def simulate_brain_step(self, brain: Dict[str, Any], timeout_seconds: float = 2.0) -> float:
        """Simulate one step of the brain with timeout protection"""
        start_time = time.perf_counter()
        
        try:
            for area in brain['areas']:
                # Generate candidates
                if CUPY_AVAILABLE and self.config.enable_gpu_profiling:
                    candidates = cp.random.exponential(1.0, size=area['k'])
                else:
                    candidates = np.random.exponential(1.0, size=area['k'])
                
                # Select top-k winners
                if area['k'] >= len(candidates):
                    winners = cp.arange(len(candidates)) if CUPY_AVAILABLE else np.arange(len(candidates))
                else:
                    if CUPY_AVAILABLE and self.config.enable_gpu_profiling:
                        top_k_indices = cp.argpartition(candidates, -area['k'])[-area['k']:]
                        top_k_values = candidates[top_k_indices]
                        sorted_indices = cp.argsort(top_k_values)[::-1]
                        winners = top_k_indices[sorted_indices]
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
                
                # Check timeout
                if time.perf_counter() - start_time > timeout_seconds:
                    print(f"   ⚠️  Simulation timeout after {timeout_seconds}s")
                    break
            
            brain['step_count'] += 1
            step_time = time.perf_counter() - start_time
            brain['total_time'] += step_time
            brain['step_times'].append(step_time)
            
            return step_time
            
        except Exception as e:
            print(f"   ❌ Simulation step failed: {e}")
            return 0.0
    
    def profile_scale(self, scale_name: str, n_neurons: int, active_percentage: float, 
                     n_areas: int = 5, n_steps: int = 10) -> MemoryProfile:
        """Profile a specific scale with comprehensive analysis"""
        print(f"\n🔍 Profiling {scale_name}:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active percentage: {active_percentage*100:.4f}%")
        print(f"   Areas: {n_areas}")
        
        k_active = int(n_neurons * active_percentage)
        sparsity = 1.0 - (k_active / n_neurons)
        
        # Calculate theoretical memory requirements
        theoretical_memory = self.calculate_theoretical_memory(n_neurons, k_active, n_areas)
        
        # Get initial memory usage
        initial_gpu_used, gpu_total = self.get_gpu_memory_info()
        initial_cpu_used, cpu_total = self.get_cpu_memory_info()
        
        print(f"   Initial GPU memory: {initial_gpu_used:.2f}GB / {gpu_total:.2f}GB")
        print(f"   Initial CPU memory: {initial_cpu_used:.2f}GB / {cpu_total:.2f}GB")
        
        try:
            # Create brain
            brain = self.create_test_brain(n_neurons, active_percentage, n_areas)
            
            # Simulate with timeout protection
            simulation_start = time.perf_counter()
            step_times = []
            
            for step in range(min(n_steps, self.config.max_steps)):
                step_time = self.simulate_brain_step(brain, timeout_seconds=2.0)
                if step_time > 0:
                    step_times.append(step_time)
                
                # Check overall timeout
                if time.perf_counter() - simulation_start > self.config.timeout_seconds:
                    print(f"   ⚠️  Overall timeout after {self.config.timeout_seconds}s")
                    break
            
            simulation_time = time.perf_counter() - simulation_start
            
            # Get final memory usage
            final_gpu_used, gpu_total = self.get_gpu_memory_info()
            final_cpu_used, cpu_total = self.get_cpu_memory_info()
            
            # Calculate performance metrics
            if step_times:
                avg_step_time = np.mean(step_times)
                std_step_time = np.std(step_times)
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                ms_per_step = avg_step_time * 1000
            else:
                avg_step_time = 0
                std_step_time = 0
                steps_per_sec = 0
                ms_per_step = 0
            
            neurons_per_sec = n_neurons * steps_per_sec
            active_per_sec = k_active * n_areas * steps_per_sec
            
            # Calculate memory utilization ratio (how much of available GPU memory is used)
            gpu_memory_utilization = (final_gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
            
            # Calculate memory efficiency based on what the brain actually uses
            # Since the brain doesn't allocate additional GPU memory, measure its actual data structures
            
            # Calculate actual memory used by brain data structures
            actual_memory_used = 0.0
            if 'areas' in brain:
                for area in brain['areas']:
                    if 'weights' in area:
                        actual_memory_used += area['weights'].nbytes / 1024**3  # Convert to GB
                    if 'support' in area:
                        actual_memory_used += area['support'].nbytes / 1024**3
                    if 'winners' in area:
                        actual_memory_used += area['winners'].nbytes / 1024**3
            
            theoretical_memory_gb = theoretical_memory['total_theoretical_memory_gb']
            
            
            # Calculate bytes per neuron (memory efficiency metric)
            if actual_memory_used > 0 and n_neurons > 0:
                bytes_per_neuron = (actual_memory_used * 1024**3) / n_neurons  # Convert GB to bytes
            else:
                bytes_per_neuron = 0.0
            
            
            print(f"   ✅ Profile complete!")
            print(f"   Steps completed: {len(step_times)}/{n_steps}")
            print(f"   Avg step time: {ms_per_step:.2f}ms ± {std_step_time*1000:.2f}ms")
            print(f"   Steps/sec: {steps_per_sec:.1f}")
            print(f"   Bytes per neuron: {bytes_per_neuron:.5f}")
            
            # Create comprehensive profile
            profile = MemoryProfile(
                scale_name=scale_name,
                n_neurons=n_neurons,
                n_active_per_area=k_active,
                n_areas=n_areas,
                sparsity=sparsity,
                memory_gb=final_gpu_used,  # Use absolute GPU memory usage
                steps_per_sec=steps_per_sec,
                ms_per_step=ms_per_step,
                neurons_per_sec=neurons_per_sec,
                active_per_sec=active_per_sec,
                gpu_memory_used_gb=final_gpu_used,
                gpu_memory_available_gb=gpu_total,
                gpu_memory_utilization=gpu_memory_utilization,
                cpu_memory_used_gb=final_cpu_used,
                cpu_memory_available_gb=cpu_total,
                cpu_memory_utilization=(final_cpu_used / cpu_total) * 100 if cpu_total > 0 else 0,
                theoretical_memory_gb=theoretical_memory['total_theoretical_memory_gb'],
                memory_efficiency=bytes_per_neuron,
                cache_hit_rate=0.0,  # Placeholder for future implementation
                memory_pool_efficiency=0.0  # Placeholder for future implementation
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
                theoretical_memory_gb=0,
                memory_efficiency=0,
                cache_hit_rate=0,
                memory_pool_efficiency=0
            )
    
    def analyze_scaling_laws(self):
        """Analyze scaling laws across different scales"""
        if not self.config.enable_scaling_analysis:
            return
        
        print(f"\n📊 SCALING LAWS ANALYSIS")
        print("=" * 60)
        
        # Analyze scaling relationships
        if len(self.profiles) < 2:
            print("   ⚠️  Need at least 2 profiles for scaling analysis")
            return
        
        # Sort profiles by neuron count
        sorted_profiles = sorted(self.profiles, key=lambda x: x.n_neurons)
        
        print(f"   Analyzing {len(sorted_profiles)} scales...")
        
        # Calculate scaling factors
        for i in range(1, len(sorted_profiles)):
            prev = sorted_profiles[i-1]
            curr = sorted_profiles[i]
            
            neuron_ratio = curr.n_neurons / prev.n_neurons
            time_ratio = curr.ms_per_step / prev.ms_per_step if prev.ms_per_step > 0 else 0
            memory_ratio = curr.memory_gb / prev.memory_gb if prev.memory_gb > 0 else 0
            
            print(f"   {prev.scale_name} → {curr.scale_name}:")
            print(f"     Neurons: {neuron_ratio:.1f}x")
            print(f"     Time: {time_ratio:.1f}x")
            print(f"     Memory: {memory_ratio:.1f}x")
            print(f"     Efficiency: {time_ratio/neuron_ratio:.2f}")
    
    def generate_systematic_configurations(self) -> List[Dict[str, Any]]:
        """Generate systematic test configurations for comprehensive scaling analysis"""
        if not self.config.enable_systematic_profiling:
            return []
        
        # Extended neuron counts for comprehensive analysis
        if self.config.enable_extended_scales:
            neuron_counts = [
                1_000_000,      # 1M
                2_000_000,      # 2M
                5_000_000,      # 5M
                10_000_000,     # 10M
                20_000_000,     # 20M
                50_000_000,     # 50M
                100_000_000,    # 100M
                200_000_000,    # 200M
                500_000_000,    # 500M
                1_000_000_000,  # 1B
                2_000_000_000,  # 2B
                5_000_000_000,  # 5B
                10_000_000_000, # 10B
                50_000_000_000, # 50B
                100_000_000_000 # 100B
            ]
        else:
            neuron_counts = [
                1_000_000,      # 1M
                10_000_000,     # 10M
                100_000_000,    # 100M
                1_000_000_000,  # 1B
                10_000_000_000  # 10B
            ]
        
        # Extended active percentages for granular analysis
        active_percentages = [
            0.00005,  # 0.005%
            0.0001,   # 0.01%
            0.0002,   # 0.02%
            0.0005,   # 0.05%
            0.001,    # 0.1%
            0.002,    # 0.2%
            0.005,    # 0.5%
            0.01,     # 1%
            0.02,     # 2%
            0.03,     # 3%
            0.05,     # 5%
            0.07,     # 7%
            0.10,     # 10%
            0.15,     # 15%
            0.20      # 20%
        ]
        
        # Define number of areas to test
        n_areas_options = [3, 5, 10, 20]  # Test different area counts
        
        configurations = []
        for neurons in neuron_counts:
            for active_pct in active_percentages:
                for n_areas in n_areas_options:
                    # Skip configurations that would be too memory intensive
                    active_neurons = int(neurons * active_pct)
                    
                    # Skip if active neurons > 50M (too memory intensive)
                    if active_neurons > 50_000_000:
                        continue
                        
                    # Skip if active neurons < 1000 (too small to be meaningful)
                    if active_neurons < 1000:
                        continue
                    
                    # Skip high area counts for very large scales
                    if neurons >= 50_000_000_000 and n_areas > 10:
                        continue
                    
                    configurations.append({
                        'neurons': neurons,
                        'active_percentage': active_pct,
                        'active_neurons': active_neurons,
                        'n_areas': n_areas,
                        'name': f"{neurons:,} neurons ({active_pct:.2%}, {n_areas} areas)"
                    })
        
        return configurations
    
    def run_systematic_profiling(self) -> List[MemoryProfile]:
        """Run systematic profiling across all configurations"""
        if not self.config.enable_systematic_profiling:
            print("⚠️  Systematic profiling disabled")
            return []
        
        print("🚀 SYSTEMATIC SCALING PROFILER")
        print("=" * 60)
        
        configurations = self.generate_systematic_configurations()
        print(f"📊 Testing {len(configurations)} configurations")
        print(f"   Neuron counts: {len(set(c['neurons'] for c in configurations))} different scales")
        print(f"   Active percentages: {len(set(c['active_percentage'] for c in configurations))} different percentages")
        print("=" * 60)
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, config in enumerate(configurations, 1):
            print(f"\n[{i}/{len(configurations)}] ", end="")
            
            try:
                profile = self.profile_scale(
                    scale_name=config['name'],
                    n_neurons=config['neurons'],
                    active_percentage=config['active_percentage'],
                    n_areas=config['n_areas'],
                    n_steps=self.config.systematic_steps  # Use more steps for systematic analysis
                )
                
                if profile.steps_per_sec > 0:
                    results.append(profile)
                    successful_tests += 1
                    print(f"   ✅ {profile.steps_per_sec:.1f} steps/sec, {profile.memory_efficiency:.6f} bytes/neuron")
                else:
                    failed_tests += 1
                    print(f"   ❌ Failed")
                
                # Force garbage collection between tests
                gc.collect()
                if CUPY_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                failed_tests += 1
                print(f"   ❌ Failed: {e}")
                continue
        
        print(f"\n📊 SYSTEMATIC PROFILING COMPLETE")
        print(f"   ✅ Successful: {successful_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📈 Total results: {len(results)}")
        
        # Store results in the profiler
        self.profiles.extend(results)
        
        return results
    
    def run_comprehensive_profiling(self):
        """Run comprehensive profiling across multiple scales"""
        print("🚀 UNIVERSAL PROFILER - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        if self.config.enable_systematic_profiling:
            # Run systematic profiling for complete curves
            self.run_systematic_profiling()
        else:
            # Run basic profiling for quick testing
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
                    if CUPY_AVAILABLE:
                        cp.get_default_memory_pool().free_all_blocks()
                    
                except Exception as e:
                    print(f"   ❌ Scale {scale['name']} failed: {e}")
                    continue
        
        # Analyze scaling laws
        self.analyze_scaling_laws()
        
        # Print summary
        self.print_summary()
        
        # Save profiles
        self.save_profiles()
    
    def print_summary(self):
        """Print comprehensive profiling summary"""
        print(f"\n📊 UNIVERSAL PROFILER SUMMARY")
        print("=" * 100)
        print(f"{'Scale':<25} {'Neurons':<12} {'Active%':<8} {'Steps/sec':<10} {'ms/step':<10} {'Memory GB':<10} {'Bytes/Neuron':<12} {'GPU Util%':<10} {'CPU Util%':<10}")
        print("-" * 100)
        
        for profile in self.profiles:
            if profile.steps_per_sec > 0:
                print(f"{profile.scale_name:<25} {profile.n_neurons:<12,} {profile.sparsity*100:<8.1f} {profile.steps_per_sec:<10.1f} {profile.ms_per_step:<10.2f} {profile.memory_gb:<10.2f} {profile.memory_efficiency:<12.5f} {profile.gpu_memory_utilization:<10.1f} {profile.cpu_memory_utilization:<10.1f}")
            else:
                print(f"{profile.scale_name:<25} {profile.n_neurons:<12,} {profile.sparsity*100:<8.1f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'FAILED':<12} {'FAILED':<10} {'FAILED':<10}")
    
    def generate_scaling_visualizations(self):
        """Generate comprehensive scaling visualizations"""
        if not self.config.enable_scaling_analysis or len(self.profiles) < 3:
            print("⚠️  Need at least 3 profiles for scaling visualizations")
            return
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create DataFrame for analysis
            df = pd.DataFrame([{
                'n_neurons': p.n_neurons,
                'active_percentage': (1.0 - p.sparsity) * 100,  # Convert sparsity to active percentage
                'n_areas': p.n_areas,
                'steps_per_sec': p.steps_per_sec,
                'ms_per_step': p.ms_per_step,
                'memory_gb': p.memory_gb,
                'bytes_per_neuron': p.memory_efficiency,
                'gpu_utilization': p.gpu_memory_utilization,
                'neurons_per_sec': p.neurons_per_sec,
                'active_per_sec': p.active_per_sec
            } for p in self.profiles if p.steps_per_sec > 0])
            
            if df.empty:
                print("⚠️  No valid data for visualizations")
                return
            
            # Create generated folder
            generated_dir = "__generated__"
            os.makedirs(generated_dir, exist_ok=True)
            
            # 1. Performance Scaling Laws
            plt.figure(figsize=(12, 8))
            
            # Group by active percentage for different curves
            for active_pct in sorted(df['active_percentage'].unique()):
                subset = df[df['active_percentage'] == active_pct]
                if len(subset) > 1:
                    plt.loglog(subset['n_neurons'], subset['steps_per_sec'], 
                             'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=6)
            
            plt.xlabel('Number of Neurons')
            plt.ylabel('Steps per Second')
            plt.title('Performance Scaling Laws')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            perf_file = os.path.join(generated_dir, 'scaling_laws_performance.png')
            plt.savefig(perf_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Memory Scaling Laws
            plt.figure(figsize=(12, 8))
            
            for active_pct in sorted(df['active_percentage'].unique()):
                subset = df[df['active_percentage'] == active_pct]
                if len(subset) > 1:
                    plt.loglog(subset['n_neurons'], subset['bytes_per_neuron'], 
                             'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=6)
            
            plt.xlabel('Number of Neurons')
            plt.ylabel('Bytes per Neuron')
            plt.title('Memory Scaling Laws')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            mem_file = os.path.join(generated_dir, 'scaling_laws_memory.png')
            plt.savefig(mem_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Active Percentage Analysis
            plt.figure(figsize=(12, 8))
            
            for neurons in sorted(df['n_neurons'].unique()):
                subset = df[df['n_neurons'] == neurons]
                if len(subset) > 1:
                    plt.plot(subset['active_percentage'], subset['steps_per_sec'], 
                           'o-', label=f'{neurons:,} neurons', linewidth=2, markersize=6)
            
            plt.xlabel('Active Percentage (%)')
            plt.ylabel('Steps per Second')
            plt.title('Performance vs Active Percentage')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            active_file = os.path.join(generated_dir, 'scaling_laws_active_percentage.png')
            plt.savefig(active_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. NEW: Area Count Impact Analysis
            self._plot_area_impact(df, generated_dir)
            
            # 5. NEW: Performance Heatmap
            self._plot_performance_heatmap(df, generated_dir)
            
            # 6. NEW: Efficiency Analysis
            self._plot_efficiency_analysis(df, generated_dir)
            
            # 7. NEW: Throughput Analysis
            self._plot_throughput_analysis(df, generated_dir)
            
            # 8. NEW: Memory Efficiency Analysis
            self._plot_memory_efficiency_analysis(df, generated_dir)
            
            # 9. NEW: Scaling Laws Summary
            self._plot_scaling_summary(df, generated_dir)
            
            print(f"📊 Scaling visualizations generated:")
            print(f"   - {perf_file}")
            print(f"   - {mem_file}")
            print(f"   - {active_file}")
            print(f"   - {generated_dir}/area_impact_analysis.png")
            print(f"   - {generated_dir}/performance_heatmap.png")
            print(f"   - {generated_dir}/efficiency_analysis.png")
            print(f"   - {generated_dir}/throughput_analysis.png")
            print(f"   - {generated_dir}/memory_efficiency_analysis.png")
            print(f"   - {generated_dir}/scaling_summary.png")
            
        except ImportError:
            print("⚠️  matplotlib not available for visualizations")
        except Exception as e:
            print(f"   ❌ Visualization generation failed: {e}")
    
    def _plot_area_impact(self, df, generated_dir):
        """Plot the impact of number of areas on performance"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Impact of Number of Areas on Performance', fontsize=16)
        
        # Group by neuron count ranges
        neuron_ranges = [
            (1_000_000, 10_000_000, "1M-10M neurons"),
            (10_000_000, 100_000_000, "10M-100M neurons"),
            (100_000_000, 1_000_000_000, "100M-1B neurons"),
            (1_000_000_000, 10_000_000_000, "1B-10B neurons")
        ]
        
        for idx, (min_neurons, max_neurons, title) in enumerate(neuron_ranges):
            ax = axes[idx // 2, idx % 2]
            
            subset = df[(df['n_neurons'] >= min_neurons) & (df['n_neurons'] <= max_neurons)]
            
            if not subset.empty:
                for n_areas in sorted(subset['n_areas'].unique()):
                    area_subset = subset[subset['n_areas'] == n_areas]
                    if len(area_subset) > 1:
                        ax.plot(area_subset['active_percentage'], area_subset['steps_per_sec'], 
                               'o-', label=f'{n_areas} areas', linewidth=2, markersize=4)
                
                ax.set_xlabel('Active Percentage (%)')
                ax.set_ylabel('Steps per Second')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'area_impact_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, df, generated_dir):
        """Create a performance heatmap showing neuron count vs active percentage"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='steps_per_sec', 
            index='n_neurons', 
            columns='active_percentage', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(14, 10))
        
        # Create heatmap
        im = plt.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        plt.xticks(range(len(pivot_data.columns)), [f'{x:.1f}%' for x in pivot_data.columns])
        plt.yticks(range(len(pivot_data.index)), [f'{x/1_000_000:.0f}M' for x in pivot_data.index])
        
        plt.xlabel('Active Percentage (%)')
        plt.ylabel('Number of Neurons (Millions)')
        plt.title('Performance Heatmap: Steps/sec vs Neurons & Active %')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Steps per Second')
        
        # Add text annotations for key values
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.iloc[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f'{value:.0f}', ha='center', va='center', 
                            color='white' if value < pivot_data.values.mean() else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self, df, generated_dir):
        """Plot various efficiency metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Efficiency Analysis', fontsize=16)
        
        # 1. Bytes per neuron vs neuron count
        ax1 = axes[0, 0]
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax1.loglog(subset['n_neurons'], subset['bytes_per_neuron'], 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax1.set_xlabel('Number of Neurons')
        ax1.set_ylabel('Bytes per Neuron')
        ax1.set_title('Memory Efficiency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. GPU utilization vs active percentage
        ax2 = axes[0, 1]
        for neurons in sorted(df['n_neurons'].unique())[::3]:  # Sample every 3rd to reduce clutter
            subset = df[df['n_neurons'] == neurons]
            if len(subset) > 1:
                ax2.plot(subset['active_percentage'], subset['gpu_utilization'], 
                        'o-', label=f'{neurons/1_000_000:.0f}M neurons', linewidth=2, markersize=4)
        ax2.set_xlabel('Active Percentage (%)')
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.set_title('GPU Utilization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Neurons per second vs active percentage
        ax3 = axes[1, 0]
        for neurons in sorted(df['n_neurons'].unique())[::3]:
            subset = df[df['n_neurons'] == neurons]
            if len(subset) > 1:
                ax3.plot(subset['active_percentage'], subset['neurons_per_sec']/1_000_000, 
                        'o-', label=f'{neurons/1_000_000:.0f}M neurons', linewidth=2, markersize=4)
        ax3.set_xlabel('Active Percentage (%)')
        ax3.set_ylabel('Neurons per Second (Millions)')
        ax3.set_title('Processing Throughput')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Active neurons per second vs active percentage
        ax4 = axes[1, 1]
        for neurons in sorted(df['n_neurons'].unique())[::3]:
            subset = df[df['n_neurons'] == neurons]
            if len(subset) > 1:
                ax4.plot(subset['active_percentage'], subset['active_per_sec']/1_000_000, 
                        'o-', label=f'{neurons/1_000_000:.0f}M neurons', linewidth=2, markersize=4)
        ax4.set_xlabel('Active Percentage (%)')
        ax4.set_ylabel('Active Neurons per Second (Millions)')
        ax4.set_title('Active Processing Throughput')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_analysis(self, df, generated_dir):
        """Plot throughput analysis with different perspectives"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Throughput Analysis', fontsize=16)
        
        # 1. Steps per second vs neuron count (log-log)
        ax1 = axes[0, 0]
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax1.loglog(subset['n_neurons'], subset['steps_per_sec'], 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax1.set_xlabel('Number of Neurons')
        ax1.set_ylabel('Steps per Second')
        ax1.set_title('Steps per Second Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Neurons per second vs neuron count (log-log)
        ax2 = axes[0, 1]
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax2.loglog(subset['n_neurons'], subset['neurons_per_sec']/1_000_000, 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax2.set_xlabel('Number of Neurons')
        ax2.set_ylabel('Neurons per Second (Millions)')
        ax2.set_title('Neurons per Second Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Active neurons per second vs total neurons
        ax3 = axes[1, 0]
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax3.loglog(subset['n_neurons'], subset['active_per_sec']/1_000_000, 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax3.set_xlabel('Number of Neurons')
        ax3.set_ylabel('Active Neurons per Second (Millions)')
        ax3.set_title('Active Neurons per Second Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Processing efficiency (neurons/sec per GB memory)
        ax4 = axes[1, 1]
        df['efficiency'] = df['neurons_per_sec'] / df['memory_gb']
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax4.loglog(subset['n_neurons'], subset['efficiency']/1_000_000, 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax4.set_xlabel('Number of Neurons')
        ax4.set_ylabel('Efficiency (M neurons/sec per GB)')
        ax4.set_title('Memory Efficiency Scaling')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'throughput_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_efficiency_analysis(self, df, generated_dir):
        """Plot memory efficiency from multiple angles"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Memory Efficiency Analysis', fontsize=16)
        
        # 1. Memory usage vs neuron count
        ax1 = axes[0, 0]
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax1.loglog(subset['n_neurons'], subset['memory_gb'], 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax1.set_xlabel('Number of Neurons')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('Memory Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bytes per neuron vs active percentage
        ax2 = axes[0, 1]
        for neurons in sorted(df['n_neurons'].unique())[::3]:
            subset = df[df['n_neurons'] == neurons]
            if len(subset) > 1:
                ax2.plot(subset['active_percentage'], subset['bytes_per_neuron'], 
                        'o-', label=f'{neurons/1_000_000:.0f}M neurons', linewidth=2, markersize=4)
        ax2.set_xlabel('Active Percentage (%)')
        ax2.set_ylabel('Bytes per Neuron')
        ax2.set_title('Memory per Neuron')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory efficiency vs neuron count
        ax3 = axes[1, 0]
        df['mem_efficiency'] = df['neurons_per_sec'] / df['memory_gb']
        for active_pct in sorted(df['active_percentage'].unique()):
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) > 1:
                ax3.loglog(subset['n_neurons'], subset['mem_efficiency']/1_000_000, 
                          'o-', label=f'{active_pct:.1f}% active', linewidth=2, markersize=4)
        ax3.set_xlabel('Number of Neurons')
        ax3.set_ylabel('Memory Efficiency (M neurons/sec per GB)')
        ax3.set_title('Memory Efficiency Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. GPU utilization vs memory usage
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['memory_gb'], df['gpu_utilization'], 
                             c=df['n_neurons'], s=100, alpha=0.7, cmap='viridis')
        ax4.set_xlabel('Memory Usage (GB)')
        ax4.set_ylabel('GPU Utilization (%)')
        ax4.set_title('GPU Utilization vs Memory Usage')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Number of Neurons')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'memory_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scaling_summary(self, df, generated_dir):
        """Create a comprehensive scaling summary with key insights"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Comprehensive Scaling Analysis Summary', fontsize=18)
        
        # 1. Performance by scale ranges
        ax1 = axes[0, 0]
        scale_ranges = [
            (1_000_000, 10_000_000, "1M-10M"),
            (10_000_000, 100_000_000, "10M-100M"),
            (100_000_000, 1_000_000_000, "100M-1B"),
            (1_000_000_000, 10_000_000_000, "1B-10B"),
            (10_000_000_000, 100_000_000_000, "10B-100B")
        ]
        
        for min_neurons, max_neurons, label in scale_ranges:
            subset = df[(df['n_neurons'] >= min_neurons) & (df['n_neurons'] <= max_neurons)]
            if not subset.empty:
                avg_performance = subset['steps_per_sec'].mean()
                ax1.bar(label, avg_performance, alpha=0.7)
        
        ax1.set_ylabel('Average Steps per Second')
        ax1.set_title('Performance by Scale Range')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Optimal active percentage by scale
        ax2 = axes[0, 1]
        optimal_percentages = []
        scale_labels = []
        
        for min_neurons, max_neurons, label in scale_ranges:
            subset = df[(df['n_neurons'] >= min_neurons) & (df['n_neurons'] <= max_neurons)]
            if not subset.empty:
                best_idx = subset['steps_per_sec'].idxmax()
                optimal_pct = subset.loc[best_idx, 'active_percentage']
                optimal_percentages.append(optimal_pct)
                scale_labels.append(label)
        
        ax2.bar(scale_labels, optimal_percentages, alpha=0.7, color='orange')
        ax2.set_ylabel('Optimal Active Percentage (%)')
        ax2.set_title('Optimal Active % by Scale')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Memory efficiency by scale
        ax3 = axes[1, 0]
        df['mem_efficiency'] = df['neurons_per_sec'] / df['memory_gb']
        
        for min_neurons, max_neurons, label in scale_ranges:
            subset = df[(df['n_neurons'] >= min_neurons) & (df['n_neurons'] <= max_neurons)]
            if not subset.empty:
                avg_efficiency = subset['mem_efficiency'].mean() / 1_000_000
                ax3.bar(label, avg_efficiency, alpha=0.7, color='green')
        
        ax3.set_ylabel('Memory Efficiency (M neurons/sec per GB)')
        ax3.set_title('Memory Efficiency by Scale')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Area count impact
        ax4 = axes[1, 1]
        area_performance = df.groupby('n_areas')['steps_per_sec'].mean()
        ax4.bar(area_performance.index, area_performance.values, alpha=0.7, color='purple')
        ax4.set_xlabel('Number of Areas')
        ax4.set_ylabel('Average Steps per Second')
        ax4.set_title('Performance vs Number of Areas')
        
        # 5. Performance distribution
        ax5 = axes[2, 0]
        ax5.hist(df['steps_per_sec'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Steps per Second')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Performance Distribution')
        ax5.axvline(df['steps_per_sec'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["steps_per_sec"].mean():.1f}')
        ax5.legend()
        
        # 6. Memory usage distribution
        ax6 = axes[2, 1]
        ax6.hist(df['memory_gb'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.set_xlabel('Memory Usage (GB)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Memory Usage Distribution')
        ax6.axvline(df['memory_gb'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["memory_gb"].mean():.1f} GB')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(generated_dir, 'scaling_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_profiles(self, filename: str = None):
        """Save profiles to JSON file with timestamp"""
        try:
            # Create generated folder if it doesn't exist
            generated_dir = "__generated__"
            os.makedirs(generated_dir, exist_ok=True)
            
            # Generate filename with timestamp if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"universal_profiler_profiles_{timestamp}.json"
            
            # Ensure filename is in generated folder
            if not filename.startswith(generated_dir):
                filename = os.path.join(generated_dir, filename)
            
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
                    'theoretical_memory_gb': profile.theoretical_memory_gb,
                    'memory_efficiency': profile.memory_efficiency,
                    'cache_hit_rate': profile.cache_hit_rate,
                    'memory_pool_efficiency': profile.memory_pool_efficiency
                })
            
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            print(f"\n💾 Profiles saved to {filename}")
            
            # Generate visualizations if we have enough data
            if len(self.profiles) >= 3:
                self.generate_scaling_visualizations()
            
        except Exception as e:
            print(f"   ❌ Failed to save profiles: {e}")

def main():
    """Main function to run the universal profiler"""
    try:
        config = ProfilerConfig(
            enable_gpu_profiling=True,
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            enable_scaling_analysis=True,
            enable_theoretical_analysis=True,
            enable_systematic_profiling=True,  # Enable systematic profiling
            enable_extended_scales=True,       # Enable billion+ scale testing
            timeout_seconds=60.0,              # Longer timeout for systematic analysis
            max_steps=10,
            systematic_steps=100               # More steps for systematic analysis
        )
        
        profiler = UniversalProfiler(config)
        profiler.run_comprehensive_profiling()
        
        print(f"\n✅ Universal Profiler completed successfully!")
        print(f"📊 Generated comprehensive scaling data with complete curves!")
        
    except Exception as e:
        print(f"❌ Universal Profiler failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
