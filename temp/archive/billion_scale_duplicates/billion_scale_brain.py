#!/usr/bin/env python3
"""
Billion-Scale Brain - Optimized for extreme scale neural simulation
"""

import time
import numpy as np
import os
import sys
from numba import jit, cuda
import threading
from concurrent.futures import ThreadPoolExecutor

class BillionScaleBrain:
    """
    Billion-Scale Brain - Optimized for extreme scale neural simulation
    """
    
    def __init__(self, n_neurons=1000000000, k_active=10000000, n_areas=5, seed=42):
        """Initialize the billion-scale brain"""
        self.n_neurons = n_neurons
        self.k_active = k_active
        self.n_areas = n_areas
        self.seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        # Initialize areas with memory-efficient data types
        self.areas = []
        for i in range(n_areas):
            area = {
                'n': n_neurons,
                'k': k_active,
                'w': 0,
                'winners': np.zeros(n_neurons, dtype=np.int32),
                'weights': np.zeros(n_neurons, dtype=np.float32),
                'support': np.zeros(n_neurons, dtype=np.float32),
                'activated': False
            }
            self.areas.append(area)
        
        # Pre-allocated arrays for maximum performance
        self._candidates = np.zeros(n_neurons, dtype=np.float32)
        self._top_k_indices = np.zeros(k_active, dtype=np.int32)
        self._top_k_values = np.zeros(k_active, dtype=np.float32)
        self._sorted_indices = np.zeros(k_active, dtype=np.int32)
        
        # Performance counters
        self.step_count = 0
        self.total_time = 0.0
        
        print(f"🌍 Billion-Scale Brain initialized:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active per area: {k_active:,}")
        print(f"   Areas: {n_areas}")
        print(f"   Memory per area: {n_neurons * 4 * 3 / 1024 / 1024 / 1024:.2f} GB")
        print(f"   Total memory: {n_neurons * 4 * 3 * n_areas / 1024 / 1024 / 1024:.2f} GB")
    
    @staticmethod
    @jit(nopython=True)
    def _generate_candidates_jit(candidates, rng_state):
        """JIT-compiled candidate generation"""
        for i in range(len(candidates)):
            # Generate exponential random numbers using inverse transform
            u = np.random.random()
            candidates[i] = -np.log(1.0 - u)
    
    @staticmethod
    @jit(nopython=True)
    def _select_top_k_jit(candidates, k, top_k_indices, top_k_values, sorted_indices):
        """JIT-compiled top-k selection"""
        if k >= len(candidates):
            for i in range(len(candidates)):
                top_k_indices[i] = i
            return
        
        # Use argpartition for partial sorting
        indices = np.argpartition(candidates, -k)[-k:]
        top_k_indices[:k] = indices
        
        # Sort only the top-k
        values = candidates[indices]
        top_k_values[:k] = values
        
        sorted_idx = np.argsort(values)[::-1]
        sorted_indices[:k] = sorted_idx
        
        # Reorder indices
        for i in range(k):
            top_k_indices[i] = indices[sorted_idx[i]]
    
    @staticmethod
    @jit(nopython=True)
    def _update_weights_jit(weights, support, winners, learn_rate=0.1, decay=0.99):
        """JIT-compiled weight updates"""
        # Hebbian learning
        for i in range(len(winners)):
            weights[winners[i]] += learn_rate
        
        # Decay all weights
        for i in range(len(weights)):
            weights[i] *= decay
        
        # Update support
        for i in range(len(winners)):
            support[winners[i]] += 1.0
    
    def _generate_candidates_optimized(self, area_idx):
        """Generate candidates using optimized operations"""
        area = self.areas[area_idx]
        
        # Use pre-allocated array
        candidates = self._candidates[:area['n']]
        
        # Generate exponential random numbers using vectorized operations
        candidates[:] = self._rng.exponential(1.0, size=len(candidates))
        
        return candidates
    
    def _select_top_k_optimized(self, candidates, k):
        """Select top-k using optimized operations"""
        if k >= len(candidates):
            return np.arange(len(candidates))
        
        # Use argpartition for partial sorting
        top_k_indices = self._top_k_indices[:k]
        top_k_indices[:] = np.argpartition(candidates, -k)[-k:]
        
        # Sort only the top-k
        top_k_values = self._top_k_values[:k]
        top_k_values[:] = candidates[top_k_indices]
        
        sorted_indices = self._sorted_indices[:k]
        sorted_indices[:] = np.argsort(top_k_values)[::-1]
        
        return top_k_indices[sorted_indices]
    
    def _update_weights_optimized(self, area_idx, winners):
        """Update weights using optimized operations"""
        area = self.areas[area_idx]
        
        # Hebbian learning
        area['weights'][winners] += 0.1
        
        # Decay
        area['weights'] *= 0.99
        
        # Support
        area['support'][winners] += 1.0
    
    def simulate_step(self):
        """Simulate one step of the brain"""
        start_time = time.perf_counter()
        
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            
            # Generate candidates
            candidates = self._generate_candidates_optimized(area_idx)
            
            # Select top-k winners
            winners = self._select_top_k_optimized(candidates, area['k'])
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights
            self._update_weights_optimized(area_idx, winners)
        
        # Record total timing
        step_time = time.perf_counter() - start_time
        self.total_time += step_time
        self.step_count += 1
        
        return step_time
    
    def simulate(self, n_steps=100, verbose=True):
        """Simulate multiple steps"""
        if verbose:
            print(f"\n🧠 SIMULATING {n_steps} STEPS")
            print("=" * 50)
        
        start_time = time.perf_counter()
        
        for step in range(n_steps):
            step_time = self.simulate_step()
            
            if verbose and (step + 1) % 10 == 0:
                avg_time = self.total_time / self.step_count
                print(f"Step {step + 1:3d}: {step_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n📊 SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.2f}ms")
            print(f"   Steps per second: {n_steps/total_time:.1f}")
        
        return total_time
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if self.step_count == 0:
            return {}
        
        avg_step_time = self.total_time / self.step_count
        steps_per_second = 1.0 / avg_step_time
        
        return {
            'total_steps': self.step_count,
            'total_time': self.total_time,
            'avg_step_time': avg_step_time,
            'steps_per_second': steps_per_second,
            'neurons_per_second': self.n_neurons * steps_per_second,
            'active_neurons_per_second': self.k_active * self.n_areas * steps_per_second
        }

def test_billion_scale():
    """Test billion-scale performance"""
    print("🌍 TESTING BILLION-SCALE BRAIN")
    print("=" * 60)
    
    # Test different scales
    test_cases = [
        {"n_neurons": 1000000, "k_active": 10000, "n_areas": 5, "name": "Million Scale"},
        {"n_neurons": 10000000, "k_active": 100000, "n_areas": 5, "name": "Ten Million Scale"},
        {"n_neurons": 100000000, "k_active": 1000000, "n_areas": 5, "name": "Hundred Million Scale"},
        {"n_neurons": 1000000000, "k_active": 10000000, "n_areas": 5, "name": "BILLION SCALE"},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}:")
        print(f"   Neurons: {test_case['n_neurons']:,}")
        print(f"   Active: {test_case['k_active']:,}")
        print(f"   Areas: {test_case['n_areas']}")
        
        try:
            # Create brain
            brain = BillionScaleBrain(
                n_neurons=test_case['n_neurons'],
                k_active=test_case['k_active'],
                n_areas=test_case['n_areas'],
                seed=42
            )
            
            # Simulate
            start_time = time.perf_counter()
            brain.simulate(n_steps=10, verbose=False)  # Fewer steps for large scale
            total_time = time.perf_counter() - start_time
            
            # Get stats
            stats = brain.get_performance_stats()
            
            # Calculate performance metrics
            neurons_per_second = stats['neurons_per_second']
            active_neurons_per_second = stats['active_neurons_per_second']
            steps_per_second = stats['steps_per_second']
            ms_per_step = stats['avg_step_time'] * 1000
            
            print(f"   ✅ Success!")
            print(f"   Time: {total_time:.3f}s")
            print(f"   Steps/sec: {steps_per_second:.1f}")
            print(f"   ms/step: {ms_per_step:.2f}ms")
            print(f"   Neurons/sec: {neurons_per_second:,.0f}")
            print(f"   Active/sec: {active_neurons_per_second:,.0f}")
            
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'k_active': test_case['k_active'],
                'n_areas': test_case['n_areas'],
                'total_time': total_time,
                'steps_per_second': steps_per_second,
                'ms_per_step': ms_per_step,
                'neurons_per_second': neurons_per_second,
                'active_neurons_per_second': active_neurons_per_second
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': test_case['name'],
                'n_neurons': test_case['n_neurons'],
                'k_active': test_case['k_active'],
                'n_areas': test_case['n_areas'],
                'total_time': float('inf'),
                'steps_per_second': 0,
                'ms_per_step': float('inf'),
                'neurons_per_second': 0,
                'active_neurons_per_second': 0
            })
    
    # Summary
    print(f"\n📊 BILLION-SCALE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Scale':<20} {'Neurons':<15} {'Steps/sec':<10} {'ms/step':<10} {'Neurons/sec':<15} {'Active/sec':<15}")
    print("-" * 80)
    
    for result in results:
        if result['steps_per_second'] > 0:
            print(f"{result['name']:<20} {result['n_neurons']:<15,} {result['steps_per_second']:<10.1f} {result['ms_per_step']:<10.2f} {result['neurons_per_second']:<15,.0f} {result['active_neurons_per_second']:<15,.0f}")
        else:
            print(f"{result['name']:<20} {result['n_neurons']:<15,} {'FAILED':<10} {'FAILED':<10} {'FAILED':<15} {'FAILED':<15}")
    
    return results

if __name__ == "__main__":
    # Test billion-scale
    results = test_billion_scale()
    
    # Find best performance
    successful_results = [r for r in results if r['steps_per_second'] > 0]
    if successful_results:
        best = max(successful_results, key=lambda x: x['steps_per_second'])
        print(f"\n🏆 BEST PERFORMANCE: {best['name']}")
        print(f"   Steps/sec: {best['steps_per_second']:.1f}")
        print(f"   ms/step: {best['ms_per_step']:.2f}ms")
        print(f"   Neurons/sec: {best['neurons_per_second']:,.0f}")
        print(f"   Active/sec: {best['active_neurons_per_second']:,.0f}")
    else:
        print(f"\n❌ No successful tests")
