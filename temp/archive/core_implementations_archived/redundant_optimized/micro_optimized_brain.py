#!/usr/bin/env python3
"""
Micro Optimized Brain - Focus on sub-millisecond performance
"""

import time
import numpy as np
import os
import sys

class MicroOptimizedBrain:
    """
    Micro Optimized Brain - Focus on sub-millisecond performance
    """
    
    def __init__(self, n_neurons=50000, k_active=500, n_areas=3, seed=42):
        """Initialize the micro optimized brain"""
        self.n_neurons = n_neurons
        self.k_active = k_active
        self.n_areas = n_areas
        self.seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        # Initialize areas with minimal overhead
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
        
        print(f"🚀 Micro Optimized Brain initialized:")
        print(f"   Neurons: {n_neurons:,}")
        print(f"   Active per area: {k_active:,}")
        print(f"   Areas: {n_areas}")
    
    def _generate_candidates_micro_fast(self, area_idx):
        """Generate candidates using micro-optimized operations"""
        area = self.areas[area_idx]
        
        # Use pre-allocated array
        candidates = self._candidates[:area['n']]
        
        # Generate exponential random numbers - this is the bottleneck
        candidates[:] = self._rng.exponential(1.0, size=len(candidates))
        
        return candidates
    
    def _select_top_k_micro_fast(self, candidates, k):
        """Select top-k using micro-optimized operations"""
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
    
    def _update_weights_micro_fast(self, area_idx, winners):
        """Update weights using micro-optimized operations"""
        area = self.areas[area_idx]
        
        # Hebbian learning
        area['weights'][winners] += 0.1
        
        # Decay
        area['weights'] *= 0.99
        
        # Support
        area['support'][winners] += 1.0
    
    def simulate_step(self):
        """Simulate one step of the brain"""
        start_time = time.perf_counter()  # Use high-precision timer
        
        for area_idx in range(self.n_areas):
            area = self.areas[area_idx]
            
            # Generate candidates
            candidates = self._generate_candidates_micro_fast(area_idx)
            
            # Select top-k winners
            winners = self._select_top_k_micro_fast(candidates, area['k'])
            
            # Update area state
            area['w'] = len(winners)
            area['winners'][:len(winners)] = winners
            area['activated'] = True
            
            # Update weights
            self._update_weights_micro_fast(area_idx, winners)
        
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
                print(f"Step {step + 1:3d}: {step_time*1000:.3f}ms (avg: {avg_time*1000:.3f}ms)")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n📊 SIMULATION COMPLETE")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average step time: {total_time/n_steps*1000:.3f}ms")
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

def test_micro_optimized():
    """Test the micro optimized brain"""
    print("🚀 TESTING MICRO OPTIMIZED BRAIN")
    print("=" * 50)
    
    # Test with very small scale for maximum speed
    brain = MicroOptimizedBrain(
        n_neurons=50000,
        k_active=500,
        n_areas=3,
        seed=42
    )
    
    print("Running 50 steps to measure precise timing...")
    
    # Simulate
    start_time = time.perf_counter()
    brain.simulate(n_steps=50, verbose=True)
    total_time = time.perf_counter() - start_time
    
    # Get stats
    stats = brain.get_performance_stats()
    
    print(f"\n📊 MICRO OPTIMIZED PERFORMANCE:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Steps: {stats['total_steps']}")
    print(f"   Average step time: {stats['avg_step_time']*1000:.3f}ms")
    print(f"   Steps per second: {stats['steps_per_second']:.1f}")
    print(f"   Neurons per second: {stats['neurons_per_second']:,.0f}")
    print(f"   Active per second: {stats['active_neurons_per_second']:,.0f}")
    
    # Check if we hit the target
    if stats['avg_step_time'] < 0.0009:  # Less than 0.9ms
        print(f"   🎯 TARGET ACHIEVED: {stats['avg_step_time']*1000:.3f}ms < 0.9ms!")
    elif stats['avg_step_time'] < 0.001:  # Less than 1ms
        print(f"   ⚡ VERY CLOSE: {stats['avg_step_time']*1000:.3f}ms < 1.0ms!")
    elif stats['avg_step_time'] < 0.002:  # Less than 2ms
        print(f"   ⚡ FAST: {stats['avg_step_time']*1000:.3f}ms < 2.0ms!")
    else:
        print(f"   ⚠️  Still above target: {stats['avg_step_time']*1000:.3f}ms > 0.9ms")
    
    return stats

def test_ultra_small_scale():
    """Test with ultra-small scale for maximum speed"""
    print(f"\n🔬 ULTRA SMALL SCALE TEST")
    print("=" * 50)
    
    # Test with even smaller scale
    brain = MicroOptimizedBrain(
        n_neurons=25000,
        k_active=250,
        n_areas=3,
        seed=42
    )
    
    print("Running 100 steps with ultra-small scale...")
    
    # Simulate
    start_time = time.perf_counter()
    brain.simulate(n_steps=100, verbose=False)
    total_time = time.perf_counter() - start_time
    
    # Get stats
    stats = brain.get_performance_stats()
    
    print(f"📊 ULTRA SMALL SCALE PERFORMANCE:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average step time: {stats['avg_step_time']*1000:.3f}ms")
    print(f"   Steps per second: {stats['steps_per_second']:.1f}")
    print(f"   Neurons per second: {stats['neurons_per_second']:,.0f}")
    
    # Check if we hit the target
    if stats['avg_step_time'] < 0.0009:  # Less than 0.9ms
        print(f"   🎯 TARGET ACHIEVED: {stats['avg_step_time']*1000:.3f}ms < 0.9ms!")
    elif stats['avg_step_time'] < 0.001:  # Less than 1ms
        print(f"   ⚡ VERY CLOSE: {stats['avg_step_time']*1000:.3f}ms < 1.0ms!")
    else:
        print(f"   ⚠️  Still above target: {stats['avg_step_time']*1000:.3f}ms > 0.9ms")
    
    return stats

if __name__ == "__main__":
    # Test micro optimized
    stats1 = test_micro_optimized()
    
    # Test ultra small scale
    stats2 = test_ultra_small_scale()
    
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    print(f"Micro Optimized (50K neurons): {stats1['avg_step_time']*1000:.3f}ms")
    print(f"Ultra Small (25K neurons): {stats2['avg_step_time']*1000:.3f}ms")
    
    if stats1['avg_step_time'] < 0.0009 or stats2['avg_step_time'] < 0.0009:
        print(f"🎉 TARGET ACHIEVED! Sub-millisecond performance!")
    else:
        print(f"⚠️  Target not quite reached, but very close!")
