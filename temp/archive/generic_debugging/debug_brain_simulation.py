#!/usr/bin/env python3
"""
Debug brain simulation step by step
"""

import time
import numpy as np
import cupy as cp

print("🔍 Debugging brain simulation step by step...")

# Test parameters
n_neurons = 1000000
active_percentage = 0.01
k_active = int(n_neurons * active_percentage)
n_areas = 5

print(f"   Neurons: {n_neurons:,}")
print(f"   Active percentage: {active_percentage*100:.4f}%")
print(f"   Active per area: {k_active:,}")
print(f"   Areas: {n_areas}")

# Initialize areas
print("\n1. Initializing areas...")
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
print("   ✅ Areas initialized")

# Pre-allocated arrays
print("\n2. Pre-allocating arrays...")
candidates = cp.zeros(k_active * 10, dtype=cp.float32)
top_k_indices = cp.zeros(k_active, dtype=cp.int32)
top_k_values = cp.zeros(k_active, dtype=cp.float32)
sorted_indices = cp.zeros(k_active, dtype=cp.int32)
print("   ✅ Arrays pre-allocated")

# Test candidate generation
print("\n3. Testing candidate generation...")
area_idx = 0
area = areas[area_idx]
candidates_slice = candidates[:area['k']]

print(f"   Candidates slice shape: {candidates_slice.shape}")
print(f"   Candidates slice dtype: {candidates_slice.dtype}")

try:
    new_candidates = cp.random.exponential(1.0, size=len(candidates_slice))
    print(f"   New candidates shape: {new_candidates.shape}")
    print(f"   New candidates dtype: {new_candidates.dtype}")
    
    candidates_slice[:] = new_candidates
    print("   ✅ Candidate generation works")
    print(f"   Sample values: {candidates_slice[:5].get()}")
except Exception as e:
    print(f"   ❌ Candidate generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test top-k selection
print("\n4. Testing top-k selection...")
try:
    k = area['k']
    if k >= len(candidates_slice):
        winners = np.arange(len(candidates_slice))
    else:
        top_k_indices_slice = top_k_indices[:k]
        top_k_indices_slice[:] = cp.argpartition(candidates_slice, -k)[-k:]
        
        top_k_values_slice = top_k_values[:k]
        top_k_values_slice[:] = candidates_slice[top_k_indices_slice]
        
        sorted_indices_slice = sorted_indices[:k]
        sorted_indices_slice[:] = cp.argsort(top_k_values_slice)[::-1]
        
        winners = top_k_indices_slice[sorted_indices_slice]
    
    print("   ✅ Top-k selection works")
    print(f"   Winners shape: {winners.shape}")
    print(f"   Sample winners: {winners[:5].get()}")
except Exception as e:
    print(f"   ❌ Top-k selection failed: {e}")
    import traceback
    traceback.print_exc()

# Test weight updates
print("\n5. Testing weight updates...")
try:
    area['weights'][winners] += 0.1
    area['weights'] *= 0.99
    area['support'][winners] += 1.0
    print("   ✅ Weight updates work")
    print(f"   Sample weights: {area['weights'][:5].get()}")
except Exception as e:
    print(f"   ❌ Weight updates failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Debug complete!")
