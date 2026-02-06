"""Test with optimal parameters from sweep: p=0.1, beta=1.0"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import cupy as cp
import numpy as np
from src.nemo.language.emergent.brain import EmergentNemoBrain
from src.nemo.language.emergent.areas import Area
from src.nemo.language.emergent.params import EmergentParams

def compute_overlap(a1, a2, k):
    s1 = set(a1.get().tolist())
    s2 = set(a2.get().tolist())
    return len(s1 & s2) / k

# Use the BEST parameters from sweep
params = EmergentParams()
params.n = 10000
params.k = 100
params.p = 0.1   # Higher connectivity (best from sweep)
params.beta = 1.0  # Best learning rate

brain = EmergentNemoBrain(params=params, verbose=False)
k = params.k
AREA = Area.NOUN_CORE

print('='*60)
print('Testing with BEST parameters: p=0.1, beta=1.0, k=100')
print('='*60)

# Create 4 assemblies
assemblies = {}
for i in range(4):
    cp.random.seed((i + 1) * 1000)
    stimulus = cp.random.randint(0, params.n, k, dtype=cp.uint32)
    
    brain._clear_area(AREA)
    for _ in range(100):
        brain._project(AREA, stimulus, learn=True)
    assemblies[f'asm{i}'] = brain.current[AREA].copy()

print('1. Created 4 assemblies (100 iterations each)')

# Build intra-assembly connections
for name, asm in assemblies.items():
    for _ in range(200):
        brain.clear_all()
        brain.current[AREA] = asm.copy()
        brain.prev[AREA] = asm.copy()
        brain._project(AREA, asm, learn=True)

print('2. Built intra-assembly connections (200 self-projections each)')

# Test pattern completion
print()
print('3. Pattern completion test (50% cue):')

correct = 0
for i in range(4):
    asm = assemblies[f'asm{i}']
    asm_indices = asm.get().tolist()
    partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
    
    brain.clear_all()
    brain.current[AREA] = partial_cue
    brain.prev[AREA] = partial_cue
    
    for _ in range(30):
        brain._project(AREA, brain.current[AREA], learn=False)
    
    retrieved = brain.current[AREA]
    overlaps = [compute_overlap(retrieved, assemblies[f'asm{j}'], k) for j in range(4)]
    best_match = np.argmax(overlaps)
    is_correct = best_match == i
    correct += is_correct
    
    print(f'   asm{i}: overlaps={[f"{o:.2f}" for o in overlaps]}, best=asm{best_match}, correct={is_correct}')

print()
print(f'   ACCURACY: {correct/4*100:.0f}%')

# Also test with MORE training
print()
print('='*60)
print('Testing with MORE training: 500 iterations + 500 self-proj')
print('='*60)

brain2 = EmergentNemoBrain(params=params, verbose=False)
assemblies2 = {}

for i in range(4):
    cp.random.seed((i + 1) * 1000)
    stimulus = cp.random.randint(0, params.n, k, dtype=cp.uint32)
    
    brain2._clear_area(AREA)
    for _ in range(500):  # Much more training
        brain2._project(AREA, stimulus, learn=True)
    assemblies2[f'asm{i}'] = brain2.current[AREA].copy()

print('1. Created 4 assemblies (500 iterations each)')

for name, asm in assemblies2.items():
    for _ in range(500):  # Much more self-projection
        brain2.clear_all()
        brain2.current[AREA] = asm.copy()
        brain2.prev[AREA] = asm.copy()
        brain2._project(AREA, asm, learn=True)

print('2. Built intra-assembly connections (500 self-projections each)')

print()
print('3. Pattern completion test (50% cue):')

correct = 0
for i in range(4):
    asm = assemblies2[f'asm{i}']
    asm_indices = asm.get().tolist()
    partial_cue = cp.array(asm_indices[:k//2], dtype=cp.uint32)
    
    brain2.clear_all()
    brain2.current[AREA] = partial_cue
    brain2.prev[AREA] = partial_cue
    
    for _ in range(30):
        brain2._project(AREA, brain2.current[AREA], learn=False)
    
    retrieved = brain2.current[AREA]
    overlaps = [compute_overlap(retrieved, assemblies2[f'asm{j}'], k) for j in range(4)]
    best_match = np.argmax(overlaps)
    is_correct = best_match == i
    correct += is_correct
    
    print(f'   asm{i}: overlaps={[f"{o:.2f}" for o in overlaps]}, best=asm{best_match}, correct={is_correct}')

print()
print(f'   ACCURACY: {correct/4*100:.0f}%')

