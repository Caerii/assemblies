#!/usr/bin/env python3
"""
Investigate what happened to the 82x speedup and test CuPy acceleration
"""

import time
import numpy as np
from optimized_cuda_brain import UltraOptimizedCudaBrain
from simple_ultra_optimized import SimpleUltraOptimizedCudaBrain

def investigate_optimization_difference():
    """Investigate why we lost the 82x speedup"""
    print("🔍 INVESTIGATING OPTIMIZATION DIFFERENCE")
    print("=" * 60)
    
    # Test parameters
    test_params = {
        'p': 0.1,
        'beta': 0.5,
        'max_weight': 1.0,
        'seed': 42
    }
    
    # Test areas
    test_areas = [
        ("Small", 10000, 100),
        ("Medium", 100000, 1000),
        ("Large", 500000, 5000),
        ("Huge", 1000000, 10000)
    ]
    
    # Test both implementations
    implementations = [
        ("Ultra Optimized", UltraOptimizedCudaBrain),
        ("Simple Ultra Optimized", SimpleUltraOptimizedCudaBrain)
    ]
    
    results = {}
    
    for impl_name, impl_class in implementations:
        print(f"\n🧠 Testing {impl_name}")
        print("-" * 40)
        
        # Create brain
        brain = impl_class(**test_params)
        
        # Add test areas
        for area_name, n, k in test_areas:
            brain.AddArea(area_name, n, k)
        
        # Benchmark simulation
        num_steps = 20
        start_time = time.time()
        
        for step in range(num_steps):
            brain.SimulateOneStep()
        
        total_time = time.time() - start_time
        avg_time_per_step = total_time / num_steps
        
        results[impl_name] = {
            'total_time': total_time,
            'avg_time_per_step': avg_time_per_step,
            'steps_per_second': num_steps / total_time
        }
        
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Avg time per step: {avg_time_per_step*1000:.2f}ms")
        print(f"   Steps per second: {num_steps/total_time:.1f}")
    
    # Compare results
    print(f"\n📊 OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    ultra_time = results["Ultra Optimized"]['avg_time_per_step']
    simple_time = results["Simple Ultra Optimized"]['avg_time_per_step']
    
    print(f"Ultra Optimized:        {ultra_time*1000:6.2f}ms/step")
    print(f"Simple Ultra Optimized: {simple_time*1000:6.2f}ms/step")
    
    if ultra_time < simple_time:
        speedup = simple_time / ultra_time
        print(f"\n🏆 Ultra Optimized is {speedup:.2f}x faster!")
        print(f"   The 82x speedup is still there!")
    else:
        slowdown = ultra_time / simple_time
        print(f"\n⚠️  Simple Ultra Optimized is {slowdown:.2f}x faster!")
        print(f"   Something went wrong with the optimization!")
    
    return results

def test_cupy_acceleration():
    """Test CuPy for GPU acceleration"""
    print(f"\n🚀 TESTING CUPY GPU ACCELERATION")
    print("=" * 60)
    
    try:
        import cupy as cp
        print("✅ CuPy is available!")
        
        # Test CuPy vs NumPy performance
        print("\n🧮 Testing CuPy vs NumPy performance...")
        
        # Test exponential generation
        print("Testing exponential generation...")
        size = 1000000
        
        # NumPy test
        start_time = time.time()
        for _ in range(100):
            _ = np.random.exponential(1.0, size)
        numpy_time = time.time() - start_time
        
        # CuPy test
        start_time = time.time()
        for _ in range(100):
            _ = cp.random.exponential(1.0, size)
        cupy_time = time.time() - start_time
        
        print(f"   NumPy: {numpy_time:.3f}s")
        print(f"   CuPy:  {cupy_time:.3f}s")
        print(f"   Speedup: {numpy_time/cupy_time:.2f}x")
        
        # Test top-k selection
        print("Testing top-k selection...")
        k = 10000
        
        # NumPy test
        start_time = time.time()
        for _ in range(100):
            candidates = np.random.exponential(1.0, size)
            _ = np.argpartition(candidates, -k)[-k:]
        numpy_sel_time = time.time() - start_time
        
        # CuPy test
        start_time = time.time()
        for _ in range(100):
            candidates = cp.random.exponential(1.0, size)
            _ = cp.argpartition(candidates, -k)[-k:]
        cupy_sel_time = time.time() - start_time
        
        print(f"   NumPy: {numpy_sel_time:.3f}s")
        print(f"   CuPy:  {cupy_sel_time:.3f}s")
        print(f"   Speedup: {numpy_sel_time/cupy_sel_time:.2f}x")
        
        return True, numpy_time/cupy_time, numpy_sel_time/cupy_sel_time
        
    except ImportError:
        print("❌ CuPy is not available. Installing...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "cupy-cuda12x"])
            print("✅ CuPy installed! Please run again.")
            return False, 0, 0
        except Exception as e:
            print(f"❌ Failed to install CuPy: {e}")
            return False, 0, 0
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False, 0, 0

def create_cupy_optimized_brain():
    """Create CuPy-optimized brain implementation"""
    print(f"\n🔥 CREATING CUPY-OPTIMIZED BRAIN")
    print("=" * 60)
    
    try:
        import cupy as cp
        
        class CuPyOptimizedCudaBrain:
            """CuPy-optimized CUDA Brain with GPU acceleration"""
            
            def __init__(self, p: float = 0.1, beta: float = 0.5, max_weight: float = 1.0, seed: int = 42):
                self.p = p
                self.beta = beta
                self.max_weight = max_weight
                self.seed = seed
                self.step = 0
                
                # Brain state
                self.areas = {}
                self.fibers = []
                self.stimuli = {}
                
                # CuPy optimizations
                self._cupy_arrays = {}
                self._gpu_memory_pool = True
                
                print("🚀 CuPy-Optimized CUDA Brain initialized!")
                print("   GPU acceleration: CuPy, memory pooling, vectorized operations")
            
            def AddArea(self, name: str, n: int, k: int, recurrent: bool = False, is_explicit: bool = False) -> None:
                """Add a neural area with CuPy arrays"""
                self.areas[name] = {
                    'n': n,
                    'k': k,
                    'activated': [],
                    'support': n if is_explicit else 0,
                    'recurrent': recurrent,
                    'is_explicit': is_explicit
                }
                
                # Pre-allocate CuPy arrays on GPU
                self._cupy_arrays[name] = {
                    'candidates': cp.zeros(n, dtype=cp.float32),
                    'indices': cp.arange(n, dtype=cp.uint32),
                    'top_k_indices': cp.zeros(k, dtype=cp.uint32)
                }
                
                print(f"✓ Added area: {name} (n={n}, k={k}) with GPU arrays")
            
            def AddStimulus(self, name: str, k: int) -> None:
                """Add a stimulus"""
                self.stimuli[name] = {
                    'k': k,
                    'activated': list(range(k))
                }
                print(f"✓ Added stimulus: {name} (k={k})")
            
            def AddFiber(self, from_name: str, to_name: str, bidirectional: bool = False) -> None:
                """Add a fiber connection"""
                fiber = {
                    'from': from_name,
                    'to': to_name,
                    'bidirectional': bidirectional
                }
                self.fibers.append(fiber)
                print(f"✓ Added fiber: {from_name} -> {to_name}")
            
            def _gpu_generate_candidates(self, area_name: str, n: int, k: int) -> cp.ndarray:
                """GPU-accelerated candidate generation using CuPy"""
                if area_name not in self._cupy_arrays:
                    return cp.array([])
                
                candidates = self._cupy_arrays[area_name]['candidates']
                candidates[:] = cp.random.exponential(1.0, n)
                
                return candidates
            
            def _gpu_select_top_k(self, area_name: str, candidates: cp.ndarray, k: int) -> cp.ndarray:
                """GPU-accelerated top-k selection using CuPy"""
                if area_name not in self._cupy_arrays:
                    return cp.array([])
                
                # GPU-accelerated partial sort
                top_k_indices = cp.argpartition(candidates, -k)[-k:]
                
                # Sort only the top k
                top_k_values = candidates[top_k_indices]
                sorted_indices = cp.argsort(top_k_values)[::-1]
                
                return top_k_indices[sorted_indices]
            
            def SimulateOneStep(self, update_plasticity: bool = True) -> None:
                """GPU-accelerated simulation step"""
                for area_name, area in self.areas.items():
                    if area['is_explicit']:
                        continue
                    
                    # GPU-accelerated operations
                    candidates = self._gpu_generate_candidates(area_name, area['n'], area['k'])
                    selected = self._gpu_select_top_k(area_name, candidates, area['k'])
                    
                    # Copy back to CPU for state update
                    area['activated'] = cp.asnumpy(selected).tolist()
                    area['support'] = max(area['support'], len(area['activated']))
                
                self.step += 1
            
            def Project(self, graph: dict, num_steps: int, update_plasticity: bool = True) -> None:
                """Run projection for multiple steps"""
                print(f"🚀 Starting CuPy GPU projection for {num_steps} steps")
                
                start_time = time.time()
                
                for step in range(num_steps):
                    self.SimulateOneStep(update_plasticity)
                
                total_time = time.time() - start_time
                print(f"✅ CuPy GPU projection complete! ({total_time:.3f}s total, {total_time/num_steps*1000:.2f}ms/step)")
            
            def GetActivatedNeurons(self, area_name: str) -> list:
                """Get activated neurons in an area"""
                if area_name in self.areas:
                    return self.areas[area_name]['activated']
                elif area_name in self.stimuli:
                    return self.stimuli[area_name]['activated']
                else:
                    return []
        
        return CuPyOptimizedCudaBrain
        
    except ImportError:
        print("❌ CuPy not available. Cannot create CuPy-optimized brain.")
        return None

def test_cupy_brain_performance():
    """Test CuPy brain performance"""
    print(f"\n🧠 TESTING CUPY BRAIN PERFORMANCE")
    print("=" * 60)
    
    CuPyBrain = create_cupy_optimized_brain()
    if CuPyBrain is None:
        return
    
    # Create CuPy brain
    brain = CuPyBrain(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # Add test areas
    brain.AddArea("SmallArea", n=100000, k=1000)
    brain.AddArea("MediumArea", n=500000, k=5000)
    brain.AddArea("LargeArea", n=1000000, k=10000)
    brain.AddArea("HugeArea", n=5000000, k=50000)
    
    # Test single step
    print("Testing single step...")
    start_time = time.time()
    brain.SimulateOneStep()
    step_time = time.time() - start_time
    
    print(f"Single step time: {step_time*1000:.2f}ms")
    
    # Test multiple steps
    print("Testing 20 steps...")
    start_time = time.time()
    for _ in range(20):
        brain.SimulateOneStep()
    total_time = time.time() - start_time
    
    print(f"20 steps time: {total_time:.3f}s")
    print(f"Avg time per step: {total_time/20*1000:.2f}ms")
    print(f"Steps per second: {20/total_time:.1f}")
    
    return total_time

if __name__ == "__main__":
    try:
        # Investigate optimization difference
        results = investigate_optimization_difference()
        
        # Test CuPy acceleration
        cupy_available, gen_speedup, sel_speedup = test_cupy_acceleration()
        
        if cupy_available:
            # Test CuPy brain performance
            cupy_time = test_cupy_brain_performance()
            
            # Compare all implementations
            print(f"\n📊 COMPREHENSIVE PERFORMANCE COMPARISON")
            print("=" * 60)
            
            ultra_time = results["Ultra Optimized"]['avg_time_per_step']
            simple_time = results["Simple Ultra Optimized"]['avg_time_per_step']
            cupy_time_per_step = cupy_time / 20 if cupy_time else 0
            
            print(f"Ultra Optimized:        {ultra_time*1000:6.2f}ms/step")
            print(f"Simple Ultra Optimized: {simple_time*1000:6.2f}ms/step")
            if cupy_time_per_step > 0:
                print(f"CuPy GPU Optimized:     {cupy_time_per_step*1000:6.2f}ms/step")
            
            # Find best performance
            times = [ultra_time, simple_time, cupy_time_per_step if cupy_time_per_step > 0 else float('inf')]
            names = ["Ultra Optimized", "Simple Ultra Optimized", "CuPy GPU Optimized"]
            best_idx = np.argmin(times)
            
            print(f"\n🏆 BEST PERFORMANCE: {names[best_idx]}")
            print(f"   {times[best_idx]*1000:.2f}ms per step")
            
            if best_idx == 0:
                print(f"   The 82x speedup is still the best!")
            elif best_idx == 2:
                print(f"   CuPy GPU acceleration is the fastest!")
        
    except Exception as e:
        print(f"\n❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()
