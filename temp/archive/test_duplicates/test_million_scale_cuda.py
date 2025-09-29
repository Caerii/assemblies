#!/usr/bin/env python3
"""
Million-scale CUDA brain performance test
"""

import time
import sys
from cuda_brain_python import CudaBrainPython

def test_million_scale():
    """Test million-scale neural simulation"""
    print("🚀 MILLION-SCALE CUDA BRAIN TEST")
    print("=" * 50)
    
    # Create CUDA brain with large parameters
    brain = CudaBrainPython(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # Add large areas (million-scale)
    print("🏗️  Building million-scale brain...")
    
    # Language areas with realistic sizes
    brain.AddArea("Wernicke", n=100000, k=1000)  # 100K neurons, 1K active
    brain.AddArea("Broca", n=100000, k=1000)     # 100K neurons, 1K active
    brain.AddArea("Arcuate", n=50000, k=500)     # 50K neurons, 500 active
    brain.AddArea("Angular", n=75000, k=750)     # 75K neurons, 750 active
    brain.AddArea("Supramarginal", n=60000, k=600) # 60K neurons, 600 active
    
    # Sensory areas
    brain.AddArea("Auditory", n=200000, k=2000)  # 200K neurons, 2K active
    brain.AddArea("Visual", n=300000, k=3000)    # 300K neurons, 3K active
    
    # Higher-order areas
    brain.AddArea("Prefrontal", n=150000, k=1500) # 150K neurons, 1.5K active
    brain.AddArea("Parietal", n=100000, k=1000)   # 100K neurons, 1K active
    brain.AddArea("Temporal", n=120000, k=1200)   # 120K neurons, 1.2K active
    
    # Add stimuli
    brain.AddStimulus("Speech", k=500)
    brain.AddStimulus("Visual", k=1000)
    brain.AddStimulus("Touch", k=300)
    
    # Add connections (simplified language network)
    connections = [
        ("Speech", "Auditory"),
        ("Visual", "Visual"),
        ("Touch", "Parietal"),
        ("Auditory", "Wernicke"),
        ("Wernicke", "Arcuate"),
        ("Arcuate", "Broca"),
        ("Broca", "Speech"),
        ("Visual", "Angular"),
        ("Angular", "Wernicke"),
        ("Wernicke", "Prefrontal"),
        ("Prefrontal", "Broca"),
        ("Parietal", "Supramarginal"),
        ("Supramarginal", "Wernicke"),
        ("Temporal", "Wernicke"),
        ("Temporal", "Broca")
    ]
    
    for from_area, to_area in connections:
        brain.AddFiber(from_area, to_area, bidirectional=True)
    
    # Log stats
    brain.LogGraphStats()
    
    total_neurons = sum(area['n'] for area in brain.areas.values())
    print(f"   Total neurons: {total_neurons:,}")
    print(f"   Active neurons per step: {sum(area['k'] for area in brain.areas.values()):,}")
    
    # Run large-scale simulation
    print(f"\n🧠 Running million-scale simulation...")
    start_time = time.time()
    
    # Run for 20 steps
    brain.Project({}, num_steps=20)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 MILLION-SCALE PERFORMANCE RESULTS:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Time per step: {total_time/20*1000:.2f}ms")
    print(f"   Neurons processed per second: {total_neurons*20/total_time:,.0f}")
    print(f"   Active neurons per second: {sum(area['k'] for area in brain.areas.values())*20/total_time:,.0f}")
    
    # Check results
    print(f"\n🧠 Final Activation State:")
    for area_name in sorted(brain.areas.keys()):
        activated = brain.GetActivatedNeurons(area_name)
        area_info = brain.GetAreaInfo(area_name)
        print(f"   {area_name:12}: {len(activated):4} neurons activated (target: {area_info['k']:4})")
    
    # Performance analysis
    neurons_per_ms = total_neurons / (total_time * 1000)
    print(f"\n🎯 Performance Analysis:")
    print(f"   Processing speed: {neurons_per_ms:.0f} neurons/ms")
    print(f"   Real-time factor: {total_neurons / (total_time * 1000):.1f}x")
    
    if total_time < 1.0:
        print(f"   ✅ SUB-SECOND PERFORMANCE! Million-scale simulation in {total_time:.3f}s")
    else:
        print(f"   ⚠️  Performance could be improved for real-time applications")
    
    return total_time

if __name__ == "__main__":
    try:
        test_million_scale()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
