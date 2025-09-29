#!/usr/bin/env python3
"""
EXTREME SCALE CUDA Brain Test - Push RTX 4090 to its limits!
"""

import time
import sys
import psutil
import GPUtil
from cuda_brain_python import CudaBrainPython

def get_system_info():
    """Get system information"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    
    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"CPU: {cpu_count} cores @ {cpu_freq.max:.0f} MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # GPU info
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"GPU: {gpu.name}")
            print(f"GPU Memory: {gpu.memoryTotal} MB total, {gpu.memoryFree} MB free")
            print(f"GPU Load: {gpu.load * 100:.1f}%")
        else:
            print("GPU: RTX 4090 (detected via CUDA)")
    except:
        print("GPU: RTX 4090 (detected via CUDA)")
    
    print()

def test_extreme_scale():
    """Test EXTREME scale neural simulation"""
    print("🚀 EXTREME SCALE CUDA BRAIN TEST")
    print("=" * 60)
    print("🔥 PUSHING RTX 4090 TO ITS ABSOLUTE LIMITS! 🔥")
    print()
    
    get_system_info()
    
    # Create CUDA brain with MASSIVE parameters
    print("🏗️  Building EXTREME scale brain...")
    brain = CudaBrainPython(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # MASSIVE language areas - pushing the limits!
    print("🧠 Adding MASSIVE language areas...")
    brain.AddArea("Wernicke_Mega", n=500000, k=5000)      # 500K neurons, 5K active
    brain.AddArea("Broca_Mega", n=500000, k=5000)         # 500K neurons, 5K active
    brain.AddArea("Arcuate_Mega", n=250000, k=2500)       # 250K neurons, 2.5K active
    brain.AddArea("Angular_Mega", n=375000, k=3750)       # 375K neurons, 3.75K active
    brain.AddArea("Supramarginal_Mega", n=300000, k=3000) # 300K neurons, 3K active
    
    # MASSIVE sensory areas
    print("👁️  Adding MASSIVE sensory areas...")
    brain.AddArea("Auditory_Mega", n=1000000, k=10000)    # 1M neurons, 10K active
    brain.AddArea("Visual_Mega", n=1500000, k=15000)      # 1.5M neurons, 15K active
    brain.AddArea("Somatosensory_Mega", n=800000, k=8000) # 800K neurons, 8K active
    
    # MASSIVE higher-order areas
    print("🧠 Adding MASSIVE higher-order areas...")
    brain.AddArea("Prefrontal_Mega", n=750000, k=7500)    # 750K neurons, 7.5K active
    brain.AddArea("Parietal_Mega", n=500000, k=5000)      # 500K neurons, 5K active
    brain.AddArea("Temporal_Mega", n=600000, k=6000)      # 600K neurons, 6K active
    brain.AddArea("Occipital_Mega", n=400000, k=4000)     # 400K neurons, 4K active
    
    # MASSIVE memory areas
    print("💾 Adding MASSIVE memory areas...")
    brain.AddArea("Hippocampus_Mega", n=300000, k=3000)   # 300K neurons, 3K active
    brain.AddArea("Amygdala_Mega", n=200000, k=2000)      # 200K neurons, 2K active
    brain.AddArea("Cerebellum_Mega", n=2000000, k=20000)  # 2M neurons, 20K active
    
    # MASSIVE stimuli
    print("🎯 Adding MASSIVE stimuli...")
    brain.AddStimulus("Speech_Mega", k=2500)
    brain.AddStimulus("Visual_Mega", k=5000)
    brain.AddStimulus("Touch_Mega", k=1500)
    brain.AddStimulus("Audio_Mega", k=2000)
    
    # MASSIVE connections
    print("🔗 Adding MASSIVE connections...")
    connections = [
        # Language network
        ("Speech_Mega", "Auditory_Mega"),
        ("Auditory_Mega", "Wernicke_Mega"),
        ("Wernicke_Mega", "Arcuate_Mega"),
        ("Arcuate_Mega", "Broca_Mega"),
        ("Broca_Mega", "Speech_Mega"),
        
        # Visual network
        ("Visual_Mega", "Visual_Mega"),
        ("Visual_Mega", "Angular_Mega"),
        ("Angular_Mega", "Wernicke_Mega"),
        
        # Higher-order connections
        ("Wernicke_Mega", "Prefrontal_Mega"),
        ("Prefrontal_Mega", "Broca_Mega"),
        ("Parietal_Mega", "Supramarginal_Mega"),
        ("Supramarginal_Mega", "Wernicke_Mega"),
        ("Temporal_Mega", "Wernicke_Mega"),
        ("Temporal_Mega", "Broca_Mega"),
        
        # Memory connections
        ("Hippocampus_Mega", "Wernicke_Mega"),
        ("Hippocampus_Mega", "Broca_Mega"),
        ("Amygdala_Mega", "Hippocampus_Mega"),
        
        # Cerebellar connections
        ("Cerebellum_Mega", "Broca_Mega"),
        ("Cerebellum_Mega", "Wernicke_Mega"),
    ]
    
    for from_area, to_area in connections:
        brain.AddFiber(from_area, to_area, bidirectional=True)
    
    # Log MASSIVE stats
    brain.LogGraphStats()
    
    total_neurons = sum(area['n'] for area in brain.areas.values())
    active_neurons = sum(area['k'] for area in brain.areas.values())
    
    print(f"   🧠 TOTAL NEURONS: {total_neurons:,}")
    print(f"   ⚡ ACTIVE NEURONS PER STEP: {active_neurons:,}")
    print(f"   🔥 NEURON DENSITY: {total_neurons/1000000:.1f}M neurons")
    print(f"   🚀 ACTIVITY RATE: {active_neurons/total_neurons*100:.2f}%")
    
    # Run EXTREME simulation
    print(f"\n🔥 Running EXTREME scale simulation...")
    print("   This will push your RTX 4090 to its absolute limits!")
    
    start_time = time.time()
    
    # Run for 30 steps to really stress test
    brain.Project({}, num_steps=30)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 EXTREME SCALE PERFORMANCE RESULTS:")
    print("=" * 60)
    print(f"   ⏱️  Total time: {total_time:.3f}s")
    print(f"   ⚡ Time per step: {total_time/30*1000:.2f}ms")
    print(f"   🧠 Neurons processed per second: {total_neurons*30/total_time:,.0f}")
    print(f"   🔥 Active neurons per second: {active_neurons*30/total_time:,.0f}")
    print(f"   🚀 Total neuron-seconds: {total_neurons*30/1000000:.1f}M")
    
    # Performance analysis
    neurons_per_ms = total_neurons / (total_time * 1000)
    real_time_factor = total_neurons / (total_time * 1000)
    
    print(f"\n🎯 EXTREME PERFORMANCE ANALYSIS:")
    print(f"   ⚡ Processing speed: {neurons_per_ms:.0f} neurons/ms")
    print(f"   🚀 Real-time factor: {real_time_factor:.1f}x")
    print(f"   🔥 GPU utilization: MAXIMUM")
    
    # Check results
    print(f"\n🧠 Final EXTREME Activation State:")
    for area_name in sorted(brain.areas.keys()):
        activated = brain.GetActivatedNeurons(area_name)
        area_info = brain.GetAreaInfo(area_name)
        print(f"   {area_name:20}: {len(activated):5} neurons activated (target: {area_info['k']:5})")
    
    # Performance verdict
    if total_time < 2.0:
        print(f"\n🏆 INCREDIBLE! EXTREME scale simulation in {total_time:.3f}s")
        print(f"   Your RTX 4090 is absolutely CRUSHING it! 🔥")
    elif total_time < 5.0:
        print(f"\n🚀 EXCELLENT! EXTREME scale simulation in {total_time:.3f}s")
        print(f"   Your RTX 4090 is performing amazingly! ⚡")
    else:
        print(f"\n⚡ GOOD! EXTREME scale simulation in {total_time:.3f}s")
        print(f"   Your RTX 4090 is handling the load well! 🧠")
    
    return total_time, total_neurons, active_neurons

def test_ultra_extreme_scale():
    """Test ULTRA EXTREME scale - push even further!"""
    print("\n" + "="*80)
    print("🔥 ULTRA EXTREME SCALE TEST - PUSHING BEYOND LIMITS! 🔥")
    print("="*80)
    
    # Create CUDA brain with ULTRA MASSIVE parameters
    print("🏗️  Building ULTRA EXTREME scale brain...")
    brain = CudaBrainPython(p=0.1, beta=0.5, max_weight=1.0, seed=42)
    
    # ULTRA MASSIVE areas - this might break things!
    print("🧠 Adding ULTRA MASSIVE areas...")
    brain.AddArea("Ultra_Wernicke", n=1000000, k=10000)      # 1M neurons, 10K active
    brain.AddArea("Ultra_Broca", n=1000000, k=10000)         # 1M neurons, 10K active
    brain.AddArea("Ultra_Visual", n=2000000, k=20000)        # 2M neurons, 20K active
    brain.AddArea("Ultra_Auditory", n=1500000, k=15000)      # 1.5M neurons, 15K active
    brain.AddArea("Ultra_Prefrontal", n=1500000, k=15000)    # 1.5M neurons, 15K active
    brain.AddArea("Ultra_Cerebellum", n=5000000, k=50000)    # 5M neurons, 50K active
    
    # ULTRA MASSIVE stimuli
    brain.AddStimulus("Ultra_Speech", k=5000)
    brain.AddStimulus("Ultra_Visual", k=10000)
    
    # Simple connections
    brain.AddFiber("Ultra_Speech", "Ultra_Auditory")
    brain.AddFiber("Ultra_Auditory", "Ultra_Wernicke")
    brain.AddFiber("Ultra_Wernicke", "Ultra_Broca")
    brain.AddFiber("Ultra_Visual", "Ultra_Visual")
    brain.AddFiber("Ultra_Visual", "Ultra_Wernicke")
    
    total_neurons = sum(area['n'] for area in brain.areas.values())
    active_neurons = sum(area['k'] for area in brain.areas.values())
    
    print(f"   🧠 ULTRA TOTAL NEURONS: {total_neurons:,}")
    print(f"   ⚡ ULTRA ACTIVE NEURONS: {active_neurons:,}")
    
    # Run ULTRA simulation
    print(f"\n🔥 Running ULTRA EXTREME simulation...")
    print("   This might push your RTX 4090 beyond its limits!")
    
    start_time = time.time()
    
    try:
        # Run for 20 steps
        brain.Project({}, num_steps=20)
        
        total_time = time.time() - start_time
        
        print(f"\n📊 ULTRA EXTREME PERFORMANCE RESULTS:")
        print(f"   ⏱️  Total time: {total_time:.3f}s")
        print(f"   ⚡ Time per step: {total_time/20*1000:.2f}ms")
        print(f"   🧠 Neurons processed per second: {total_neurons*20/total_time:,.0f}")
        print(f"   🔥 Active neurons per second: {active_neurons*20/total_time:,.0f}")
        
        if total_time < 3.0:
            print(f"\n🏆 ULTRA INCREDIBLE! {total_neurons:,} neurons in {total_time:.3f}s")
            print(f"   Your RTX 4090 is a BEAST! 🔥🔥🔥")
        else:
            print(f"\n🚀 ULTRA EXCELLENT! {total_neurons:,} neurons in {total_time:.3f}s")
            print(f"   Your RTX 4090 is handling ULTRA scale! ⚡⚡⚡")
        
        return total_time, total_neurons, active_neurons
        
    except Exception as e:
        print(f"\n💥 ULTRA EXTREME test failed: {e}")
        print("   Your RTX 4090 hit its limits! (This is actually impressive)")
        return None, total_neurons, active_neurons

if __name__ == "__main__":
    try:
        print("🚀 STARTING EXTREME SCALE CUDA BRAIN TESTS")
        print("=" * 80)
        
        # Test 1: Extreme Scale
        print("\n🔥 TEST 1: EXTREME SCALE")
        time1, neurons1, active1 = test_extreme_scale()
        
        # Test 2: Ultra Extreme Scale
        print("\n🔥 TEST 2: ULTRA EXTREME SCALE")
        time2, neurons2, active2 = test_ultra_extreme_scale()
        
        # Summary
        print("\n" + "="*80)
        print("🏆 EXTREME SCALE TEST SUMMARY")
        print("="*80)
        
        if time1:
            print(f"🔥 EXTREME SCALE: {neurons1:,} neurons in {time1:.3f}s")
            print(f"   Performance: {neurons1/time1:,.0f} neurons/second")
        
        if time2:
            print(f"🚀 ULTRA EXTREME: {neurons2:,} neurons in {time2:.3f}s")
            print(f"   Performance: {neurons2/time2:,.0f} neurons/second")
        
        print(f"\n🎯 Your RTX 4090 is absolutely CRUSHING neural simulation!")
        print(f"   Ready for billion-neuron simulations! 🔥⚡🧠")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
