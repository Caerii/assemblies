# Universal Brain Simulator v2.0.0

A modular, high-performance brain simulation system with multiple backends (CUDA, CuPy, NumPy) and optimization levels.

## 🚀 Quick Start

### Lightweight Client (Recommended for most users)

```python
from universal_brain_simulator.client import BrainSimulator

# Simple usage
sim = BrainSimulator(neurons=1000000, active_percentage=0.01)
results = sim.run(steps=100)
print(f"Performance: {results['summary']['steps_per_second']:.1f} steps/sec")
```

### Even Simpler with Convenience Functions

```python
from universal_brain_simulator.client import quick_sim, quick_benchmark

# Quick simulation
results = quick_sim(neurons=500000, steps=50, optimized=True)

# Quick benchmark
benchmark = quick_benchmark(neurons=1000000, optimized=True)
```

## 📁 Architecture

### Modular Components
- **`UniversalBrainSimulator`**: Main orchestrator class
- **`BrainSimulator`**: Lightweight thin client (recommended)
- **`SimulationConfig`**: Configuration management
- **`PerformanceMonitor`**: Performance tracking
- **`CUDAManager`**: CUDA kernel management
- **`MemoryManager`**: Memory allocation and pooling
- **`AreaManager`**: Brain area management
- **`SimulationEngine`**: Core simulation logic

### File Structure
```
universal_brain_simulator/
├── __init__.py                    # Package initialization
├── client.py                      # Lightweight thin client
├── config.py                      # Configuration classes
├── metrics.py                     # Performance monitoring
├── cuda_manager.py                # CUDA kernel management
├── memory_manager.py              # Memory management
├── area_manager.py                # Brain area management
├── simulation_engine.py           # Core simulation logic
├── universal_brain_simulator.py   # Main orchestrator
├── utils.py                       # Utility functions
├── examples/                      # Example scripts
│   ├── quick_start.py            # Getting started
│   ├── benchmark_example.py      # Performance benchmarking
│   ├── profiling_example.py      # Detailed profiling
│   ├── scaling_example.py        # Scaling analysis
│   ├── advanced_example.py       # Advanced usage
│   ├── performance_comparison.py # Configuration comparison
│   └── README.md                 # Examples documentation
└── monolithic_old/               # Archived monolithic version
    └── universal_brain_simulator.py
```

## 🎯 Key Features

### Performance Options
- **Optimized CUDA**: O(N log K) algorithms for maximum performance
- **Original CUDA**: O(N²) algorithms for compatibility
- **CuPy Only**: GPU acceleration without custom kernels
- **CPU Only**: NumPy fallback for any hardware

### Memory Management
- **Memory Efficient**: Optimized memory usage patterns
- **Sparse Mode**: Only allocate memory for active neurons
- **Memory Pooling**: Pre-allocated arrays for better performance
- **Dynamic Allocation**: Automatic memory management

### Monitoring & Profiling
- **Real-time Performance**: Steps/sec, neurons/sec, memory usage
- **Detailed Profiling**: Step-by-step performance analysis
- **Custom Callbacks**: Monitor simulation progress
- **JSON Export**: Save results for analysis

## 📊 Performance Expectations

### Optimized CUDA (Recommended)
- **1M neurons**: ~100-500 steps/sec
- **2M neurons**: ~50-250 steps/sec
- **5M neurons**: ~20-100 steps/sec

### Original CUDA
- **1M neurons**: ~50-200 steps/sec
- **2M neurons**: ~25-100 steps/sec

### CuPy Only
- **1M neurons**: ~20-100 steps/sec

### CPU Only (NumPy)
- **1M neurons**: ~5-20 steps/sec

*Performance varies based on hardware specifications.*

## 🧪 Examples

### 1. Quick Start
```bash
python universal_brain_simulator/examples/quick_start.py
```

### 2. Benchmarking
```bash
python universal_brain_simulator/examples/benchmark_example.py
```

### 3. Performance Comparison
```bash
python universal_brain_simulator/examples/performance_comparison.py
```

### 4. Scaling Analysis
```bash
python universal_brain_simulator/examples/scaling_example.py
```

### 5. Advanced Usage
```bash
python universal_brain_simulator/examples/advanced_example.py
```

### 6. Detailed Profiling
```bash
python universal_brain_simulator/examples/profiling_example.py
```

## 🔧 Configuration Options

### Basic Parameters
- `neurons`: Total number of neurons (default: 1,000,000)
- `active_percentage`: Percentage of active neurons (default: 0.01)
- `areas`: Number of brain areas (default: 5)
- `seed`: Random seed for reproducibility (default: 42)

### Performance Options
- `use_gpu`: Enable GPU acceleration (default: True)
- `use_optimized_cuda`: Use optimized CUDA kernels O(N log K) (default: True)
- `memory_efficient`: Enable memory-efficient mode (default: True)
- `sparse_mode`: Use sparse memory representation (default: True)
- `enable_profiling`: Enable performance profiling (default: True)

## 🚀 Usage Patterns

### Simple Simulation
```python
from universal_brain_simulator.client import BrainSimulator

sim = BrainSimulator(neurons=1000000, active_percentage=0.01)
results = sim.run(steps=100)
```

### Benchmarking
```python
benchmark_results = sim.benchmark(warmup_steps=5, measure_steps=10)
```

### Custom Monitoring
```python
def my_callback(step, step_time):
    if step % 10 == 0:
        print(f"Step {step}: {step_time*1000:.2f}ms")

sim.run_with_callback(steps=100, callback=my_callback, callback_interval=1)
```

### Configuration Comparison
```python
from universal_brain_simulator.client import compare_configurations

configs = [
    {'neurons': 1000000, 'use_optimized_cuda': True},
    {'neurons': 1000000, 'use_optimized_cuda': False}
]

results = compare_configurations(configs, steps=100)
```

## 🏗️ Advanced Usage

### Full Modular Access
```python
from universal_brain_simulator import (
    UniversalBrainSimulator, SimulationConfig, 
    CUDAManager, MemoryManager, AreaManager
)

config = SimulationConfig(
    n_neurons=1000000,
    active_percentage=0.01,
    use_optimized_kernels=True
)

simulator = UniversalBrainSimulator(config)
simulator.simulate(n_steps=100)
```

### Custom Components
```python
from universal_brain_simulator import SimulationConfig, CUDAManager

config = SimulationConfig(n_neurons=1000000)
cuda_manager = CUDAManager(config)
cuda_manager.load_kernels()
```

## 📈 Performance Tips

1. **Use Optimized CUDA**: Always use `use_optimized_cuda=True` for best performance
2. **Enable Memory Efficiency**: Use `memory_efficient=True` and `sparse_mode=True`
3. **Warmup Runs**: Use warmup steps in benchmarks for accurate results
4. **Profile First**: Run profiling examples to understand your hardware capabilities
5. **Scale Gradually**: Start with smaller neuron counts and scale up

## 🔍 Troubleshooting

### Common Issues
1. **CUDA not found**: Ensure CUDA drivers and toolkit are installed
2. **CuPy import error**: Install CuPy with `pip install cupy-cuda11x` (or appropriate version)
3. **Memory errors**: Reduce neuron count or enable memory-efficient mode
4. **Slow performance**: Check if optimized CUDA kernels are being used

### Getting Help
- Check the examples in the `examples/` directory
- Run the test script: `python test_client.py`
- Use the profiling examples to diagnose performance issues

## 🎉 What's New in v2.0.0

- **Modular Architecture**: Clean separation of concerns with focused components
- **Lightweight Client**: Simple, high-level interface for easy usage
- **Multiple Examples**: Comprehensive examples for different use cases
- **Performance Monitoring**: Advanced profiling and monitoring capabilities
- **Memory Optimization**: Efficient memory management and pooling
- **CUDA Optimization**: O(N log K) algorithms for maximum performance
- **Cross-Platform**: Works on Windows, Linux, and macOS

## 📝 License

This project is part of the Universal Brain Simulator system.

---

**Ready to get started?** Try the quick start example:
```bash
python universal_brain_simulator/examples/quick_start.py
```
