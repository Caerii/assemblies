# Universal Brain Simulator - Examples

This directory contains comprehensive examples demonstrating how to use the Universal Brain Simulator's lightweight thin client.

## Quick Start

```python
from universal_brain_simulator.client import BrainSimulator

# Simple usage
sim = BrainSimulator(neurons=1000000, active_percentage=0.01)
results = sim.run(steps=100)
```

## Examples Overview

### 1. `quick_start.py` - Getting Started
- **Purpose**: Simplest way to get started with the simulator
- **Features**: Basic simulation, convenience functions
- **Best for**: First-time users, quick testing

```bash
python quick_start.py
```

### 2. `benchmark_example.py` - Performance Benchmarking
- **Purpose**: Compare different configurations and find optimal settings
- **Features**: Single benchmarks, configuration comparison, best configuration detection
- **Best for**: Performance optimization, hardware evaluation

```bash
python benchmark_example.py
```

### 3. `profiling_example.py` - Detailed Performance Analysis
- **Purpose**: Deep dive into performance characteristics
- **Features**: Detailed profiling, custom callbacks, memory analysis
- **Best for**: Performance debugging, optimization analysis

```bash
python profiling_example.py
```

### 4. `scaling_example.py` - Scaling Analysis
- **Purpose**: Understand how performance scales with different parameters
- **Features**: Neuron count scaling, active percentage analysis, area count testing
- **Best for**: Capacity planning, scaling decisions

```bash
python scaling_example.py
```

### 5. `advanced_example.py` - Advanced Usage Patterns
- **Purpose**: Complex scenarios and custom monitoring
- **Features**: Custom callbacks, memory management, multiple runs, result saving
- **Best for**: Production use, custom integrations

```bash
python advanced_example.py
```

### 6. `performance_comparison.py` - Comprehensive Comparison
- **Purpose**: Compare all available configurations
- **Features**: Full configuration matrix, speedup analysis, memory efficiency
- **Best for**: Hardware evaluation, configuration selection

```bash
python performance_comparison.py
```

## Configuration Options

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

## Common Usage Patterns

### Quick Simulation
```python
from universal_brain_simulator.client import quick_sim

# Run a quick simulation
results = quick_sim(neurons=500000, steps=50, optimized=True)
print(f"Performance: {results['summary']['steps_per_second']:.1f} steps/sec")
```

### Benchmarking
```python
from universal_brain_simulator.client import quick_benchmark

# Run a quick benchmark
results = quick_benchmark(neurons=1000000, optimized=True)
print(f"Benchmark: {results['performance']['steps_per_second']:.1f} steps/sec")
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

### Custom Monitoring
```python
def my_callback(step, step_time):
    if step % 10 == 0:
        print(f"Step {step}: {step_time*1000:.2f}ms")

sim = BrainSimulator(neurons=1000000)
sim.run_with_callback(steps=100, callback=my_callback, callback_interval=1)
```

## Performance Tips

1. **Use Optimized CUDA**: Always use `use_optimized_cuda=True` for best performance
2. **Enable Memory Efficiency**: Use `memory_efficient=True` and `sparse_mode=True`
3. **Warmup Runs**: Use warmup steps in benchmarks for accurate results
4. **Profile First**: Run profiling examples to understand your hardware capabilities
5. **Scale Gradually**: Start with smaller neuron counts and scale up

## Expected Performance

### Optimized CUDA (Recommended)
- **1M neurons**: ~100-500 steps/sec
- **2M neurons**: ~50-250 steps/sec
- **5M neurons**: ~20-100 steps/sec

### Original CUDA
- **1M neurons**: ~50-200 steps/sec
- **2M neurons**: ~25-100 steps/sec
- **5M neurons**: ~10-50 steps/sec

### CuPy Only
- **1M neurons**: ~20-100 steps/sec
- **2M neurons**: ~10-50 steps/sec

### CPU Only (NumPy)
- **1M neurons**: ~5-20 steps/sec
- **2M neurons**: ~2-10 steps/sec

*Performance varies significantly based on hardware specifications.*

## Troubleshooting

### Common Issues
1. **CUDA not found**: Ensure CUDA drivers and toolkit are installed
2. **CuPy import error**: Install CuPy with `pip install cupy-cuda11x` (or appropriate version)
3. **Memory errors**: Reduce neuron count or enable memory-efficient mode
4. **Slow performance**: Check if optimized CUDA kernels are being used

### Getting Help
- Check the main README for detailed installation instructions
- Run the examples to verify your setup
- Use the profiling examples to diagnose performance issues
