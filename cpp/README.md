# C++ and accelerator work

`cpp/` contains lower-level implementation and accelerator experiments for the
assembly runtime.

Treat this directory as engineering infrastructure, not as the default Python
package API. The maintained package path remains `neural_assemblies/`.

## What Lives Here

- C++ implementation experiments
- Python binding experiments
- CUDA kernels and tests
- build scripts
- performance notes

## Build Requirements

- C++17 compiler
- Python development environment
- `pybind11`
- `numpy`
- CMake for some build paths

## Local Build Sketch

```bash
cd cpp
pip install pybind11 numpy
python setup.py build_ext --inplace
```

Some scripts may assume local compiler, CUDA, or Visual Studio configuration.
Use the performance tests as environment checks before treating benchmark
numbers as meaningful.

## Python Usage Sketch

```python
from neural_assemblies.core.brain_cpp import BrainCPP

brain = BrainCPP(p=0.05, beta=0.1, max_weight=10000.0, seed=7777)
brain.add_area("A", n=100000, k=317, beta=0.1)
brain.add_stimulus("stimA", k=317)
brain.add_fiber("stimA", "A")
brain.project({"stimA": ["A"]}, {})
```

## Benchmarking

Benchmark results depend on build flags, hardware, problem size, and which
engine path actually runs. Avoid writing fixed speedup claims without the
benchmark command, environment, and result artifact.

## Related Paths

- `tests/performance/`
- `cpp/cuda_kernels/`
- `docs/scientific_status.md`
- `docs/supported_surfaces.md`
