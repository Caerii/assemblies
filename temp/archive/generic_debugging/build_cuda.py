#!/usr/bin/env python3
"""
Build script for CUDA-accelerated brain simulation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_cuda_availability():
    """Check if CUDA is available on the system"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✓ CUDA compiler found:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CUDA compiler (nvcc) not found")
        print("Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-toolkit")
        return False

def check_cuda_libraries():
    """Check if required CUDA libraries are available"""
    libraries = ['cublas', 'curand', 'cusparse']
    missing = []
    
    for lib in libraries:
        try:
            result = subprocess.run(['pkg-config', '--exists', f'cuda-{lib}'], 
                                  capture_output=True, check=True)
            print(f"✓ {lib} library found")
        except subprocess.CalledProcessError:
            missing.append(lib)
            print(f"✗ {lib} library not found")
    
    return len(missing) == 0, missing

def build_cuda_extension():
    """Build the CUDA extension"""
    print("\n🔨 Building CUDA Brain Extension...")
    
    # Create build directory
    build_dir = Path("build_cuda")
    build_dir.mkdir(exist_ok=True)
    
    # Compile CUDA kernels
    cuda_files = [
        "cuda_kernels.cu",
        "cuda_brain.cu"
    ]
    
    object_files = []
    
    for cuda_file in cuda_files:
        if not Path(cuda_file).exists():
            print(f"✗ CUDA file not found: {cuda_file}")
            continue
            
        obj_file = build_dir / f"{cuda_file}.o"
        print(f"Compiling {cuda_file}...")
        
        cmd = [
            'nvcc',
            '-c', cuda_file,
            '-o', str(obj_file),
            '--compiler-options', '-fPIC',
            '-O3',
            '--use_fast_math',
            '-Xptxas', '-O3',
            '--gpu-architecture=compute_75',  # Support RTX 20xx and newer
            '--gpu-code=sm_75,sm_80,sm_86,sm_89'  # Multiple GPU architectures
        ]
        
        try:
            subprocess.run(cmd, check=True)
            object_files.append(str(obj_file))
            print(f"✓ Compiled {cuda_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to compile {cuda_file}: {e}")
            return False
    
    # Create shared library
    if object_files:
        print("Creating shared library...")
        cmd = [
            'nvcc',
            '--shared',
            '-o', str(build_dir / 'cuda_brain.so'),
            *object_files,
            '-lcublas',
            '-lcurand',
            '-lcusparse'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ CUDA extension built successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create shared library: {e}")
            return False
    
    return False

def create_cuda_setup():
    """Create setup.py for CUDA extension"""
    setup_content = '''
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os

# CUDA paths
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
CUDA_INCLUDE = os.path.join(CUDA_HOME, 'include')
CUDA_LIB = os.path.join(CUDA_HOME, 'lib64')

# Define the extension
ext_modules = [
    Pybind11Extension(
        "cuda_brain",
        [
            "cuda_brain.cu",
            "cuda_kernels.cu",
            "cuda_pybind11_wrapper.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            CUDA_INCLUDE,
            "."
        ],
        libraries=['cublas', 'curand', 'cusparse', 'cudart'],
        library_dirs=[CUDA_LIB],
        language='c++',
        cxx_std=17,
        define_macros=[('VERSION_INFO', '"dev"')],
    ),
]

setup(
    name="cuda_brain",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
'''
    
    with open('setup_cuda.py', 'w') as f:
        f.write(setup_content)
    
    print("✓ Created setup_cuda.py")

def create_cuda_wrapper():
    """Create Python wrapper for CUDA brain"""
    wrapper_content = '''
import pybind11
import numpy as np
from cuda_brain import CudaBrain, CudaArea, CudaFiber

class BrainCUDA:
    """Python wrapper for CUDA-accelerated brain simulation"""
    
    def __init__(self, p, beta, max_weight, seed=0):
        self._cuda_brain = CudaBrain(p, beta, max_weight, seed)
        self.p = p
        self.beta = beta
        self.max_weight = max_weight
        self.seed = seed
        
        # Python-side compatibility
        self.areas = {}
        self.area_by_name = {}
        self.fibers = []
        self.step = 0
    
    def add_area(self, name, n, k, beta=None, recurrent=True, is_explicit=False):
        """Add a neural area"""
        cuda_area = self._cuda_brain.AddArea(name, n, k, recurrent, is_explicit)
        
        # Create Python-compatible area object
        class AreaWrapper:
            def __init__(self, cuda_area_ref, name, n, k):
                self._cuda_area = cuda_area_ref
                self.name = name
                self.n = n
                self.k = k
                self.support = 0
                self.activated = np.array([], dtype=np.uint32)
                self.saved_winners = []
                self.saved_w = []
            
            @property
            def winners(self):
                # This would sync from GPU and return activated neurons
                return self.activated
        
        py_area = AreaWrapper(cuda_area, name, n, k)
        self.areas[name] = py_area
        self.area_by_name[name] = py_area
        return py_area
    
    def add_stimulus(self, name, k):
        """Add a stimulus"""
        self._cuda_brain.AddStimulus(name, k)
    
    def add_fiber(self, from_area, to_area, bidirectional=False):
        """Add a fiber connection"""
        self._cuda_brain.AddFiber(from_area, to_area, bidirectional)
    
    def project(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """Project activity through the network"""
        # Convert to C++ format
        graph = {}
        for stim, dst_areas in areas_by_stim.items():
            graph[stim] = dst_areas
        for src_area, dst_areas in dst_areas_by_src_area.items():
            if src_area not in graph:
                graph[src_area] = []
            graph[src_area].extend(dst_areas)
        
        # Run simulation
        self._cuda_brain.Project(graph, 1, update_plasticity=True)
        self.step += 1
    
    def get_overlap(self, area_name, assembly_index_1, assembly_index_2):
        """Calculate overlap between two assemblies"""
        if area_name not in self.areas:
            raise ValueError(f"Area {area_name} not found")
        
        winners = self.areas[area_name].saved_winners
        if assembly_index_1 >= len(winners) or assembly_index_2 >= len(winners):
            raise IndexError("Assembly index out of bounds")
        
        set1 = set(winners[assembly_index_1])
        set2 = set(winners[assembly_index_2])
        return len(set1.intersection(set2))
    
    def set_log_level(self, level):
        """Set logging level"""
        self._cuda_brain.SetLogLevel(level)
    
    def log_performance_stats(self):
        """Log GPU performance statistics"""
        self._cuda_brain.LogPerformanceStats()
'''
    
    with open('cuda_pybind11_wrapper.cpp', 'w') as f:
        f.write('''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuda_brain.h"

namespace py = pybind11;
namespace nemo = nemo::cuda;

PYBIND11_MODULE(cuda_brain, m) {
    m.doc() = "CUDA-accelerated neural assembly simulation";
    
    py::class_<nemo::CudaArea>(m, "CudaArea")
        .def_property_readonly("index", [](const nemo::CudaArea& a) { return a.index; })
        .def_property_readonly("n", [](const nemo::CudaArea& a) { return a.n; })
        .def_property_readonly("k", [](const nemo::CudaArea& a) { return a.k; })
        .def_property_readonly("support", [](const nemo::CudaArea& a) { return a.support; })
        .def_property_readonly("is_fixed", [](const nemo::CudaArea& a) { return a.is_fixed; });
    
    py::class_<nemo::CudaBrain>(m, "CudaBrain")
        .def(py::init<float, float, float, uint32_t>(),
             py::arg("p"), py::arg("beta"), py::arg("max_weight"), py::arg("seed"))
        .def("AddArea", &nemo::CudaBrain::AddArea,
             py::arg("name"), py::arg("n"), py::arg("k"),
             py::arg("recurrent") = true, py::arg("is_explicit") = false,
             py::return_value_policy::reference_internal)
        .def("AddStimulus", &nemo::CudaBrain::AddStimulus, py::arg("name"), py::arg("k"))
        .def("AddFiber", &nemo::CudaBrain::AddFiber,
             py::arg("from"), py::arg("to"), py::arg("bidirectional") = false)
        .def("GetArea", (nemo::CudaArea& (nemo::CudaBrain::*)(const std::string&)) &nemo::CudaBrain::GetArea,
             py::arg("name"), py::return_value_policy::reference_internal)
        .def("Project", &nemo::CudaBrain::Project,
             py::arg("graph"), py::arg("num_steps"), py::arg("update_plasticity") = true)
        .def("SimulateOneStep", &nemo::CudaBrain::SimulateOneStep, py::arg("update_plasticity") = true)
        .def("SetLogLevel", &nemo::CudaBrain::SetLogLevel, py::arg("level"))
        .def("LogPerformanceStats", &nemo::CudaBrain::LogPerformanceStats);
}
''')
    
    with open('cuda_brain_wrapper.py', 'w') as f:
        f.write(wrapper_content)
    
    print("✓ Created CUDA wrapper files")

def main():
    """Main build process"""
    print("🚀 CUDA Brain Simulation Builder")
    print("=" * 50)
    
    # Check prerequisites
    if not check_cuda_availability():
        print("\n❌ CUDA not available. Cannot build CUDA extension.")
        return False
    
    cuda_available, missing_libs = check_cuda_libraries()
    if not cuda_available:
        print(f"\n❌ Missing CUDA libraries: {missing_libs}")
        print("Please install CUDA libraries or set CUDA_HOME environment variable")
        return False
    
    # Create wrapper files
    create_cuda_wrapper()
    create_cuda_setup()
    
    # Build CUDA extension
    if build_cuda_extension():
        print("\n🎉 CUDA Brain Extension built successfully!")
        print("\nNext steps:")
        print("1. Test the CUDA extension:")
        print("   python -c \"import cuda_brain; print('CUDA Brain loaded successfully!')\"")
        print("2. Run performance benchmarks:")
        print("   python test_cuda_performance.py")
        print("3. Use in your simulations:")
        print("   from cuda_brain_wrapper import BrainCUDA")
        return True
    else:
        print("\n❌ Failed to build CUDA extension")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
