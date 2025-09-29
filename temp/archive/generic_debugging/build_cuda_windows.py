#!/usr/bin/env python3
"""
Windows-specific CUDA build script for RTX 4090
"""

import os
import sys
import subprocess
import time

def main():
    print("🚀 BUILDING CUDA NEURAL SIMULATION FOR RTX 4090")
    print("=" * 60)
    print()
    
    # CUDA environment
    cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    cuda_bin = os.path.join(cuda_home, "bin")
    cuda_lib = os.path.join(cuda_home, "lib", "x64")
    cuda_include = os.path.join(cuda_home, "include")
    
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['PATH'] = f"{cuda_bin};{cuda_lib};" + os.environ.get('PATH', '')
    
    nvcc_path = os.path.join(cuda_bin, "nvcc.exe")
    
    print(f"✓ CUDA Home: {cuda_home}")
    print(f"✓ NVCC Path: {nvcc_path}")
    print()
    
    # Check if files exist
    cuda_files = ["cuda_kernels.cu", "cuda_brain.cu", "cuda_brain.h"]
    for file in cuda_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    
    print()
    print("🔨 COMPILING CUDA KERNELS...")
    print("-" * 40)
    
    # Compile CUDA kernels first
    compile_cmd = [
        nvcc_path,
        "-c", "cuda_kernels.cu",
        "-o", "cuda_kernels.obj",
        "--compiler-options", "/MD",  # Use dynamic runtime
        "-O3",  # Optimization
        "--use_fast_math",  # Fast math
        "-Xptxas", "-O3",  # PTX optimization
        "--gpu-architecture=compute_75",  # RTX 4090 compatible
        "--gpu-code=sm_75,sm_80,sm_86,sm_89",  # Multiple architectures
        "-I", cuda_include,
        "-std=c++17"
    ]
    
    print("Compiling CUDA kernels...")
    print(" ".join(compile_cmd))
    print()
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ CUDA kernels compiled successfully!")
        else:
            print("❌ CUDA kernel compilation failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Compilation timed out")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False
    
    print()
    print("🔨 COMPILING CUDA BRAIN...")
    print("-" * 40)
    
    # Compile CUDA brain
    compile_cmd = [
        nvcc_path,
        "-c", "cuda_brain.cu",
        "-o", "cuda_brain.obj",
        "--compiler-options", "/MD",
        "-O3",
        "--use_fast_math",
        "-Xptxas", "-O3",
        "--gpu-architecture=compute_75",
        "--gpu-code=sm_75,sm_80,sm_86,sm_89",
        "-I", cuda_include,
        "-std=c++17"
    ]
    
    print("Compiling CUDA brain...")
    print(" ".join(compile_cmd))
    print()
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ CUDA brain compiled successfully!")
        else:
            print("❌ CUDA brain compilation failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Compilation timed out")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False
    
    print()
    print("🔗 LINKING CUDA OBJECTS...")
    print("-" * 40)
    
    # Link into shared library
    link_cmd = [
        nvcc_path,
        "--shared",
        "-o", "cuda_brain.dll",
        "cuda_kernels.obj",
        "cuda_brain.obj",
        "-lcublas",
        "-lcurand", 
        "-lcusparse",
        "-L", cuda_lib
    ]
    
    print("Linking CUDA library...")
    print(" ".join(link_cmd))
    print()
    
    try:
        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ CUDA library linked successfully!")
        else:
            print("❌ CUDA linking failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Linking timed out")
        return False
    except Exception as e:
        print(f"❌ Linking error: {e}")
        return False
    
    # Check output files
    output_files = ["cuda_kernels.obj", "cuda_brain.obj", "cuda_brain.dll"]
    print()
    print("📁 OUTPUT FILES:")
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} missing")
    
    print()
    print("🎉 CUDA BUILD COMPLETE!")
    print("=" * 60)
    print("✓ CUDA kernels compiled for RTX 4090")
    print("✓ GPU acceleration ready")
    print("✓ Next step: Create Python bindings")
    print()
    print("Your RTX 4090 is ready for neural simulation!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready for GPU-accelerated neural simulation!")
    else:
        print("\n❌ Build failed - check errors above")

