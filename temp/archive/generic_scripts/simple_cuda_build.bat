@echo off
echo 🚀 SIMPLE CUDA BUILD FOR RTX 4090
echo =================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

echo Building simple CUDA test...
nvcc simple_cuda_test.cu -o simple_cuda_test.exe -O3 -arch=sm_75
if errorlevel 1 (
    echo ❌ Simple test failed
    exit /b 1
)

echo ✓ Simple test compiled successfully!

echo Running simple test...
simple_cuda_test.exe
if errorlevel 1 (
    echo ❌ Simple test execution failed  
    exit /b 1
)

echo ✓ Simple test passed!

echo.
echo Building basic CUDA kernels...
nvcc -c cuda_kernels_fixed.cu -o cuda_kernels.obj -O3 -arch=sm_75 -std=c++17
if errorlevel 1 (
    echo ❌ Kernel compilation failed
    exit /b 1
)

echo ✓ Kernels compiled successfully!

echo Creating CUDA library...
nvcc --shared -o cuda_kernels.dll cuda_kernels.obj -lcublas -lcurand -lcusparse
if errorlevel 1 (
    echo ❌ Library creation failed
    exit /b 1
)

echo.
echo ✅ BASIC CUDA BUILD COMPLETE!
echo =============================
echo Files created:
dir *.obj
dir *.dll
echo.
echo 🎉 Basic CUDA kernels ready for neural simulation!

