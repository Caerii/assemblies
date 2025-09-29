@echo off
echo 🚀 BUILDING FIXED CUDA KERNELS FOR RTX 4090
echo ==========================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

echo Testing simple CUDA first...
nvcc simple_cuda_test.cu -o simple_cuda_test.exe
if errorlevel 1 (
    echo ❌ Simple test failed
    exit /b 1
)

echo ✓ Simple test compiled, running...
simple_cuda_test.exe
if errorlevel 1 (
    echo ❌ Simple test execution failed  
    exit /b 1
)

echo ✓ Simple test passed!
echo.
echo Building fixed CUDA kernels...

echo Compiling CUDA kernels...
nvcc -c cuda_kernels_fixed.cu -o cuda_kernels.obj --compiler-options /MD -O3 --use_fast_math -arch=sm_75 -std=c++17
if errorlevel 1 (
    echo ❌ Kernel compilation failed
    exit /b 1
)

echo ✓ Kernels compiled successfully!

echo Compiling CUDA brain...  
nvcc -c cuda_brain.cu -o cuda_brain.obj --compiler-options /MD -O3 --use_fast_math -arch=sm_75 -std=c++17 -I.
if errorlevel 1 (
    echo ❌ Brain compilation failed
    exit /b 1
)

echo ✓ Brain compiled successfully!

echo Linking CUDA library...
nvcc --shared -o cuda_brain.dll cuda_kernels.obj cuda_brain.obj -lcublas -lcurand -lcusparse
if errorlevel 1 (
    echo ❌ Linking failed
    exit /b 1
)

echo.
echo ✅ BUILD COMPLETE!
echo ==================
echo Files created:
dir *.obj
dir *.dll
echo.
echo 🎉 RTX 4090 GPU acceleration ready!

