@echo off
echo Checking MSVC version and CUDA compatibility...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo MSVC Version:
cl

echo.
echo CUDA Version:
nvcc --version

echo.
echo Testing simple CUDA compilation...
echo #include <cuda_runtime.h> > test_simple.cu
echo __global__ void test_kernel() {} >> test_simple.cu
echo int main() { return 0; } >> test_simple.cu

nvcc --compiler-options "/EHsc /std:c++17" -o test_simple.exe test_simple.cu

if %ERRORLEVEL% EQU 0 (
    echo ✅ Simple CUDA compilation successful!
    del test_simple.cu test_simple.exe
) else (
    echo ❌ Simple CUDA compilation failed!
    echo This confirms the MSVC/CUDA compatibility issue
)

echo.
echo Recommended fixes:
echo 1. Use C++17 instead of C++20
echo 2. Avoid certain C++20 features
echo 3. Consider using older MSVC version if needed
