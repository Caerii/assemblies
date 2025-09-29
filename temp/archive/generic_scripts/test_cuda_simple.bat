@echo off
echo Testing CUDA compilation with MSVC compatibility fixes...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo MSVC Version: 19.43.34808 (VS 2022 v17.13.2)
echo CUDA Version: 13.0
echo.

echo Creating minimal CUDA test...
echo #include ^<cuda_runtime.h^> > test_minimal.cu
echo __global__ void test_kernel^(^) {^} >> test_minimal.cu
echo int main^(^) { return 0; } >> test_minimal.cu

echo.
echo Testing CUDA compilation with C++17...
nvcc --compiler-options "/EHsc /std:c++17" -o test_minimal.exe test_minimal.cu

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with C++17!
    del test_minimal.cu test_minimal.exe
) else (
    echo ❌ CUDA compilation failed with C++17
    echo.
    echo Trying with C++14...
    nvcc --compiler-options "/EHsc /std:c++14" -o test_minimal.exe test_minimal.cu
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ CUDA compilation successful with C++14!
        del test_minimal.cu test_minimal.exe
    ) else (
        echo ❌ CUDA compilation failed with C++14
        echo.
        echo This confirms MSVC 19.43.34808 + CUDA 13.0 compatibility issue
        echo.
        echo Solutions:
        echo 1. Use older MSVC version (19.29 or earlier)
        echo 2. Use older CUDA version (12.x)
        echo 3. Use different compiler flags
    )
)

del test_minimal.cu 2>nul
del test_minimal.exe 2>nul
