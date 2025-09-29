@echo off
echo Testing CUDA compilation with MSVC v14.39 (VS 2022 17.9)...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Override with MSVC v14.39
set PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64";%PATH%

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo MSVC Version (should be 19.39.33523):
cl 2>&1 | findstr "Version"

echo.
echo CUDA Version:
nvcc --version

echo.
echo Testing CUDA compilation with MSVC v14.39...

REM Test compilation
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_v14_39.dll --shared cuda_minimal.cu -lcudart -lcurand --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with MSVC v14.39!
    echo    Output: cuda_v14_39.dll
    echo.
    echo Testing the DLL...
    python test_cuda_v14_39.py
) else (
    echo ❌ CUDA compilation failed with MSVC v14.39
    echo.
    echo This might mean:
    echo 1. There's still a compatibility issue
    echo 2. We need to try different compiler flags
    echo 3. We need to check CUDA 13.0 compatibility
)

echo.
echo Build process complete.

