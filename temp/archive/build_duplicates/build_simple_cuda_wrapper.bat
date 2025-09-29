@echo off
echo Building Simple CUDA Brain Wrapper...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

REM Compile simple CUDA wrapper
nvcc --compiler-options "/EHsc /std:c++17 /O2" -o simple_cuda_wrapper.dll --shared simple_cuda_wrapper.cu -lcudart -lcurand --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ Simple CUDA Brain Wrapper built successfully!
    echo    Output: simple_cuda_wrapper.dll
) else (
    echo ❌ Build failed!
    exit /b 1
)

echo.
echo Testing simple CUDA wrapper...
python test_simple_cuda_wrapper.py
