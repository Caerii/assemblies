@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Building NEMO Implicit Connectivity Kernels
echo ========================================

REM Set up Visual Studio environment
set VS2022_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
call "%VS2022_PATH%" -vcvars_ver=14.39

REM Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo Compiling nemo_implicit_kernels.cu...
echo.

REM Try compilation with minimal flags first
nvcc --compiler-options "/EHsc" -o ..\dlls\nemo_implicit_kernels.dll --shared nemo_implicit_kernels.cu -lcudart

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Built: ..\dlls\nemo_implicit_kernels.dll
    echo ========================================
    
    REM Test load
    python -c "import ctypes; dll = ctypes.CDLL('../dlls/nemo_implicit_kernels.dll'); print('DLL loaded successfully!')"
) else (
    echo.
    echo Trying with explicit architecture...
    nvcc --compiler-options "/EHsc /O2" -o ..\dlls\nemo_implicit_kernels.dll --shared nemo_implicit_kernels.cu -lcudart --gpu-architecture=compute_89 --gpu-code=sm_89
    
    if %ERRORLEVEL% EQU 0 (
        echo SUCCESS with compute_89!
    ) else (
        echo.
        echo Trying compute_86...
        nvcc --compiler-options "/EHsc /O2" -o ..\dlls\nemo_implicit_kernels.dll --shared nemo_implicit_kernels.cu -lcudart --gpu-architecture=compute_86 --gpu-code=sm_86
        
        if %ERRORLEVEL% EQU 0 (
            echo SUCCESS with compute_86!
        ) else (
            echo FAILED to compile
        )
    )
)

echo.
pause

