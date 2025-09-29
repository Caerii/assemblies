@echo off
echo Testing CUDA compilation with MSVC v14.39 (VS 2022 17.9)...

REM Try to use MSVC v14.39 if available
set MSVC_V14_39_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"

if exist %MSVC_V14_39_PATH%\cl.exe (
    echo ✅ MSVC v14.39 found!
    echo Using MSVC v14.39 for CUDA compilation...
    
    REM Set up environment with MSVC v14.39
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    
    REM Override with v14.39
    set PATH=%MSVC_V14_39_PATH%;%PATH%
    
    echo.
    echo MSVC Version:
    cl 2>&1 | findstr "Version"
    
    echo.
    echo Testing CUDA compilation with MSVC v14.39...
    
    REM Set CUDA path
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
    set PATH=%CUDA_PATH%\bin;%PATH%
    
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
        echo 1. MSVC v14.39 is not properly installed
        echo 2. CUDA 13.0 still has compatibility issues
        echo 3. We need to try a different approach
    )
    
) else (
    echo ❌ MSVC v14.39 not found!
    echo.
    echo Please install MSVC v14.39 first:
    echo 1. Run: .\install_msvc_v14_39.bat
    echo 2. Follow the installation steps
    echo 3. Then run this script again
    echo.
    echo Alternative: Try upgrading to CUDA 12.6 or 12.7
)

echo.
echo Build process complete.
