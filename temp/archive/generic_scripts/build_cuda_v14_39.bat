@echo off
echo Building CUDA kernels with MSVC v14.39...
echo.

REM Set up Visual Studio environment with MSVC v14.39
set "VSINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VCINSTALLDIR=%VSINSTALLDIR%\VC"
set "VCToolsInstallDir=%VCINSTALLDIR%\Tools\MSVC\14.39.33519"
set "WindowsSdkDir=C:\Program Files (x86)\Windows Kits\10"
set "WindowsSdkVersion=10.0.22621.0"

REM Set up environment variables
set "PATH=%VCToolsInstallDir%\bin\Hostx64\x64;%WindowsSdkDir%\bin\10.0.22621.0\x64;%PATH%"
set "INCLUDE=%VCToolsInstallDir%\include;%WindowsSdkDir%\include\10.0.22621.0\ucrt;%WindowsSdkDir%\include\10.0.22621.0\shared;%WindowsSdkDir%\include\10.0.22621.0\um;%WindowsSdkDir%\include\10.0.22621.0\winrt"
set "LIB=%VCToolsInstallDir%\lib\x64;%WindowsSdkDir%\lib\10.0.22621.0\ucrt\x64;%WindowsSdkDir%\lib\10.0.22621.0\um\x64"

REM Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_HOME=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%

echo MSVC Version:
cl 2>&1 | findstr "Version"
echo.

echo CUDA Version:
nvcc --version
echo.

echo Building CUDA kernels with MSVC v14.39...
echo.

REM Build the CUDA kernels with dynamic linking
nvcc --shared --compiler-options "/EHsc /std:c++17 /O2" ^
     --gpu-architecture=compute_89 --gpu-code=sm_89 ^
     -o cuda_kernels_v14_39.dll ^
     cuda_kernels_fixed.cu ^
     -lcudart -lcurand -lcublas -lcublasLt -lcufft -lcusparse

if %ERRORLEVEL% neq 0 (
    echo ❌ CUDA compilation failed!
    pause
    exit /b 1
)

echo ✅ CUDA kernels compiled successfully!

REM Copy all CUDA runtime DLLs
echo Copying CUDA runtime DLLs...
copy "%CUDA_PATH%\bin\x64\cudart64_13.dll" .
copy "%CUDA_PATH%\bin\x64\curand64_10.dll" .
copy "%CUDA_PATH%\bin\x64\cublas64_13.dll" .
copy "%CUDA_PATH%\bin\x64\cublasLt64_13.dll" .
copy "%CUDA_PATH%\bin\x64\cufft64_12.dll" .
copy "%CUDA_PATH%\bin\x64\cusparse64_12.dll" .
copy "%CUDA_PATH%\bin\x64\nvJitLink_130_0.dll" .
copy "%CUDA_PATH%\bin\x64\nvrtc64_130_0.dll" .
copy "%CUDA_PATH%\bin\x64\nvrtc-builtins64_130.dll" .

echo ✅ All CUDA runtime DLLs copied!

REM Test the DLL
echo Testing CUDA kernels DLL...
python -c "import ctypes; lib = ctypes.CDLL('cuda_kernels_v14_39.dll'); print('✅ cuda_kernels_v14_39.dll loaded successfully!')"

if %ERRORLEVEL% neq 0 (
    echo ❌ DLL loading failed!
    pause
    exit /b 1
)

echo.
echo 🎉 SUCCESS! CUDA kernels with MSVC v14.39 are ready!
echo.
pause

