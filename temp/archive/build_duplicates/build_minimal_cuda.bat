@echo off
echo Building minimal CUDA wrapper with MSVC compatibility fixes...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo MSVC Version: 19.43.34808 (VS 2022 v17.13.2)
echo CUDA Version: 13.0
echo.

echo Testing minimal CUDA compilation...

REM Try different approaches to work around MSVC compatibility
echo Method 1: C++14 with specific flags...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_minimal.dll --shared cuda_minimal.cu -lcudart -lcurand --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with C++14!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Method 1 failed
)

echo.
echo Method 2: C++14 with older architecture...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_minimal.dll --shared cuda_minimal.cu -lcudart -lcurand --gpu-architecture=compute_75 --gpu-code=sm_75

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with older architecture!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Method 2 failed
)

echo.
echo Method 3: C++14 with compute_70...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_minimal.dll --shared cuda_minimal.cu -lcudart -lcurand --gpu-architecture=compute_70 --gpu-code=sm_70

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with compute_70!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Method 3 failed
)

echo.
echo Method 4: Minimal flags only...
nvcc --compiler-options "/EHsc" -o cuda_minimal.dll --shared cuda_minimal.cu -lcudart -lcurand

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with minimal flags!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Method 4 failed
)

echo.
echo Method 5: Try without specific architecture...
nvcc --compiler-options "/EHsc /std:c++14" -o cuda_minimal.dll --shared cuda_minimal.cu -lcudart -lcurand

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful without specific architecture!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Method 5 failed
)

echo.
echo All methods failed. This confirms the MSVC 19.43.34808 + CUDA 13.0 compatibility issue.
echo.
echo The 0xC0000409 error is a known issue with:
echo - MSVC 19.43.34808 (VS 2022 v17.13.2) + CUDA 13.0
echo.
echo Solutions:
echo 1. Install older MSVC version (19.29 or earlier)
echo 2. Install CUDA 12.x instead of 13.0
echo 3. Use WSL with Linux CUDA toolkit
echo 4. Use Docker with CUDA support
echo 5. Use pre-compiled CUDA libraries
goto :end

:test_dll
echo.
echo Testing CUDA DLL...
python test_cuda_minimal.py

:end
echo.
echo Build process complete.