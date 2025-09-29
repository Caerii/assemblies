@echo off
echo Building CUDA wrapper with static linking to include all dependencies...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo MSVC Version: 19.43.34808 (VS 2022 v17.13.2)
echo CUDA Version: 13.0
echo.

echo Building CUDA wrapper with static linking...

REM Try static linking approach
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_static.dll --shared cuda_minimal.cu -lcudart_static -lcurand_static --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with static linking!
    echo    Output: cuda_static.dll
    goto :test_dll
) else (
    echo ❌ Static linking failed
)

echo.
echo Trying with explicit CUDA library paths...
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_static.dll --shared cuda_minimal.cu -L%CUDA_LIB_PATH% -lcudart -lcurand --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with explicit library paths!
    echo    Output: cuda_static.dll
    goto :test_dll
) else (
    echo ❌ Explicit library paths failed
)

echo.
echo Trying with minimal dependencies...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_minimal.dll --shared cuda_minimal.cu --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with minimal dependencies!
    echo    Output: cuda_minimal.dll
    goto :test_dll
) else (
    echo ❌ Minimal dependencies failed
)

echo.
echo All methods failed. Let's try a different approach...
goto :end

:test_dll
echo.
echo Testing CUDA DLL...
python test_cuda_minimal.py

:end
echo.
echo Build process complete.
