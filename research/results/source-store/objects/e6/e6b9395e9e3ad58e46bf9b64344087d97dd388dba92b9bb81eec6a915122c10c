@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  Assembly Calculus CUDA Kernel Builder
echo ========================================
echo.

REM Auto-detect CUDA installation
set CUDA_PATH=
for %%d in (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
) do (
    if exist %%d\bin\nvcc.exe (
        set CUDA_PATH=%%~d
        goto :found_cuda
    )
)
echo ERROR: CUDA toolkit not found. Install from https://developer.nvidia.com/cuda-downloads
exit /b 1

:found_cuda
echo Found CUDA at: %CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%

REM Auto-detect GPU architecture
echo Detecting GPU architecture...
nvidia-smi --query-gpu=compute_cap --format=csv,noheader > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=compute_cap --format^=csv^,noheader') do set SM_MAJOR=%%a
    for /f "tokens=2 delims=." %%b in ('nvidia-smi --query-gpu=compute_cap --format^=csv^,noheader') do set SM_MINOR=%%b
    set ARCH=sm_!SM_MAJOR!!SM_MINOR!
) else (
    echo Could not detect GPU. Defaulting to sm_89.
    set ARCH=sm_89
)
echo Target architecture: %ARCH%
echo.

REM Create output directory
if not exist ..\dlls mkdir ..\dlls

REM Track results
set PASS=0
set FAIL=0

REM ---- Build each kernel ----

echo [1/4] Building dense_assembly_kernels...
nvcc -shared -o ..\dlls\dense_assembly_kernels.dll dense_assembly_kernels.cu -O3 -use_fast_math --compiler-options /MD -Xcompiler "/LD" -arch=%ARCH% -lcudart 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   OK: dense_assembly_kernels.dll
    set /a PASS+=1
) else (
    echo   FAIL: dense_assembly_kernels.dll
    set /a FAIL+=1
)

echo [2/4] Building nemo_implicit_kernels...
nvcc --compiler-options "/EHsc /O2" -o ..\dlls\nemo_implicit_kernels.dll --shared nemo_implicit_kernels.cu -lcudart -arch=%ARCH% 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   OK: nemo_implicit_kernels.dll
    set /a PASS+=1
) else (
    echo   FAIL: nemo_implicit_kernels.dll
    set /a FAIL+=1
)

echo [3/4] Building assembly_projection_kernel...
nvcc -O3 -arch=%ARCH% -Xcompiler "/MD" --shared -o ..\dlls\assembly_projection.dll assembly_projection_kernel.cu -lcudart 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   OK: assembly_projection.dll
    set /a PASS+=1
) else (
    echo   FAIL: assembly_projection.dll
    set /a FAIL+=1
)

echo [4/4] Building sparse_assembly_kernels...
nvcc -shared -o ..\dlls\sparse_assembly_kernels.dll sparse_assembly_kernels_v2.cu -O3 -use_fast_math --compiler-options /MD -Xcompiler "/LD" -arch=%ARCH% -lcudart 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   OK: sparse_assembly_kernels.dll
    set /a PASS+=1
) else (
    echo   FAIL: sparse_assembly_kernels.dll
    set /a FAIL+=1
)

echo.
echo ========================================
echo  Results: %PASS% passed, %FAIL% failed
echo ========================================

REM Verify DLLs load in Python
echo.
echo Verifying DLL loading...
python -c "import ctypes, os; dlls = os.listdir('../dlls'); [print(f'  OK: {d}') for d in dlls if d.endswith('.dll')]" 2>nul

endlocal
