@echo off
REM Simple Build Script for CUDA Kernel Optimization Tests
REM ======================================================

setlocal enabledelayedexpansion

REM Configuration
set CUDA_ARCH=compute_89
set CUDA_CODE=sm_89
set CPP_STD=c++20
set OUTPUT_DIR=bin

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo 🚀 Building CUDA Kernel Optimization Tests
echo ==========================================
echo.

REM Build warp reduction test
echo Building test_warp_reduction.cu...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\test_warp_reduction.exe" "test_warp_reduction.cu" -lcudart -lcurand --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ test_warp_reduction.exe built successfully
) else (
    echo ❌ Failed to build test_warp_reduction.exe
)

REM Build radix selection test
echo.
echo Building test_radix_selection.cu...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\test_radix_selection.exe" "test_radix_selection.cu" -lcudart --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ test_radix_selection.exe built successfully
) else (
    echo ❌ Failed to build test_radix_selection.exe
)

REM Build memory coalescing test
echo.
echo Building test_memory_coalescing.cu...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\test_memory_coalescing.exe" "test_memory_coalescing.cu" -lcudart --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ test_memory_coalescing.exe built successfully
) else (
    echo ❌ Failed to build test_memory_coalescing.exe
)

REM Build consolidated kernels test
echo.
echo Building test_consolidated_kernels.cu...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\test_consolidated_kernels.exe" "test_consolidated_kernels.cu" -lcudart -lcurand --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ test_consolidated_kernels.exe built successfully
) else (
    echo ❌ Failed to build test_consolidated_kernels.exe
)

REM Build benchmark suite test
echo.
echo Building test_benchmark_suite.cu...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\test_benchmark_suite.exe" "test_benchmark_suite.cu" -lcudart -lcurand --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ test_benchmark_suite.exe built successfully
) else (
    echo ❌ Failed to build test_benchmark_suite.exe
)

echo.
echo ✅ Build process complete!
echo.
echo Built executables are in the %OUTPUT_DIR% directory:
dir /b "%OUTPUT_DIR%\*.exe"
echo.
echo To run tests:
echo   %OUTPUT_DIR%\test_warp_reduction.exe
echo   %OUTPUT_DIR%\test_radix_selection.exe
echo   %OUTPUT_DIR%\test_memory_coalescing.exe
echo   %OUTPUT_DIR%\test_consolidated_kernels.exe
echo   %OUTPUT_DIR%\test_benchmark_suite.exe
echo.

endlocal
