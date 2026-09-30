@echo off
REM Build Script for CUDA Kernel Optimization Tests
REM ================================================
REM 
REM This script builds all the isolated test files for validating
REM CUDA kernel algorithmic improvements.
REM 
REM Usage: build_tests.bat [test_name]
REM   - If no test_name provided, builds all tests
REM   - If test_name provided, builds only that specific test
REM 
REM Examples:
REM   build_tests.bat                    # Build all tests
REM   build_tests.bat warp_reduction     # Build only warp reduction test
REM   build_tests.bat radix_selection    # Build only radix selection test

setlocal enabledelayedexpansion

REM Configuration
set CUDA_ARCH=compute_89
set CUDA_CODE=sm_89
set CPP_STD=c++20
set BUILD_DIR=build
set OUTPUT_DIR=bin

REM Create build directories
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Get the test name from command line argument
set TEST_NAME=%1

REM Function to build a specific test
:build_test
set TEST_FILE=%1
set OUTPUT_NAME=%2
set EXTRA_LIBS=%3

echo.
echo Building %TEST_FILE%...
echo ========================

REM Try different compilation methods
echo Method 1: Minimal flags...
nvcc --compiler-options "/EHsc" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 1 successful!
    goto :next_test
)

echo Method 2: C++20 with specific flags...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS% --gpu-architecture=%CUDA_ARCH% --gpu-code=%CUDA_CODE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 2 successful!
    goto :next_test
)

echo Method 3: RTX 30xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS% --gpu-architecture=compute_86 --gpu-code=sm_86
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 3 successful!
    goto :next_test
)

echo Method 4: RTX 20xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS% --gpu-architecture=compute_75 --gpu-code=sm_75
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 4 successful!
    goto :next_test
)

echo Method 5: RTX 10xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS% --gpu-architecture=compute_70 --gpu-code=sm_70,sm_60
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 5 successful!
    goto :next_test
)

echo Method 6: No specific architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD%" -o "%OUTPUT_DIR%\%OUTPUT_NAME%.exe" "%TEST_FILE%" -lcudart %EXTRA_LIBS%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 6 successful!
    goto :next_test
)

echo ❌ All compilation methods failed for %TEST_FILE%
goto :next_test

:next_test
echo.

REM Main build logic
echo 🚀 Building CUDA Kernel Optimization Tests
echo ==========================================
echo.

REM Check if specific test requested
if "%TEST_NAME%"=="" (
    echo Building all tests...
    goto :build_all
) else (
    echo Building specific test: %TEST_NAME%
    goto :build_specific
)

:build_all
echo Building all test files...
call :build_test "test_warp_reduction.cu" "test_warp_reduction" "-lcurand"
call :build_test "test_radix_selection.cu" "test_radix_selection" ""
call :build_test "test_memory_coalescing.cu" "test_memory_coalescing" ""
call :build_test "test_consolidated_kernels.cu" "test_consolidated_kernels" "-lcurand"
call :build_test "test_benchmark_suite.cu" "test_benchmark_suite" "-lcurand"
goto :build_complete

:build_specific
if "%TEST_NAME%"=="warp_reduction" (
    call :build_test "test_warp_reduction.cu" "test_warp_reduction" "-lcurand"
) else if "%TEST_NAME%"=="radix_selection" (
    call :build_test "test_radix_selection.cu" "test_radix_selection" ""
) else if "%TEST_NAME%"=="memory_coalescing" (
    call :build_test "test_memory_coalescing.cu" "test_memory_coalescing" ""
) else if "%TEST_NAME%"=="consolidated_kernels" (
    call :build_test "test_consolidated_kernels.cu" "test_consolidated_kernels" "-lcurand"
) else if "%TEST_NAME%"=="benchmark_suite" (
    call :build_test "test_benchmark_suite.cu" "test_benchmark_suite" "-lcurand"
) else (
    echo ❌ Unknown test name: %TEST_NAME%
    echo Available tests: warp_reduction, radix_selection, memory_coalescing, consolidated_kernels, benchmark_suite
    exit /b 1
)
goto :build_complete

:build_complete
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
echo To run all tests at once:
echo   %OUTPUT_DIR%\test_benchmark_suite.exe
echo.

endlocal
