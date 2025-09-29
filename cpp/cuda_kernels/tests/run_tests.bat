@echo off
REM Run Script for CUDA Kernel Optimization Tests
REM =============================================
REM 
REM This script runs all the built test executables and generates
REM a comprehensive test report.
REM 
REM Usage: run_tests.bat [test_name]
REM   - If no test_name provided, runs all tests
REM   - If test_name provided, runs only that specific test
REM 
REM Examples:
REM   run_tests.bat                    # Run all tests
REM   run_tests.bat warp_reduction     # Run only warp reduction test
REM   run_tests.bat benchmark_suite    # Run only benchmark suite

setlocal enabledelayedexpansion

REM Configuration
set OUTPUT_DIR=bin
set RESULTS_DIR=results
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

REM Create results directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM Get the test name from command line argument
set TEST_NAME=%1

echo.
echo 🧪 Running CUDA Kernel Optimization Tests
echo ========================================
echo.

REM Check if specific test requested
if "%TEST_NAME%"=="" (
    echo Running all tests...
    goto :run_all
) else (
    echo Running specific test: %TEST_NAME%
    goto :run_specific
)

:run_all
echo.
echo 📊 Running Warp Reduction Test...
echo ================================
if exist "%OUTPUT_DIR%\test_warp_reduction.exe" (
    "%OUTPUT_DIR%\test_warp_reduction.exe" > "%RESULTS_DIR%\warp_reduction_%TIMESTAMP%.txt" 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Warp reduction test completed successfully
    ) else (
        echo ❌ Warp reduction test failed with error code %ERRORLEVEL%
    )
) else (
    echo ❌ test_warp_reduction.exe not found. Please build first.
)

echo.
echo 📊 Running Radix Selection Test...
echo =================================
if exist "%OUTPUT_DIR%\test_radix_selection.exe" (
    "%OUTPUT_DIR%\test_radix_selection.exe" > "%RESULTS_DIR%\radix_selection_%TIMESTAMP%.txt" 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Radix selection test completed successfully
    ) else (
        echo ❌ Radix selection test failed with error code %ERRORLEVEL%
    )
) else (
    echo ❌ test_radix_selection.exe not found. Please build first.
)

echo.
echo 📊 Running Memory Coalescing Test...
echo ===================================
if exist "%OUTPUT_DIR%\test_memory_coalescing.exe" (
    "%OUTPUT_DIR%\test_memory_coalescing.exe" > "%RESULTS_DIR%\memory_coalescing_%TIMESTAMP%.txt" 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Memory coalescing test completed successfully
    ) else (
        echo ❌ Memory coalescing test failed with error code %ERRORLEVEL%
    )
) else (
    echo ❌ test_memory_coalescing.exe not found. Please build first.
)

echo.
echo 📊 Running Consolidated Kernels Test...
echo ======================================
if exist "%OUTPUT_DIR%\test_consolidated_kernels.exe" (
    "%OUTPUT_DIR%\test_consolidated_kernels.exe" > "%RESULTS_DIR%\consolidated_kernels_%TIMESTAMP%.txt" 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Consolidated kernels test completed successfully
    ) else (
        echo ❌ Consolidated kernels test failed with error code %ERRORLEVEL%
    )
) else (
    echo ❌ test_consolidated_kernels.exe not found. Please build first.
)

echo.
echo 📊 Running Benchmark Suite...
echo ============================
if exist "%OUTPUT_DIR%\test_benchmark_suite.exe" (
    "%OUTPUT_DIR%\test_benchmark_suite.exe" > "%RESULTS_DIR%\benchmark_suite_%TIMESTAMP%.txt" 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Benchmark suite completed successfully
    ) else (
        echo ❌ Benchmark suite failed with error code %ERRORLEVEL%
    )
) else (
    echo ❌ test_benchmark_suite.exe not found. Please build first.
)

goto :run_complete

:run_specific
if "%TEST_NAME%"=="warp_reduction" (
    echo Running warp reduction test...
    if exist "%OUTPUT_DIR%\test_warp_reduction.exe" (
        "%OUTPUT_DIR%\test_warp_reduction.exe"
    ) else (
        echo ❌ test_warp_reduction.exe not found. Please build first.
    )
) else if "%TEST_NAME%"=="radix_selection" (
    echo Running radix selection test...
    if exist "%OUTPUT_DIR%\test_radix_selection.exe" (
        "%OUTPUT_DIR%\test_radix_selection.exe"
    ) else (
        echo ❌ test_radix_selection.exe not found. Please build first.
    )
) else if "%TEST_NAME%"=="memory_coalescing" (
    echo Running memory coalescing test...
    if exist "%OUTPUT_DIR%\test_memory_coalescing.exe" (
        "%OUTPUT_DIR%\test_memory_coalescing.exe"
    ) else (
        echo ❌ test_memory_coalescing.exe not found. Please build first.
    )
) else if "%TEST_NAME%"=="consolidated_kernels" (
    echo Running consolidated kernels test...
    if exist "%OUTPUT_DIR%\test_consolidated_kernels.exe" (
        "%OUTPUT_DIR%\test_consolidated_kernels.exe"
    ) else (
        echo ❌ test_consolidated_kernels.exe not found. Please build first.
    )
) else if "%TEST_NAME%"=="benchmark_suite" (
    echo Running benchmark suite...
    if exist "%OUTPUT_DIR%\test_benchmark_suite.exe" (
        "%OUTPUT_DIR%\test_benchmark_suite.exe"
    ) else (
        echo ❌ test_benchmark_suite.exe not found. Please build first.
    )
) else (
    echo ❌ Unknown test name: %TEST_NAME%
    echo Available tests: warp_reduction, radix_selection, memory_coalescing, consolidated_kernels, benchmark_suite
    exit /b 1
)
goto :run_complete

:run_complete
echo.
echo ✅ Test execution complete!
echo.
echo Results are saved in the %RESULTS_DIR% directory:
dir /b "%RESULTS_DIR%\*.txt"
echo.
echo To view results:
echo   type "%RESULTS_DIR%\warp_reduction_%TIMESTAMP%.txt"
echo   type "%RESULTS_DIR%\radix_selection_%TIMESTAMP%.txt"
echo   type "%RESULTS_DIR%\memory_coalescing_%TIMESTAMP%.txt"
echo   type "%RESULTS_DIR%\consolidated_kernels_%TIMESTAMP%.txt"
echo   type "%RESULTS_DIR%\benchmark_suite_%TIMESTAMP%.txt"
echo.

endlocal
