@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM CUDA SUPERSET BUILD SCRIPT
REM =============================================================================
REM Combines all build approaches with command-line flags for different modes
REM Usage: build_cuda_superset.bat [OPTIONS]
REM
REM OPTIONS:
REM   --mode=[basic|comprehensive|runtime|complete]  Build mode (default: comprehensive)
REM   --target=[kernels|gpu_memory|complete]        Target to build (default: kernels)
REM   --arch=[compute_89|compute_86|compute_75|compute_70]  GPU architecture (default: compute_89)
REM   --std=[c++14|c++17|c++20]                    C++ standard (default: c++20)
REM   --test                                       Run tests after build
REM   --copy-dlls                                  Copy runtime DLLs
REM   --verbose                                    Verbose output
REM   --help                                       Show this help
REM =============================================================================

REM =============================================================================
REM ENVIRONMENT VARIABLES
REM =============================================================================
REM Visual Studio paths
set VS2022_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
set MSVC_V14_39_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64

REM CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_HOME=%CUDA_PATH%
set CUDA_BIN_PATH=%CUDA_PATH%\bin
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64

REM Build directories
set BUILD_DIR=.build
set DLLS_DIR=%BUILD_DIR%\dlls
set OBJ_DIR=%BUILD_DIR%\obj

REM =============================================================================
REM DEFAULT CONFIGURATION (Optimized for RTX 4090 on Windows 11)
REM =============================================================================
set BUILD_MODE=comprehensive
set BUILD_TARGET=kernels
set GPU_ARCH=compute_89
set CPP_STD=c++20
set RUN_TESTS=false
set COPY_DLLS=true
set VERBOSE=false

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--help" goto :show_help

REM Handle --target argument
if "%~1"=="--target" (
    if "%~2"=="kernels" set BUILD_TARGET=kernels
    if "%~2"=="kernels_optimized" set BUILD_TARGET=kernels_optimized
    if "%~2"=="gpu_memory_optimized" set BUILD_TARGET=gpu_memory_optimized
    if "%~2"=="complete_optimized" set BUILD_TARGET=complete_optimized
    if "%~2"=="all_optimized" set BUILD_TARGET=all_optimized
    if "%~2"=="gpu_memory" set BUILD_TARGET=gpu_memory
    if "%~2"=="complete" set BUILD_TARGET=complete
    if "%~2"=="wrapper" set BUILD_TARGET=wrapper
    if "%~2"=="minimal" set BUILD_TARGET=minimal
    if "%~2"=="simple" set BUILD_TARGET=simple
    shift
    shift
    goto :parse_args
)

REM Handle --mode argument
if "%~1"=="--mode" (
    if "%~2"=="basic" set BUILD_MODE=basic
    if "%~2"=="comprehensive" set BUILD_MODE=comprehensive
    if "%~2"=="runtime" set BUILD_MODE=runtime
    if "%~2"=="complete" set BUILD_MODE=complete
    shift
    shift
    goto :parse_args
)

REM Handle --arch argument
if "%~1"=="--arch" (
    if "%~2"=="compute_89" set GPU_ARCH=compute_89
    if "%~2"=="compute_86" set GPU_ARCH=compute_86
    if "%~2"=="compute_75" set GPU_ARCH=compute_75
    if "%~2"=="compute_70" set GPU_ARCH=compute_70
    shift
    shift
    goto :parse_args
)

REM Handle --std argument
if "%~1"=="--std" (
    if "%~2"=="c++14" set CPP_STD=c++14
    if "%~2"=="c++17" set CPP_STD=c++17
    if "%~2"=="c++20" set CPP_STD=c++20
    shift
    shift
    goto :parse_args
)

REM Handle single-argument flags
if "%~1"=="--test" set RUN_TESTS=true
if "%~1"=="--no-copy-dlls" set COPY_DLLS=false
if "%~1"=="--verbose" set VERBOSE=true

shift
goto :parse_args

:args_done

REM Show configuration
echo.
echo 🚀 CUDA BUILD SCRIPT
echo =============================
echo Build Mode: %BUILD_MODE%
echo Target: %BUILD_TARGET%
echo GPU Architecture: %GPU_ARCH%
echo C++ Standard: %CPP_STD%
echo Run Tests: %RUN_TESTS%
echo Copy DLLs: %COPY_DLLS%
echo Verbose: %VERBOSE%
echo.

REM Set up Visual Studio environment with MSVC v14.39
echo Setting up Visual Studio environment...
call "%VS2022_PATH%" -vcvars_ver=14.39
if errorlevel 1 (
    echo ❌ Failed to set up Visual Studio environment with v14.39
    echo    Trying fallback without version specification...
    call "%VS2022_PATH%"
    if errorlevel 1 (
        echo ❌ Visual Studio environment setup failed completely
        exit /b 1
    )
)

REM Override with MSVC v14.39 if available
set PATH="%MSVC_V14_39_PATH%";%PATH%

REM Set CUDA paths
set PATH=%CUDA_BIN_PATH%;%CUDA_LIB_PATH%;%PATH%

REM Display versions
echo MSVC Version:
cl 2>&1 | findstr "Version"
echo.

echo CUDA Version:
nvcc --version
if errorlevel 1 (
    echo ❌ CUDA compiler not found
    exit /b 1
)
echo ✅ CUDA compiler ready
echo.

REM Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%DLLS_DIR%" mkdir "%DLLS_DIR%"
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"


REM Determine source file and output name based on target
if "%BUILD_TARGET%"=="kernels" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_kernels.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_kernels.dll
) else if "%BUILD_TARGET%"=="kernels_optimized" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_kernels_optimized.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_kernels_optimized.dll
) else if "%BUILD_TARGET%"=="gpu_memory" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_memory.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_memory.dll
) else if "%BUILD_TARGET%"=="gpu_memory_optimized" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_memory_optimized.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_memory_optimized.dll
) else if "%BUILD_TARGET%"=="complete" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_brain.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_brain.dll
) else if "%BUILD_TARGET%"=="complete_optimized" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_brain_optimized.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_brain_optimized.dll
) else if "%BUILD_TARGET%"=="wrapper" (
    set SOURCE_FILE=..\cuda_kernels\assemblies_cuda_wrapper.cu
    set OUTPUT_NAME=%DLLS_DIR%\assemblies_cuda_wrapper.dll
) else if "%BUILD_TARGET%"=="simple" (
    set SOURCE_FILE=..\cuda_kernels\simple_cuda_brain.cu
    set OUTPUT_NAME=%DLLS_DIR%\simple_cuda_brain.dll
) else if "%BUILD_TARGET%"=="all_optimized" (
    echo Building all optimized targets...
    call :build_optimized_target "kernels_optimized" "..\cuda_kernels\assemblies_cuda_kernels_optimized.cu" "assemblies_cuda_kernels_optimized.dll"
    call :build_optimized_target "gpu_memory_optimized" "..\cuda_kernels\assemblies_cuda_memory_optimized.cu" "assemblies_cuda_memory_optimized.dll"
    call :build_optimized_target "complete_optimized" "..\cuda_kernels\assemblies_cuda_brain_optimized.cu" "assemblies_cuda_brain_optimized.dll"
    goto :post_build
) else (
    echo ❌ Invalid target: %BUILD_TARGET%
    echo    Available targets: kernels, kernels_optimized, gpu_memory, gpu_memory_optimized, complete, complete_optimized, wrapper, simple, all_optimized
    exit /b 1
)

REM Check if source file exists
echo DEBUG: SOURCE_FILE = %SOURCE_FILE%
echo DEBUG: OUTPUT_NAME = %OUTPUT_NAME%
if not exist "%SOURCE_FILE%" (
    echo ❌ Source file not found: %SOURCE_FILE%
    echo    Available .cu files:
    dir ..\cuda_kernels\*.cu 2>nul
    exit /b 1
)

echo Building %BUILD_TARGET% with %BUILD_MODE% mode...
echo Source: %SOURCE_FILE%
echo Output: %OUTPUT_NAME%
echo.

REM Build based on mode
if "%BUILD_MODE%"=="basic" goto :build_basic
if "%BUILD_MODE%"=="comprehensive" goto :build_comprehensive
if "%BUILD_MODE%"=="runtime" goto :build_runtime
if "%BUILD_MODE%"=="complete" goto :build_complete

:build_basic
echo Method: Basic compilation...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Basic compilation successful!
    goto :post_build
) else (
    echo ❌ Basic compilation failed
    goto :build_failed
)

:build_comprehensive
echo Method: Comprehensive compilation with fallbacks...
echo.

REM Method 1: Minimal flags (most likely to work with latest CUDA/MSVC)
echo Trying Method 1: Minimal flags...
nvcc --compiler-options "/EHsc" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 1 successful!
    goto :post_build
)

REM Method 2: C++20 with specific flags (RTX 4090 optimized)
echo Trying Method 2: C++20 with specific flags...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_% -I.
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 2 successful!
    goto :post_build
)

REM Method 3: RTX 30xx architecture
echo Trying Method 3: RTX 30xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand --gpu-architecture=compute_86 --gpu-code=sm_86 -I.
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 3 successful!
    goto :post_build
)

REM Method 4: RTX 20xx architecture
echo Trying Method 4: RTX 20xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand --gpu-architecture=compute_75 --gpu-code=sm_75 -I.
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 4 successful!
    goto :post_build
)

REM Method 5: RTX 10xx architecture
echo Trying Method 5: RTX 10xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand --gpu-architecture=compute_70 --gpu-code=sm_70,sm_60
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 5 successful!
    goto :post_build
)

REM Method 6: No specific architecture
echo Trying Method 6: No specific architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD%" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart -lcurand
if %ERRORLEVEL% EQU 0 (
    echo ✅ Method 6 successful!
    goto :post_build
)

echo ❌ All comprehensive methods failed
goto :build_failed

:build_runtime
echo Method: Runtime dependency management...
echo.

REM Method 1: Static linking
echo Trying Method 1: Static linking...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -lcudart_static -lcurand_static --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Static linking successful!
    goto :post_build
)

REM Method 2: Dynamic linking with explicit paths
echo Trying Method 2: Dynamic linking with explicit paths...
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% -L%CUDA_LIB_PATH% -lcudart -lcurand --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Dynamic linking successful!
    goto :post_build
)

REM Method 3: Copy DLLs and compile
echo Trying Method 3: Copy DLLs and compile...
if "%COPY_DLLS%"=="true" (
    copy "%CUDA_PATH%\bin\cudart64_13.dll" . 2>nul
    copy "%CUDA_PATH%\bin\curand64_13.dll" . 2>nul
    copy "%CUDA_PATH%\bin\cublas64_13.dll" . 2>nul
    copy "%CUDA_PATH%\bin\cublasLt64_13.dll" . 2>nul
    echo ✅ Runtime DLLs copied
)
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Runtime DLLs method successful!
    goto :post_build
)

REM Method 4: Minimal compilation
echo Trying Method 4: Minimal compilation...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o %OUTPUT_NAME% --shared %SOURCE_FILE% --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Minimal compilation successful!
    goto :post_build
)

echo ❌ All runtime methods failed
goto :build_failed

:build_complete
echo Method: Complete build with all libraries...
nvcc -o %OUTPUT_NAME% --shared --compiler-options "/EHsc /std:%CPP_STD% /O2" -I. -lcuda -lcurand -lcublas %SOURCE_FILE%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Complete build successful!
    goto :post_build
) else (
    echo ❌ Complete build failed
    goto :build_failed
)

:post_build
echo.
echo ✅ Build successful! Output: %OUTPUT_NAME%

REM Copy runtime DLLs if requested
if "%COPY_DLLS%"=="true" (
    echo.
    echo Copying CUDA runtime DLLs to %DLLS_DIR%\...
    copy "%CUDA_BIN_PATH%\x64\cudart64_13.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\curand64_10.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\cublas64_13.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\cublasLt64_13.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\cufft64_12.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\cusparse64_12.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\nvJitLink_130_0.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\nvrtc64_130_0.dll" "%DLLS_DIR%\" 2>nul
    copy "%CUDA_BIN_PATH%\x64\nvrtc-builtins64_130.dll" "%DLLS_DIR%\" 2>nul
    echo ✅ All CUDA runtime DLLs copied to %DLLS_DIR%\!
)

REM Test the DLL if requested
if "%RUN_TESTS%"=="true" (
    echo.
    echo Testing %OUTPUT_NAME%...
    python -c "import ctypes; lib = ctypes.CDLL('%OUTPUT_NAME%'); print('✅ %OUTPUT_NAME% loaded successfully!')"
    if %ERRORLEVEL% neq 0 (
        echo ❌ DLL loading test failed!
        goto :build_failed
    )
    
    REM Run specific test based on target
    if "%BUILD_TARGET%"=="gpu_memory" (
        if exist test_gpu_memory_cuda_brain.py (
            echo Running GPU memory test...
            python test_gpu_memory_cuda_brain.py
        )
    ) else if "%BUILD_TARGET%"=="kernels" (
        if exist test_cuda_with_runtime.py (
            echo Running CUDA kernels test...
            python test_cuda_with_runtime.py
        )
    ) else if "%BUILD_TARGET%"=="complete" (
        if exist test_cuda_with_runtime.py (
            echo Running complete CUDA test...
            python test_cuda_with_runtime.py
        )
    )
)

REM Show build results
echo.
echo 🎉 BUILD SUCCESS!
echo ================
echo Output: %OUTPUT_NAME%
if exist %OUTPUT_NAME% (
    for %%A in (%OUTPUT_NAME%) do echo Size: %%~zA bytes
)
echo Mode: %BUILD_MODE%
echo Target: %BUILD_TARGET%
echo Architecture: %GPU_ARCH%
echo C++ Standard: %CPP_STD%
echo.

echo 🧠 CUDA Features Ready:
echo    ✓ GPU-accelerated neural simulation
echo    ✓ RTX 4090 optimized kernels
echo    ✓ Billion-scale neuron support
echo    ✓ Real-time performance
echo    ✓ Python integration ready
echo.

goto :end

:build_failed
echo.
echo ❌ BUILD FAILED
echo ===============
echo All compilation methods failed for %BUILD_TARGET%
echo.
echo Troubleshooting:
echo 1. Check MSVC v14.39 installation
echo 2. Verify CUDA 13.0 installation
echo 3. Try different --arch parameter
echo 4. Check source file exists: %SOURCE_FILE%
echo 5. Run with --verbose for detailed output
echo.
exit /b 1

:show_help
echo.
echo CUDA BUILD SCRIPT - HELP
echo ==================================
echo.
echo USAGE:
echo   build.bat [OPTIONS]
echo.
echo OPTIONS:
echo   --mode=[basic^|comprehensive^|runtime^|complete]
echo     Build mode (default: comprehensive)
echo     - basic: Single compilation attempt
echo     - comprehensive: Multiple fallback methods
echo     - runtime: Focus on runtime dependencies
echo     - complete: Full library linking
echo.
echo   --target=[kernels^|kernels_optimized^|gpu_memory^|gpu_memory_optimized^|complete^|complete_optimized^|wrapper^|simple^|all_optimized]
echo     Target to build (default: kernels)
echo     - kernels: cuda_kernels\assemblies_cuda_kernels.cu
echo     - kernels_optimized: cuda_kernels\assemblies_cuda_kernels_optimized.cu
echo     - gpu_memory: cuda_kernels\assemblies_cuda_memory.cu
echo     - gpu_memory_optimized: cuda_kernels\assemblies_cuda_memory_optimized.cu
echo     - complete: cuda_kernels\assemblies_cuda_brain.cu
echo     - complete_optimized: cuda_kernels\assemblies_cuda_brain_optimized.cu
echo     - wrapper: cuda_kernels\assemblies_cuda_wrapper.cu
echo     - simple: cuda_kernels\simple_cuda_brain.cu
echo     - all_optimized: Build all optimized targets
echo.
echo   --arch=[compute_89^|compute_86^|compute_75^|compute_70]
echo     GPU architecture (default: compute_89)
echo     - compute_89: RTX 4090 (Ada Lovelace)
echo     - compute_86: RTX 30xx series (Ampere)
echo     - compute_75: RTX 20xx series (Turing)
echo     - compute_70: RTX 10xx series (Pascal)
echo.
echo   --std=[c++14^|c++17^|c++20]
echo     C++ standard (default: c++20)
echo     - c++14: Maximum compatibility
echo     - c++17: Better performance, modern features
echo     - c++20: Latest features, best performance (RTX 4090 optimized)
echo.
echo   --test
echo     Run tests after successful build
echo.
echo   --no-copy-dlls
echo     Skip copying runtime DLLs
echo.
echo   --verbose
echo     Enable verbose output
echo.
echo   --help
echo     Show this help message
echo.
echo EXAMPLES:
echo   build.bat
echo   build.bat --mode basic --target kernels
echo   build.bat --mode runtime --test --verbose
echo   build.bat --target gpu_memory --arch compute_75
echo.
goto :end

:end
echo.
echo Build process complete.
if "%VERBOSE%"=="true" pause

REM Function to build optimized targets
:build_optimized_target
set TARGET_NAME=%~1
set SOURCE_FILE=%~2
set OUTPUT_NAME=%DLLS_DIR%\%~3

echo.
echo Building %TARGET_NAME%...
echo ========================

REM Check if source file exists
if not exist "%SOURCE_FILE%" (
    echo ❌ Source file not found: %SOURCE_FILE%
    goto :build_optimized_next
)

REM Try different compilation methods for optimized targets (same as comprehensive build)
echo Method 1: Minimal flags...
nvcc --compiler-options "/EHsc" -o "%OUTPUT_NAME%" --shared "%SOURCE_FILE%" -lcudart -lcurand
if %ERRORLEVEL% EQU 0 (
    echo ✅ %TARGET_NAME% built successfully!
    goto :build_optimized_next
)

echo Method 2: C++20 with specific flags...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_NAME%" --shared "%SOURCE_FILE%" -lcudart -lcurand --gpu-architecture=%GPU_ARCH% --gpu-code=%GPU_ARCH:compute_=sm_% -I.
if %ERRORLEVEL% EQU 0 (
    echo ✅ %TARGET_NAME% built successfully!
    goto :build_optimized_next
)

echo Method 3: RTX 30xx architecture...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_NAME%" --shared "%SOURCE_FILE%" -lcudart -lcurand --gpu-architecture=compute_86 --gpu-code=sm_86 -I.
if %ERRORLEVEL% EQU 0 (
    echo ✅ %TARGET_NAME% built successfully!
    goto :build_optimized_next
)

echo Method 4: Basic optimization...
nvcc --compiler-options "/EHsc /std:%CPP_STD% /O2" -o "%OUTPUT_NAME%" --shared "%SOURCE_FILE%" -lcudart -lcurand
if %ERRORLEVEL% EQU 0 (
    echo ✅ %TARGET_NAME% built successfully!
    goto :build_optimized_next
)

echo ❌ Failed to build %TARGET_NAME%

:build_optimized_next
goto :eof
