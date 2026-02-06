@echo off
REM Build Dense Assembly CUDA Kernels

echo Building Dense Assembly CUDA Kernels...

nvcc -shared -o ..\dlls\dense_assembly_kernels.dll ^
    dense_assembly_kernels.cu ^
    -O3 ^
    -use_fast_math ^
    --compiler-options /MD ^
    -Xcompiler "/LD" ^
    -arch=sm_89 ^
    -lcudart

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: dense_assembly_kernels.dll created
    dir ..\dlls\dense_assembly_kernels.dll
) else (
    echo FAILED: Build failed with error %ERRORLEVEL%
)

