@echo off
REM Build the assembly projection CUDA kernel

echo Building assembly_projection_kernel.cu...

nvcc -O3 -arch=sm_89 ^
     -Xcompiler "/MD" ^
     --shared ^
     -o assembly_projection.dll ^
     assembly_projection_kernel.cu ^
     -lcudart

if %ERRORLEVEL% EQU 0 (
    echo Build successful! Created assembly_projection.dll
) else (
    echo Build failed!
)

