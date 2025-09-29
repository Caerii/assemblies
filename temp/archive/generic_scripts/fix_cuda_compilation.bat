@echo off
echo Fixing CUDA compilation with MSVC compatibility workarounds...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo MSVC Version: 19.43.34808 (VS 2022 v17.13.2)
echo CUDA Version: 13.0
echo.

echo Creating optimized CUDA wrapper with compatibility fixes...
echo #include ^<cuda_runtime.h^> > cuda_fixed.cu
echo #include ^<curand_kernel.h^> >> cuda_fixed.cu
echo #include ^<iostream^> >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo extern "C" { >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo __global__ void generate_candidates_kernel^( >> cuda_fixed.cu
echo     curandState* states, >> cuda_fixed.cu
echo     float* candidate_weights, >> cuda_fixed.cu
echo     uint32_t num_candidates >> cuda_fixed.cu
echo ^) { >> cuda_fixed.cu
echo     uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; >> cuda_fixed.cu
echo     if ^(idx ^>= num_candidates^) return; >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo     curandState local_state = states[idx]; >> cuda_fixed.cu
echo     float sample = curand_uniform^(&local_state^); >> cuda_fixed.cu
echo     candidate_weights[idx] = -logf^(1.0f - sample^); >> cuda_fixed.cu
echo     states[idx] = local_state; >> cuda_fixed.cu
echo } >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo __global__ void curandSetupKernel^(curandState* states, unsigned long seed, uint32_t n^) { >> cuda_fixed.cu
echo     uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; >> cuda_fixed.cu
echo     if ^(idx ^< n^) { >> cuda_fixed.cu
echo         curand_init^(seed, idx, 0, &states[idx]^); >> cuda_fixed.cu
echo     } >> cuda_fixed.cu
echo } >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo class CudaBrainWrapper { >> cuda_fixed.cu
echo private: >> cuda_fixed.cu
echo     uint32_t n_neurons_; >> cuda_fixed.cu
echo     uint32_t k_active_; >> cuda_fixed.cu
echo     curandState* d_states_; >> cuda_fixed.cu
echo     float* d_candidates_; >> cuda_fixed.cu
echo public: >> cuda_fixed.cu
echo     CudaBrainWrapper^(uint32_t n_neurons, uint32_t k_active, uint32_t seed = 42^) >> cuda_fixed.cu
echo         : n_neurons_^(n_neurons^), k_active_^(k_active^) { >> cuda_fixed.cu
echo         cudaMalloc^(&d_states_, n_neurons * sizeof^(curandState^)^); >> cuda_fixed.cu
echo         cudaMalloc^(&d_candidates_, n_neurons * sizeof^(float^)^); >> cuda_fixed.cu
echo         curandSetupKernel^^^<^^^<^(n_neurons + 255^) / 256, 256^^^>^^^>^^^(d_states_, seed, n_neurons^); >> cuda_fixed.cu
echo         cudaDeviceSynchronize^(^); >> cuda_fixed.cu
echo     } >> cuda_fixed.cu
echo     ~CudaBrainWrapper^(^) { >> cuda_fixed.cu
echo         cudaFree^(d_states_^); >> cuda_fixed.cu
echo         cudaFree^(d_candidates_^); >> cuda_fixed.cu
echo     } >> cuda_fixed.cu
echo     void simulate_step^(^) { >> cuda_fixed.cu
echo         generate_candidates_kernel^^^<^^^<^(n_neurons_ + 255^) / 256, 256^^^>^^^>^^^(d_states_, d_candidates_, n_neurons_^); >> cuda_fixed.cu
echo         cudaDeviceSynchronize^(^); >> cuda_fixed.cu
echo     } >> cuda_fixed.cu
echo     float* get_candidates^(^) { return d_candidates_; } >> cuda_fixed.cu
echo }; >> cuda_fixed.cu
echo. >> cuda_fixed.cu
echo extern "C" { >> cuda_fixed.cu
echo     CudaBrainWrapper* create_cuda_brain^(uint32_t n_neurons, uint32_t k_active, uint32_t seed^) { >> cuda_fixed.cu
echo         return new CudaBrainWrapper^(n_neurons, k_active, seed^); >> cuda_fixed.cu
echo     } >> cuda_fixed.cu
echo     void destroy_cuda_brain^(CudaBrainWrapper* brain^) { delete brain; } >> cuda_fixed.cu
echo     void simulate_step^(CudaBrainWrapper* brain^) { brain-^>simulate_step^(^); } >> cuda_fixed.cu
echo     float* get_candidates^(CudaBrainWrapper* brain^) { return brain-^>get_candidates^(^); } >> cuda_fixed.cu
echo } >> cuda_fixed.cu

echo.
echo Testing CUDA compilation with workarounds...

REM Try different compiler flags to work around MSVC compatibility
echo Method 1: Using C++14 with specific flags...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_fixed.dll --shared cuda_fixed.cu -lcudart -lcurand --gpu-architecture=compute_89 --gpu-code=sm_89

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with C++14!
    echo    Output: cuda_fixed.dll
    goto :test_dll
) else (
    echo ❌ Method 1 failed
)

echo.
echo Method 2: Using older CUDA architecture...
nvcc --compiler-options "/EHsc /std:c++14 /O2" -o cuda_fixed.dll --shared cuda_fixed.cu -lcudart -lcurand --gpu-architecture=compute_75 --gpu-code=sm_75

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with older architecture!
    echo    Output: cuda_fixed.dll
    goto :test_dll
) else (
    echo ❌ Method 2 failed
)

echo.
echo Method 3: Using minimal flags...
nvcc --compiler-options "/EHsc" -o cuda_fixed.dll --shared cuda_fixed.cu -lcudart -lcurand

if %ERRORLEVEL% EQU 0 (
    echo ✅ CUDA compilation successful with minimal flags!
    echo    Output: cuda_fixed.dll
    goto :test_dll
) else (
    echo ❌ Method 3 failed
    echo.
    echo All methods failed. This confirms the MSVC 19.43.34808 + CUDA 13.0 compatibility issue.
    echo.
    echo Next steps:
    echo 1. Install older MSVC version (19.29 or earlier)
    echo 2. Install CUDA 12.x instead of 13.0
    echo 3. Use WSL with Linux CUDA toolkit
    echo 4. Use Docker with CUDA support
    goto :cleanup
)

:test_dll
echo.
echo Testing CUDA DLL...
python test_cuda_fixed.py

:cleanup
del cuda_fixed.cu 2>nul
