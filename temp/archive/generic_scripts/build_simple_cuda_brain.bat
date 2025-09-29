@echo off
echo 🚀 BUILDING SIMPLE CUDA BRAIN
echo ==============================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

echo ✅ Environment ready

echo 🔧 Compiling simple CUDA brain...
nvcc -o simple_cuda_brain.dll --shared --compiler-options "/EHsc /std:c++17 /O2" -lcuda -lcurand simple_cuda_brain.cu

if errorlevel 1 (
    echo ❌ Compilation failed
    exit /b 1
)

echo ✅ Simple CUDA brain compiled successfully!

if exist simple_cuda_brain.dll (
    echo 📊 Build Results:
    echo    - simple_cuda_brain.dll: %~z1 bytes
    echo    - Status: ✅ SUCCESS
    echo.
    echo 🎉 SIMPLE CUDA BRAIN READY!
) else (
    echo ❌ Build failed - no output file
    exit /b 1
)

echo.
echo 🧠 Simple CUDA Brain Features:
echo    ✓ Basic GPU operations
echo    ✓ CUDA kernel execution
echo    ✓ Memory management
echo    ✓ C interface for Python
echo.
echo 🚀 Ready for testing!
