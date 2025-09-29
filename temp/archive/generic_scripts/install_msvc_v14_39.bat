@echo off
echo Installing MSVC v14.39 (VS 2022 17.9) for CUDA 13.0 compatibility...
echo.
echo This will install the compatible MSVC build tools alongside your current version.
echo.
echo Steps:
echo 1. Open Visual Studio Installer
echo 2. Click "Modify" on Visual Studio 2022 Build Tools
echo 3. Go to "Individual Components" tab
echo 4. Search for "MSVC v14.39"
echo 5. Check "MSVC v14.39 - VS 2022 C++ x64/x86 build tools"
echo 6. Click "Modify" to install
echo.
echo After installation, we'll update our build scripts to use the compatible version.
echo.
echo Opening Visual Studio Installer...
start "" "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"

echo.
echo Once installation is complete, run: .\test_cuda_with_v14_39.bat
echo.
pause
