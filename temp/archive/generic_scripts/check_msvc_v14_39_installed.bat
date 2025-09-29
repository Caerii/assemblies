@echo off
echo Checking if MSVC v14.39 was installed correctly...

REM Check for MSVC v14.39 in common locations
set MSVC_V14_39_PATH1="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set MSVC_V14_39_PATH2="C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set MSVC_V14_39_PATH3="C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set MSVC_V14_39_PATH4="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"

echo.
echo Checking for MSVC v14.39 installation...

if exist %MSVC_V14_39_PATH1%\cl.exe (
    echo ✅ MSVC v14.39 found at: %MSVC_V14_39_PATH1%
    set FOUND_PATH=%MSVC_V14_39_PATH1%
    goto :test_version
)

if exist %MSVC_V14_39_PATH2%\cl.exe (
    echo ✅ MSVC v14.39 found at: %MSVC_V14_39_PATH2%
    set FOUND_PATH=%MSVC_V14_39_PATH2%
    goto :test_version
)

if exist %MSVC_V14_39_PATH3%\cl.exe (
    echo ✅ MSVC v14.39 found at: %MSVC_V14_39_PATH3%
    set FOUND_PATH=%MSVC_V14_39_PATH3%
    goto :test_version
)

if exist %MSVC_V14_39_PATH4%\cl.exe (
    echo ✅ MSVC v14.39 found at: %MSVC_V14_39_PATH4%
    set FOUND_PATH=%MSVC_V14_39_PATH4%
    goto :test_version
)

echo ❌ MSVC v14.39 not found in common locations
echo.
echo Let's check what MSVC versions are available...
echo.

REM Check what MSVC versions are installed
for /d %%i in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*") do (
    echo Found MSVC version: %%i
)

for /d %%i in ("C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*") do (
    echo Found MSVC version: %%i
)

for /d %%i in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*") do (
    echo Found MSVC version: %%i
)

for /d %%i in ("C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\*") do (
    echo Found MSVC version: %%i
)

echo.
echo If MSVC v14.39 is not found, please:
echo 1. Check Visual Studio Installer
echo 2. Look for "MSVC v14.39 - VS 2022 C++ x64/x86 build tools"
echo 3. Make sure it's installed
goto :end

:test_version
echo.
echo Testing MSVC v14.39 version...
%FOUND_PATH%\cl.exe 2>&1 | findstr "Version"

echo.
echo MSVC v14.39 is ready for CUDA compilation!
echo.
echo Next step: Test CUDA compilation with MSVC v14.39
echo Run: .\test_cuda_compilation_v14_39.bat

:end
echo.
echo Check complete.

