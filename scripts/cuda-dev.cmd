@echo off
rem Run from cmd.exe, or: cmd /k scripts\cuda-dev.cmd
rem Intentionally no setlocal: the caller needs the compiler environment.
set "ASSEMBLIES_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%ASSEMBLIES_VSWHERE%" (
  echo Missing vswhere.exe. Install Visual Studio C++ build tools.
  exit /b 1
)
set "ASSEMBLIES_VSINSTALL="
for /f "usebackq delims=" %%i in (`"%ASSEMBLIES_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "ASSEMBLIES_VSINSTALL=%%i"
if not defined ASSEMBLIES_VSINSTALL (
  echo No Visual Studio installation with C++ x64 build tools was found.
  exit /b 1
)
if not exist "%ASSEMBLIES_VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
  echo Missing vcvars64.bat in the detected Visual Studio installation.
  exit /b 1
)
call "%ASSEMBLIES_VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not defined CUDA_HOME if defined CUDA_PATH set "CUDA_HOME=%CUDA_PATH%"
uv run python "%~dp0check_cuda_toolchain.py"
exit /b %errorlevel%
