@echo off
rem Run from cmd.exe, or: cmd /k scripts\cuda-dev.cmd
rem Intentionally no setlocal: the caller needs the compiler environment.
set "ASSEMBLIES_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%ASSEMBLIES_VSWHERE%" (
  echo Missing vswhere.exe. Install Visual Studio C++ build tools.
  exit /b 1
)
rem Compiler selection and its configurable version range live in the checker.
set "ASSEMBLIES_VCVARS64="
for /f "usebackq delims=" %%i in (`uv run python "%~dp0check_cuda_toolchain.py" --print-vcvars64`) do set "ASSEMBLIES_VCVARS64=%%i"
if not defined ASSEMBLIES_VCVARS64 (
  echo No Visual Studio C++ x64 build tools match the requested ASSEMBLIES_VS_VERSION range.
  exit /b 1
)
if not exist "%ASSEMBLIES_VCVARS64%" (
  echo Missing vcvars64.bat in the detected Visual Studio installation.
  exit /b 1
)
call "%ASSEMBLIES_VCVARS64%"
if errorlevel 1 exit /b 1
if not defined CUDA_HOME if defined CUDA_PATH set "CUDA_HOME=%CUDA_PATH%"
uv run python "%~dp0check_cuda_toolchain.py"
exit /b %errorlevel%
