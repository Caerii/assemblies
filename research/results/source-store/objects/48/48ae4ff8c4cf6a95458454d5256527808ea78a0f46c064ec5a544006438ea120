@echo off
rem The GPU gate: every test that needs CUDA or the fused kernels, run so that
rem a skip cannot pass for a pass.
rem
rem   scripts\gpu-gate.cmd                 the gate (fast tier)
rem   scripts\gpu-gate.cmd -m "gpu"        include the slow GPU tests too
rem   scripts\gpu-gate.cmd -k parity       any extra pytest arguments pass through
rem
rem It enters the compiler environment (scripts\cuda-dev.cmd), then runs pytest
rem with ASSEMBLIES_REQUIRE_DEVICE=fused, which turns "the kernels did not
rem build" from a skip into a failure (neural_assemblies\tests\_devices.py).
rem Without that setting a broken toolchain turns the parity gates into skips
rem and the run still reads green. Run it from cmd.exe, from the repository
rem root, with no other GPU job running.
call "%~dp0cuda-dev.cmd"
if errorlevel 1 (
  echo GPU gate: the compiler environment is not ready; see the checker output above.
  exit /b 1
)
set "ASSEMBLIES_REQUIRE_DEVICE=fused"
set "GATE_SELECT=gpu and not slow"
for %%a in (%*) do if "%%~a"=="-m" set "GATE_SELECT="
if defined GATE_SELECT (
  uv run --frozen python -m pytest neural_assemblies/tests -o addopts= -q -p no:cacheprovider -m "%GATE_SELECT%" %*
) else (
  uv run --frozen python -m pytest neural_assemblies/tests -o addopts= -q -p no:cacheprovider %*
)
exit /b %errorlevel%
