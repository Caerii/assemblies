# Preparing the fused CUDA build

The maintained extension is compiled by
`neural_assemblies/core/torch_engine/_fused_cuda.py` on demand. The build scripts
under `cpp/` target other prototypes and are not its setup procedure.

From the repository root on Windows:

```powershell
uv sync
cmd /k scripts\cuda-dev.cmd
```

The second command opens a command shell, discovers Visual Studio through
`vswhere`, loads its x64 developer environment, and checks the prerequisites.
Run subsequent commands in that shell: a child command shell cannot change the
environment of its parent PowerShell session. If already in cmd.exe, use
`call scripts\cuda-dev.cmd`.

Discovery defaults to Visual Studio versions `[16.0,18.0)` (2019/2022), since
this machine's CUDA 13.1 rejects the newer VS 2026 compiler. The batch helper
uses the Python checker's discovery function, so both select the same range.
Set `ASSEMBLIES_VS_VERSION` before setup to choose a different vswhere version
range for another verified toolkit; tool discovery alone does not prove that
combination compiles. The actual compiler in PATH is reported separately.

For an isolated checkout, set an isolated extension cache in the prepared shell.
Include the compiler version: Ninja can otherwise reuse objects compiled by a
different cl.exe after PATH changes. Choose a fresh cache when changing other
toolchain components as well:

```cmd
set "TORCH_EXTENSIONS_DIR=%CD%\.cache\torch-extensions-msvc-%VCToolsVersion%"
```


The checker identifies missing `vcvars64.bat`, `cl.exe`, `ninja`, torch,
and `CUDA_HOME`/`CUDA_PATH` with its `bin/nvcc.exe`. If ninja is missing,
install it in the environment used for the build. Set CUDA_HOME to the installed
toolkit root if neither CUDA variable is set.

You can run the checker independently, without compiling or using the GPU:

```powershell
uv run python scripts/check_cuda_toolchain.py --json
```

Exit code 1 means a prerequisite is missing. Exit code 0 means the tools were
found; it does not establish compiler compatibility, a working device, successful
compilation, or numerical parity.

## The GPU gate

After all active GPU studies have finished, from cmd.exe at the repository
root:

```cmd
scripts\gpu-gate.cmd
```

It enters the developer environment through `cuda-dev.cmd`, then runs every
test that needs CUDA or the fused kernels (`-m "gpu and not slow"`; pass
`-m gpu` to include the slow ones, or any other pytest arguments). It sets
`ASSEMBLIES_REQUIRE_DEVICE=fused`, and that is what makes it a gate: a test
whose kernels did not build FAILS instead of skipping. Without the setting a
broken toolchain turns every parity test into a skip and the run reads green.

Every device requirement in the test suite goes through one module,
`neural_assemblies/tests/_devices.py`, as a marker (`requires_torch`,
`requires_cuda`, `requires_fused`, `requires_cupy`) or the `fused_kernels`
fixture. On an ordinary machine a missing level skips with its reason (the
compiler error, the missing import); `ASSEMBLIES_REQUIRE_DEVICE` names the
levels that must be present (`torch`, `cuda`, `fused`, `cupy`, or `all`; a
level implies the ones below it). The CPU job in CI installs the CPU build of
the locked torch and sets `ASSEMBLIES_REQUIRE_DEVICE=torch`.
`test_device_gate.py` holds the true negative and refuses any test module
that asks the device question in its own way.

The gate may compile the extension. Do not rebuild it while a process holds
it, and do not run it beside a GPU study. Hardware parity is a separate gate
from the fast CPU research-contract tests.
