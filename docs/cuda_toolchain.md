# Preparing the fused CUDA build

The maintained extension is compiled by
`neural_assemblies/core/torch_engine/_fused_cuda.py` on demand. The build scripts
under `cpp/` target other prototypes and are not its setup procedure.

From the repository root on Windows:

```powershell
uv sync --extra gpu
cmd /k scripts\cuda-dev.cmd
```

The second command opens a command shell, discovers Visual Studio through
`vswhere`, loads its x64 developer environment, and checks the prerequisites.
Run subsequent commands in that shell: a child command shell cannot change the
environment of its parent PowerShell session. If already in cmd.exe, use
`call scripts\cuda-dev.cmd`.

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

After all active GPU studies have finished, run the extension tests in the
prepared shell:

```cmd
uv run pytest neural_assemblies/tests/test_fused_cuda.py neural_assemblies/tests/test_hashed_substrate_parity.py -q
```

These may compile the extension. Do not rebuild it while a process holds it,
and do not run these tests beside a GPU study. Hardware parity is a separate
gate from the fast CPU research-contract tests.
