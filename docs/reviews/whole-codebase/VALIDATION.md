# Verification of the isolated migration

Worktree: `assemblies-astra-audit-20260909`, based on
`3334876cf57991d27b1651c343f4640b4bcffaa6`. Tests used the existing Python
environment with editable-install finders removed and this worktree first on
`sys.path`. No dependencies were installed and no source files were edited in
the main checkout.

The final combined targeted CPU run passed **122 tests**, skipped **2**, and
emitted the expected sampled-recurrence warning. It covered:

- shared research contracts, runner, capacity configuration and cell identity;
- paired study orchestration and training fingerprints;
- public index boundaries and the controlled CPU teaching investigation;
- golden comparison, register rendering/citations and both methodology ratchets;
- example execution and the missing-real-data MNIST golden path.

The skips were optional Nemo dependencies and unavailable real MNIST data.
The earlier core/model compatibility run passed 83 tests (one deselected), covering
areas, brains, compact-index caches, the exact-engine ladder and examples.
Ruff passed on every changed/new Python file. `git diff --check` passed.

The new pull-request workflow has not run on GitHub. The full package suite,
CUDA compilation, hardware parity, historical scientific verdict recalculation,
and preregistered study runs were not performed. These CPU checks establish
instrument behavior and selected compatibility, not scientific adoption.

The CUDA prerequisite checker found the installed toolkit and Visual Studio
developer script. In the direct-interpreter shell, compiler and ninja were absent
from PATH. The checker did not import torch, compile, or probe the device.

The structural coverage manifest covers the baseline's 2,501 tracked files.
It does not claim line-by-line semantic review of the roughly 309,000 source lines.
`evidence-summary.json` and `evidence-review.tsv` retain the literal-reference
audit's unresolved work without treating candidate orphans as proven dead files.

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for remaining migration and scientific
gates. The run records remain UNJUDGED or VOID; this migration adopts no new
assembly-calculus result.
