# Inventory with dispositions: cpp/ and legacy/ (2026-09-12)

> Drafted by a read-only pass over the 293 tracked files at commit 6754199d, grouped into 25 clusters; every consumer claim is a grep hit cited as path:line. Dispositions are STAY / MOVE / ARCHIVE / DELETE-CANDIDATE; the backlog's rule stands: nothing is removed without a passing replacement and an evidence audit. This file is the disposition record TODO.md asks for under 'Review every tracked cpp/ and legacy-tree file'; decisions on the three clusters named in its summary are the open work.

# cpp/ and legacy/ inventory with dispositions (draft, 2026-09-12)

Worktree read: `C:\Users\locke\AppData\Local\Temp\assemblies-astra-audit-20260909` (read-only). `git ls-files cpp legacy` = 293 files
(cpp 186, legacy 107). Backlog item: TODO.md:192-194. Every consumer claim below is a grep hit (path:line); "none found" otherwise.

Facts that apply to both trees
- Neither tree ships or is tested by default: `pyproject.toml:75-77` packages = `neural_assemblies*`; `pyproject.toml:140` testpaths = `neural_assemblies/tests`.
- CI: `.github/workflows/research-contracts.yml:37-38` runs `test_legacy_result_storage.py`/`test_legacy_aggregate_summary.py` (about legacy *result files*, not `legacy/`); `publish.yml` has no cpp/legacy reference. No CI consumer of either tree.
- The pytest marker `legacy` (`pyproject.toml:162`) has no user (grep `mark.legacy`: none). The two root-level legacy tests are run by hand: `docs/api.md:339`.
- `docs/reviews/whole-codebase/coverage.tsv` and `inventory.json` list every file; they are inventories, not consumers. `.cache/` and `build/` copies are untracked (git ls-files: 0).
- `docs/cuda_toolchain.md:3-5`: the maintained CUDA extension is built on demand by `torch_engine/_fused_cuda.py`; "the build scripts under `cpp/` target other prototypes and are not its setup procedure."
- `docs/gpu_scale_design.md:338-343`: GPU levers were "superseded for research use by the hashed substrate (`core/torch_engine/_hashed.py`)"; `docs/RUST_KERNELS.md` covers `crates/na-kernels` (initialiser only). Neither doc names any `cpp/` file as current.
- Last touch: cpp/core_cpp, cpp/documentation, cpp/build_scripts 2025-09-29; the rest of cpp/ 2026-02-06; legacy/root_modules, matlab, artifacts, experiments 2026-04-22; legacy/root_shims and scripts 2026-09-09.

## cpp/ (186 files)

### C1. `cpp/README.md` (1) -- STAY
What: entry doc; declares cpp/ "engineering infrastructure, not the default Python package API". Consumers: `docs/architecture.md:183`, `docs/supported_surfaces.md:51`, `docs/cuda_toolchain.md:5`.
Build/spec/replacement: n/a. Defect: its "Local Build Sketch" says `python setup.py build_ext --inplace`, but no `setup.py`, `CMakeLists`, or Bazel `BUILD` is tracked under cpp/ (`.gitignore:4 cpp/bazel-*` is the remnant). Fix the sketch when C3 is decided.

### C2. `cpp/build_scripts/` (3: README_BUILD.md, build.bat, build.env.example) -- ARCHIVE
What: nvcc "superset" build for the assemblies_cuda_* DLLs; hard-codes CUDA 13.0, MSVC 14.39, RTX 4090 (`build.bat:24-31`). Consumers: none found outside cpp/. Build path: it is the build path for C4 (`build.bat:167-194`). Spec: README_BUILD.md, but `README_BUILD.md:35` names `cuda_kernels.cu -> cuda_kernels_v14_39.dll` while `build.bat:167` compiles `assemblies_cuda_kernels.cu` (doc/script mismatch). Replacement: `docs/cuda_toolchain.md`, `scripts/check_cuda_toolchain.py`, `scripts/cuda-dev.cmd`. Becomes DELETE-CANDIDATE jointly with C4.

### C3. `cpp/core_cpp/` (5: brain.h, brain_util.h, brain.cc, brain_test.cc, pybind11_wrapper.cpp) -- STAY (flagged)
What: the original C++ Brain (`NEMO_BRAIN_H_`) plus a pybind11 module `brain_cpp`. Consumers (maintained): `neural_assemblies/core/brain_cpp.py:14` (`importlib.import_module("brain_cpp")`), `neural_assemblies/simulation/association_simulator_cpp.py:9`, `tests/performance/test_cpp.py:8`, `tests/performance/test_cuda.py:30-37`, cpp/README.md usage sketch.
Build path: none tracked (see C1). Spec: cpp/README.md. Replacement: none for the pybind path; supported engines are numpy/torch/hashed (`docs/supported_surfaces.md`).
Decision needed: the wrappers cannot import a module nobody can build from the tree. Either add a tracked build (pyproject optional target) or retire `brain_cpp.py`, `association_simulator_cpp.py`, `tests/performance/test_cpp.py` and this cluster together. Not a delete candidate while the wrappers exist.

### C4. `cpp/cuda_kernels/` assemblies_cuda_* family (11: assemblies_cuda_{brain,brain_optimized,kernels,kernels_optimized,memory,memory_optimized,wrapper}.cu, assemblies_cuda_optimized.h, cuda_brain.h, cuda_kernels.cu, simple_cuda_brain.cu) -- DELETE-CANDIDATE (jointly with C9-C16)
What: dense n x n / curand ctypes DLL kernels for the "universal brain simulator". Consumers: `cpp/python_implementations/core_implementations/universal_brain_simulator/cuda_manager.py:75,97` (loads assemblies_cuda_brain_optimized.dll, assemblies_cuda_kernels.dll) -- inside cpp/ only; outside cpp/: none found. Build: C2 `build.bat:167-194` (assemblies_cuda_*, simple_cuda_brain); `cuda_kernels.cu`, `cuda_brain.h`: no build script. Defect: `assemblies_cuda_brain.cu:1` includes `cuda_brain_fixed.h`, which is not tracked. Spec: `cpp/documentation/gpu_acceleration_analysis.md`. Replacement: `neural_assemblies/core/cuda_engine.py` + `core/kernels/{implicit,sparse_ops}.py`, `torch_engine/_fused_cuda.py`, `torch_engine/_hashed.py` (`docs/gpu_scale_design.md:338`).

### C5. `cpp/cuda_kernels/` nemo_implicit_kernels.cu, dense_assembly_kernels.cu, build_all.bat, build_nemo_kernels.bat, build_dense_kernels.bat (5) -- STAY
What: hash-based implicit-connectivity kernels for the nemo emergent-language brain; dense kernels are its fallback. Consumers (maintained): `neural_assemblies/nemo/language/emergent/cuda_backend.py:21-22` (DLL_DIR = `cpp/dlls`, `nemo_implicit_kernels.dll`), `:54` (dense fallback); imported by `neural_assemblies/nemo/language/emergent/brain.py:36` and `interactive/interactive_learner.py:64`; also `cpp/python_implementations/benchmarks/custom_cuda_wrapper.py:15` (dense). Build: `build_all.bat` steps [1/4]-[2/4], `build_nemo_kernels.bat`, `build_dense_kernels.bat` -> `cpp/dlls/` (untracked; `.gitignore:18 *.dll`). Spec: cuda_backend.py docstring only; none in docs/. Replacement: `core/cuda_engine.py:14` ("hash function matches CUDA kernels in kernels/implicit.py") -- but the nemo subtree still binds to the DLL. `research/notes/language/nemo_subtree_audit.md:88` recommends removing `nemo/`; if adopted, this cluster follows it.

### C6. `cpp/cuda_kernels/` assembly_projection_kernel.cu, sparse_assembly_kernels.cu, sparse_assembly_kernels_v2.cu, build_projection_kernel.bat (4) -- DELETE-CANDIDATE (jointly with C13)
What: billion-scale k x k sparse kernels and an "ultra-optimized" projection kernel. Consumers: `cpp/python_implementations/benchmarks/billion_scale_benchmark.py:22-23` (v1/v2 DLL paths) -- inside cpp/; outside: none found. Build: `build_all.bat` [3/4]-[4/4] (builds v2 only; v1 has no build path), `build_projection_kernel.bat`. Spec: none. Replacement: `torch_engine/_hashed.py` present-only layout (`docs/gpu_scale_design.md:338-343`, `research/notes/substrate/DESIGN_present_only.md`).

### C7. `cpp/cuda_kernels/tests/` (11: 5 .cu microbenchmarks, build_tests.bat, build_tests_simple.bat, run_tests.bat, README.md, test_algorithmic_improvements.py, algorithmic_improvements_test_20250923_113504.json) -- ARCHIVE
What: standalone warp-reduction / radix-select / coalescing benchmarks and their result JSON. Consumers: `research/plans/control/CONTROL_SPEED_FROM_CUDA.md:35` cites the JSON as its table 2.1. Build: `build_tests.bat:108-125` (nvcc per .cu -> .exe). Spec: tests/README.md. Replacement: `neural_assemblies/tests/test_fused_cuda.py`, `neural_assemblies/benchmarks/throughput.py`. Keep the producer with the cited number; option: MOVE the JSON to `research/results/` and update the plan's path.

### C8. `cpp/documentation/` (7 .md: paper drafts, research roadmap, GPU analysis, critique) -- ARCHIVE
What: 2025-09 "billion-scale assembly calculus" paper drafts and roadmap. Consumers: none found. Build/spec: n/a. Replacement: `docs/gpu_scale_design.md`, `research/plans/PAPERS.md`. Option: MOVE to `research/notes/archive/` if any claim is ever cited (none is).

### C9. `cpp/python_implementations/core_implementations/universal_brain_simulator/` + `core_implementations/README.md` + `README_ORGANIZATION.md` (24) -- DELETE-CANDIDATE
What: modular CuPy/ctypes simulator package (client, config, cuda_manager, memory_manager, ...) with examples/ and a monolithic_old/ copy. Consumers: siblings only (`core_implementations/README.md` Quick Start `from universal_brain_simulator.client import BrainSimulator`); outside cpp/: none found. Build: none (run from its own directory per README). Spec: its README.md and C12. Replacement: `core/cuda_engine.py`, `torch_engine/`, `neural_assemblies/benchmarks/throughput.py`. `README_ORGANIZATION.md` lists source files (working_cuda_brain_v14_39.py, ...) that are not tracked.

### C10. `core_implementations/tests/` (17) + `core_implementations/experiments/` (6) (23) -- DELETE-CANDIDATE
What: scripts against C9 (large-scale, extreme-limit, cleanup, oscillation demo, bottleneck sweep, fix_errors.py). Consumers: none found; not collected (`pyproject.toml:140`). Build: none. Spec: their README.md files. Replacement: `neural_assemblies/tests/test_cuda_kernels.py`, `tests/performance/`.

### C11. `core_implementations/optimized_implementations/` (13 incl. 4 result JSON) + `core_implementations/results/` (6 JSON) (19) -- DELETE-CANDIDATE
What: O(N log K) variants, their comparison scripts, and unregistered timing JSON. Consumers: none found (no research/notes registration, no `theory.py` edge; `theory.py` grep for cpp/ paths: none). Replacement: `core/kernels/sparse_ops.py`, `torch_engine/_hashed.py`. Evidence audit: none of the JSON is cited anywhere.

### C12. `core_implementations/analysis/` (13 .md) -- ARCHIVE
What: refactoring, quantization, oscillation and memory-access notes for C9. Consumers: none found. Replacement: `docs/gpu_scale_design.md`. Same option as C8.

### C13. `billion_scale/` (9) + `benchmarks/` (5) + `optimizations/` (2) (16) -- DELETE-CANDIDATE
What: standalone billion-scale CuPy/GPU scripts and DLL wrappers. Consumers: `benchmarks/custom_cuda_wrapper.py:15` and `billion_scale_benchmark.py:22-23` consume C5/C6 DLLs; outside cpp/: none found. Build: none. Spec: `README_ORGANIZATION.md`. Replacement: `torch_engine/_hashed.py` batching (`docs/gpu_scale_design.md:338-343`), `neural_assemblies/benchmarks/`.

### C14. `analysis_tools/` (9) -- DELETE-CANDIDATE
What: Hodgkin-Huxley / Purkinje / "realistic brain" analyzers and `assembly_calculus_implications.py`; not assembly-calculus runtime. Consumers: none found. Build/spec: none (`README_ORGANIZATION.md` lists them). Replacement: none needed (out of scope of the package); `research/experiments/` is where such studies belong.

### C15. `profilers/` (8 scripts) + `profilers/__generated__/` (21: 8 PNG, 13 JSON, 13 MB) (29) -- DELETE-CANDIDATE
What: profilers for C9 and their generated plots/JSON. Consumers: none found. Replacement: `neural_assemblies/benchmarks/profile_operations.py`, `benchmarks/throughput.py`. The directory name says generated; nothing registers the numbers.

### C16. `cpp/tests/` (6) -- DELETE-CANDIDATE except `test_ms_per_step.py` (ARCHIVE)
What: ad-hoc GPU memory / CuPy / scipy scripts. Broken imports: `test_extreme_scale_cuda.py:10` (`cuda_brain_python`, untracked), `test_ms_per_step.py:6` (`ultra_optimized_cuda_brain_v2`, untracked), `test_gpu_memory_cuda_brain.py:18` (`gpu_memory_cuda_brain.dll`, no source tracked). Consumers: `research/plans/control/CONTROL_SPEED_FROM_CUDA.md:48` cites `cpp/tests/test_ms_per_step.py` (a script that cannot run). Build: none. Replacement: `tests/performance/test_cuda.py`, `test_cuda_env.py`, `test_simple_cuda.py`.

## legacy/ (107 files)

### L1. `legacy/README.md`, `legacy/__init__.py` (2) -- STAY
What: index of the archive and its rule ("new runtime code in neural_assemblies/"). Consumers: `docs/architecture.md:167-174`, `docs/supported_surfaces.md:58-59`, `docs/project_context.md:68`. Replacement: n/a. Defect: `legacy/README.md:20-21` lists `experiments/` (see L9).

### L2. `legacy/root_shims/` (8: README + brain, brain_util, image_learner, learner, parser, recursive_parser, simulations) -- STAY
What: 3-line re-exports so `import brain` etc. work with the directory on PYTHONPATH; `brain.py` routes to the package, the rest to L3. Consumers: `tests/test_legacy_root_shims.py:6-52`, `tests/test_legacy_archived_layout.py:18`, `docs/api.md:328-339`, `docs/supported_surfaces.md:25-38` (incl. the `robust_grammatical_brain.py` exception), `docs/architecture.md:163`; L3 and L5 scripts need it (`learner.py:1`, `turing_sim.py:14`, `overlap_sim.py:23`, `test_brain_core.py:1` do `import brain`). Build: none. Spec: root_shims/README.md. Replacement: `neural_assemblies.core` (for `brain`); none for the others.

### L3. `legacy/root_modules/` (7: brain_util, image_learner, learner, parser, recursive_parser, simulations, __init__) -- STAY
What: Mitropolsky's original implementations (4.6k lines). Consumers: `tests/test_legacy_root_shims.py:29-33` (module identity), `neural_assemblies/tests/test_index_space_ratchet.py:61,219,223,238,239` (frozen baseline rows), `research/literature/parity/configs/pnas2020.yaml:31,80`, `configs/colt2022.yaml:32,56`, `parity/golden/colt2022_mnist.json:26`, `reproduction_matrix_supplement.json:88,373`, `build_matrix_supplement.py:19,40`, `research/notes/language/nemo_subtree_audit.md:83`. Deps: `image_learner.py:7` imports L7; `parser.py:3` `pptree` (declared, `pyproject.toml:40`). Build: none. Spec: legacy/README.md, root_shims/README.md. Replacement: `neural_assemblies/simulation/*` (simulations.py), `assembly_calculus/emergent/parser.py` (parser, recursive_parser); none for learner.py / image_learner.py (parity matrix: "Port to package or pin legacy" P3, "TODO port" P1).

### L4. `legacy/matlab/` (7 .m) -- ARCHIVE
What: MATLAB prototypes (assembly_sim, merge, reciprocal_project, randomized_project, kernel). Consumers: `research/literature/reproduction_matrix_supplement.json:1618` (XINF-C02, gap "Cross-lang golden"), `build_matrix_supplement.py:135`, `legacy/root_shims/brain.py:5`. Build: none. Spec: legacy/README.md. Replacement: none (the cross-language golden was never produced). Keep; cited.

### L5. `legacy/scripts/simulations/` (6) + `legacy/scripts/README.md`, `legacy/scripts/__init__.py` (8) -- STAY
What: overlap_sim, project, turing_sim, run_turing_sim, and the Python 2 `test_brain_core.py` (`xrange`; not collected). Consumers: `tests/test_legacy_archived_layout.py:8-14` (asserts each path exists), `neural_assemblies/tests/test_index_space_ratchet.py:73,229,230`, `docs/reviews/2026-09-09-codebase-and-research-organization.md:74`. Build: none. Spec: legacy/scripts/README.md. Replacement: `neural_assemblies/simulation/turing_simulations.py` (turing_sim), `projection_simulator.py` / `association_simulator.py` (project, overlap).

### L6. `legacy/scripts/tooling/build_cuda_simple.py` + `__init__.py` (2) -- ARCHIVE
What: nvcc smoke build with a hard-coded CUDA 13.0 path (`build_cuda_simple.py:17`). Consumers: `tests/test_legacy_archived_layout.py:12` only. Replacement: `scripts/check_cuda_toolchain.py`, `docs/cuda_toolchain.md`. If that assertion row is retired, DELETE-CANDIDATE.

### L7. `legacy/scripts/visualization/animator.py` + `__init__.py` (2) -- STAY
What: matplotlib GIF writer for the image learner; writes to `artifacts/.../animations` (`animator.py:17-19`). Consumers: `legacy/root_modules/image_learner.py:7`, `tests/test_legacy_archived_layout.py:13`. Replacement: none.

### L8. `legacy/artifacts/image_learning/` (70: 66 GIF, 4 results.txt; 114 MB = 96 MB animations incl. "archived experiments" 2-10, 19 MB experiment_data connectomes) -- ARCHIVE (relocate storage)
What: outputs of the CIFAR-10 image-learning runs. Consumers: none found outside legacy/ (`docs/architecture.md:172` lists the directory only). Producer: L7. Build/spec: none; the four `results.txt` files are the only record of experiments 7-10. Replacement: none. Recommendation: move the GIFs out of the working tree (git LFS, release asset, or a `research/results/archive/` tarball) and keep the results.txt; an evidence audit precedes any deletion.

### L9. `legacy/experiments/README.md` (1) -- ARCHIVE (rewrite)
What: describes `hyperdimensional_assemblies/assembly_hdc_investigation.py`, which is not tracked anywhere (git ls-files: only `neural_assemblies/compute/hyperdimensional.py`, `neural_assemblies/tests/test_hyperdimensional_contract.py`). Consumers: `docs/architecture.md:173`, `legacy/README.md:20-21` list the directory. Replacement: `neural_assemblies/compute/hyperdimensional.py`. Rewrite to one line pointing at the package module, or drop the directory and both bullets; DELETE-CANDIDATE once those two bullets go.

## Summary

Counts (clusters / files): STAY 8 / 38 (C1, C3, C5, L1, L2, L3, L5, L7); ARCHIVE 8 / 115 (C2, C7, C8, C12, L4, L6, L8, L9, plus `cpp/tests/test_ms_per_step.py`); DELETE-CANDIDATE 9 / 140 (C4, C6, C9, C10, C11, C13, C14, C15, C16 minus one file); MOVE 0 primary (secondary options noted in C7, C8, C12, L8). Total 293.

Nothing marked DELETE-CANDIDATE is referenced by a test, a doc under docs/, a registration under research/notes/, or a `theory.py` evidence edge (theory.py's "legacy" strings refer to legacy result files, not to `legacy/`). Every delete candidate has a maintained replacement named above; the backlog still requires the passing replacement plus an evidence audit before removal.

Three decisions that free the most maintenance at the least risk:
1. The universal-brain-simulator stack (C4, C6, C9-C11, C13-C16: ~140 files, 1.5 MB Python + 13 MB generated) -- zero consumers outside cpp/, broken includes/imports (C4, C16), superseded by `cuda_engine.py` / `torch_engine`. One reviewable removal commit after the evidence audit; keep C7's cited JSON and `test_ms_per_step.py` (or move the plan's citations).
2. `legacy/artifacts/` (L8: 70 files, 114 MB, the bulk of both trees' weight) -- zero consumers; relocating the GIFs out of the checkout and keeping the four results.txt costs nothing scientifically.
3. The 20 markdown drafts in C8 + C12 -- zero consumers, no claims cited; archive in place or move under `research/notes/archive/`.
Also decide C3 (`cpp/core_cpp`): the maintained `brain_cpp.py` / `association_simulator_cpp.py` wrappers import a module with no tracked build; restore a build or retire the wrappers with the cluster.
