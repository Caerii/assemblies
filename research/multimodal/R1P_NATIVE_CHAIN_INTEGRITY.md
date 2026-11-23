# R1p — Native affine-chain per-step integrity gate

**Date:** 2026-11-23  
**Canonical branch:** `research/multimodal-continual-assemblies`  
**Status:** executable adapter tests pass on the mounted R1o source bundle; **native numerical execution remains unverified**.

## Motivation

R1o's `run_one()` checks `model.fingerprint()` after each *complete*
composition. This does not rule out a pathological adapter that mutates
the native graph during one atomic projection and restores it on a later
projection. Final-state equality is weaker than per-step immutability,
particularly when evaluating AB versus BA compositions and recurrence.

## Changes implemented

- `NativeAffineChain.predict` now validates the initial k-of-n cue
  **before** calling native code, and rejects empty/unknown role schedules.
- The native model's full explicit-connectome fingerprint is recorded
  before the first step and checked after **every atomic role projection**.
  Any graph mutation terminates evaluation immediately with the offending
  step and role, even if a later step might undo it.
- Every intermediate output is revalidated as a unique, in-range k-of-n
  assembly before feeding it into the next native step.
- Absent intermediate output returns `None` without driving later roles.
- Inputs and targets are passed to native training as validated copies;
  an inner implementation cannot mutate the caller's source buffers.
- Existing R1o API, benchmark factory registration and model-layout estimate
  remain unchanged.

## Local validation

A copy of the actual R1o source archive was extracted to an isolated local
workspace. The revised files passed **44 tests** in the R1o module,
including seven new adversarial tests covering:

1. correct BA role ordering and non-mutating read;
2. transient first-step mutation that would be restored by a later step;
3. second-step mutation;
4. malformed intermediate winner set;
5. invalid external cues and unknown roles;
6. inner training attempting to corrupt caller-owned buffers;
7. early-stop on absent intermediate output.

`python -m compileall` also completed on the R1o research modules.

**Exact Git blob match to tested local files:**

- `native_affine_chain.py`:
  `69a5bdba7f25571753a7f3c7f692f8106f23be6d`
- `test_native_chain_integrity.py`:
  `76b4639d44e53cd095a1e9c1bd416bedd8444e9e`

The Git blob hashes were checked with local `git hash-object` and then
matched to fresh `GitHub.fetch_file` results on the canonical branch.
This guarantees exact source identity for these two files, not that the
entire native repository was executed.

## Native source contract inspection

Read these source blobs directly from the same branch, not from a summary:

| Native module | Inspected Git blob |
|---|---|
| `neural_assemblies/core/brain.py` | `d5739c5d728756b21226e5429250afbfbebdfc87` |
| `neural_assemblies/core/numpy_engine/_explicit.py` | `58a2c0929974752fc7e5ae92c69ea67097c02976` |
| `neural_assemblies/assembly_calculus/ops.py` | `8fbb8d4e185f393bc86072a28e6e0a85a1bac113` |
| `neural_assemblies/assembly_calculus/assembly.py` | `859eecec9f77535e474450f5367d8ebcfd65855e` |

The adapter relies on `Brain.read_only()`, `Brain.project` with
explicit `external_drive` during *training only*, stable neuron-ID
`activate_assembly` / `Assembly.from_area`, and R1h's full graph
fingerprint. The exact backend routes and semantics must still pass a
real native test in a checkout.

## Limitations and next gate

The environment cannot clone GitHub over HTTPS (DNS resolution blocked),
and `neural_assemblies` is not installed here. The mounted R1o bundle
contains the composition benchmark and mocks, **not** the whole native
runtime. No native Brain result, new affine-operator intelligence claim,
Rust speedup or FPGA measurement was generated in this round.

Run on a machine with the actual branch checkout:

```sh
python -m pytest -q research/multimodal/test_native_affine_chain.py \
  research/multimodal/test_native_chain_integrity.py
python -m research.multimodal.native_affine_chain \
  --n 64 --k 8 --atomic-examples 8 --test-cues 8 \
  --data-seed 23 --model-seed 8123
python -m research.multimodal.run_local_gates \
  --output /tmp/assemblies-native-gates-r1p
```

Record actual native failures and correct them on **this same branch**.
Before making a compositionality claim, compare strict AB vs BA exact
recovery, unmutated state, teacher/no-plasticity controls, and native
winners against independent held-out targets. Only then start replacing
high-cost Python kernels with exact-parity compiled Rust.
