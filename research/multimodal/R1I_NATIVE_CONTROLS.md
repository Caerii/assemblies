# R1i — paired teacher controls and executable local gates

**Branch:** `research/multimodal-continual-assemblies`.
**Status:** controlled comparison implemented; native numerical results NOT established.
**Date:** 2026-11-23. No GitHub Actions required or assumed.

## Why this closes a major gap

The native teacher-driven adapter R1h now supports a configurable plasticity
rate (`beta=0` is the causal no-learning control). R1i introduces three
SEED-MATCHED conditions operating on the same independent-target dataset:

1. **correct_teacher:** observed (cue, role, target) training pairs; native
   `beta=0.1`, teacher drive during the write only.
2. **shuffled_teacher:** same pair keys and number of writes, with targets
   cyclically permuted *only among training pairs of the same role*;
   `beta=0.1`. No held-out targets enter training.
3. **no_plasticity:** same observed correct targets and the same
   teacher drive, but the native target and lexical areas use `beta=0`.

Each arm instantiates its own native `Brain` with the same seed. Readout
never includes teacher drive, and queries are evaluated against the SAME
ground-truth target codes in all arms. Correct-teacher, shuffled-teacher,
and no-plasticity outcomes can now be compared on trained exact recovery,
held-out exact recovery, strict positive-margin rank-1, overlap and repeated
output consistency. Do not use only trained exact accuracy to assert
generalization. Tied predictions are not successes.

The backend-neutral implementation is `teacher_control_matrix.py`.
The executable native adapter is `native_teacher_controls.py`.
The scheduling tests are `test_teacher_control_matrix.py`.

## Local execution evidence (NOT native Brain evidence)

In a Python 3.13 + NumPy 2.3.5 + pytest 9.0.2 sandbox, an independent
local copy of the backend-neutral control matrix and the local gate
runner was executed:

- `pytest test_teacher_control_matrix.py test_run_local_gates.py -q`:
  **12 tests passed** (7 control tests, 5 execution-ledger tests).
- The local `run_local_gates.py` dry run with two chosen gates generated
  `run.json` with `all_passed=false` and gate status `not-executed`
  for both, as required. DRY RUN IS NOT A SUCCESSFUL NATIVE EXPERIMENT.
- The local runner tests verified stdout/stderr retention, exit status,
  SHA-256 recording, timeouts, and failure propagation.
- The full `neural_assemblies` package and Rust compiler were unavailable
  in that sandbox, and the native experiments were NOT run.

## Native execution — single-command protocol

From a checkout of this exact research branch, using your normal local
Python environment with native package dependencies installed:

```sh
python -m research.multimodal.run_local_gates \
  --output /tmp/assemblies-native-gates-20261123
```

The default gate matrix runs protocol tests and lightweight smoke variants
of R1b, R1c, R1d, R1g, R1h and R1i. It captures individual stdout/stderr
files, hashes them, records duration/exit status, and writes `run.json`
with the Git HEAD SHA. A nonzero gate, exception or timeout fails the
overall command. There is NO external CI dependence.

For a specific R1i native experiment at fuller scale:

```sh
python -m research.multimodal.native_teacher_controls \
  --family cyclic_shift --seed 42 --n 512 --k 16 --contents 12 \
  --output /tmp/assemblies-r1i-seed42.json
```

Repeat over preregistered seeds and nuisance parameters only after
reviewing each smoke failure. Preserve logs on this branch with a new
results manifest and appropriate generated file hashes.

## Interpretation and next gates

- Native actual teacher-clamp winning fraction and source-to-target
  synaptic changes still need direct measurement. A large external
  teacher drive may force winners without forming a useful fiber.
- Repeated identical predictions are NOT proof of no mutation; add
  a native before/after weight/connectome fingerprint test.
- No positive zero-shot native learning claim has been established.
- Compare R1i against the train-only `ShiftRoleLearner` (which receives
  the correct shift-hypothesis family) as well as pair-memory controls.
- The cyclic-shift task is intentionally structured and learnable;
  the random-permutation task is generally underidentified from the
  available training pairs.
- Do NOT claim native or FPGA acceleration until compiled execution and
  end-to-end p50/p95 throughput benchmarking have been performed.
