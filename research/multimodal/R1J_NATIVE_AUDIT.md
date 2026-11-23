# R1j — Explicit-synapse audit and fail-closed local gates

**Date:** 2026-11-23  
**Canonical branch:** `research/multimodal-continual-assemblies`  
**Status:** Code and independent pure-Python tests committed; native Brain execution still pending.

## Why this milestone matters

Repeatedly returning the same winners is NOT evidence that `Brain.read_only`
preserves learned structure. Likewise, a teacher signal can force the correct
winner set even if no Hebbian synapse was modified. Previously R1h/R1i did not
measure those two failure modes directly.

The native NumPy explicit engine's `project_into` implementation was
inspected. For active `LEX` source winners and selected role target winners,
the backend updates their existing synapses by multiplying by `(1+beta)`
and clipping at the configured ceiling. An external teacher drive enters
before k-WTA selection; it can change the winners without necessarily
changing a single eligible edge. Those are distinct events and must be
reported independently.

## New implementation

1. `learning_audit.py` provides `active_synapses`,
   `write_diagnostic`, `weight_fingerprint`, and
   `full_explicit_fingerprint`. It verifies that native explicit storage
   is present, finite and dense; missing internals or mixed engine ownership
   fail with errors rather than allowing false success.
2. `NativeTeacherDriven.train` now measures active source-target edge
   changes before and after the teacher-driven write, plus the fraction
   of teacher target neurons selected. A `beta=0` write changing active
   edges raises an error.
3. `NativeTeacherDriven.state_fingerprint` hashes **every area-to-area
   matrix** in the common explicit engine, including recurrent and reverse
   pathways, not just the directly taught `LEX -> role` matrix.
4. `teacher_control_matrix.run_conditions` compares this digest after
   each individual prediction to the post-training digest. Any difference
   fails the experiment. A model without this optional protocol reports
   that the audit was unavailable, rather than pretending its output
   repeatability proved state immutability.
5. `native_teacher_controls.run` requires all three native arms to
   expose a complete weight fingerprint and one write diagnostic per
   training pair. The no-plasticity arm must change no active synapses.
6. `run_local_gates.py` now executes **all**
   `research/multimodal/test_*.py` tests via subtree discovery. The prior
   fixed list silently omitted many previously added test modules.

## Local evidence / boundary

Independent local mirror tested pure-Python audit, control comparison and
gate-runner behavior on Python 3.13 / NumPy / pytest. **24 tests passed.**
This validates the logic of those pure components, including fabricated
weight changes outside the target fiber and all three fail-closed negative
cases. It does NOT establish that the new native adapter, actual
`NumpyExplicitEngine`, or all GitHub branch tests work end-to-end.

No new native numerical accuracy, memory capacity, Rust speedup, or FPGA
projection rate is claimed.

## The real next command

In a complete checkout of this precise branch with dependencies installed:

```sh
python -m research.multimodal.run_local_gates \
  --output /tmp/assemblies-native-gates-r1j

python -m research.multimodal.native_teacher_controls \
  --family cyclic_shift --seed 42 \
  --n 512 --k 16 --contents 12 \
  --output /tmp/assemblies-r1j-native-seed42.json
```

Preserve both JSON files, stdout and stderr, Git HEAD, environment,
failed seeds, and source hashes. Review specifically:

- Correct-teacher: teacher target overlap, active-edge delta, unprompted
  training recall, held-out exact recall.
- Shuffled-teacher: matched seed, same write count, incorrect teacher
  targets, and correctly scored original labels.
- `beta=0`: clamped output may still match the teacher, but active
  synaptic delta **must be zero**.
- Every predicted read: equal full-connectome fingerprints before and
  after, without relying on equal outputs alone.

**Research caveat:** the full explicit fingerprint covers native explicit
area-to-area matrices. This R1h configuration does not add stimuli. If
stimulus pathways, sparse mixed engines, or other mutable global state
are introduced, the fingerprint contract needs expansion. It does not
prove read-only state immutability outside the checked matrices.

After these run cleanly, establish a fixed preregistered seed matrix and
compare actual held-out accuracies against train-only ShiftRoleLearner,
sparse Hebbian pair memory, and exact-key episodic baselines. Do not
promote a one-shot association result to zero-shot composition.
