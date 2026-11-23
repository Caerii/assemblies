# R1h — Teacher-guided native role-projection protocol

**Status:** code committed, native execution pending. 2026-11-23.

The prior R1g adapter intentionally ignored target supervision. R1h
adds the missing train-only teaching signal using a public `Brain.project`
feature: `external_drive={role: dense_vector}`. The repository explicitly
restricts external drive to the dense explicit engine, so both lexical and
role areas use `explicit=True`. This is a different substrate from R1g;
comparison must note that distinction.

Each observed (item,role) pair is trained with a fixed k-neuron cue and
teacher drive on the correct k target neurons. `teacher_gain=100` and
`presentations=3` are exploratory defaults, not tuned or validated.
No teacher drive is supplied to inference. The benchmark itself withholds
every checkerboard held-out pair from `train`, and the adapter receives
neither the hidden permutation nor the shift parameter.

Run from a full checkout:

```sh
uv run python -m pytest -q research/multimodal/test_native_teacher_drive.py
uv run python -m research.multimodal.native_teacher_drive \
  --family cyclic_shift --seed 42 --n 512 --k 16 --contents 12
```

**What this can establish:** whether teacher-guided local potentiation
recovers training pairs and extrapolates to held-out role-conditioned
target assemblies. A failure to generalize is a meaningful outcome
because the cyclic-shift family has a train-only learnable rule.

**What this cannot establish:** that native Brain has learned an explicit
cyclic transformation or that a transformer-scale multimodal system
would benefit. Strong teacher drive can clamp winners without producing
effective source->target synapses; the experiment MUST check native
weight changes, held-out recall and no-read mutation before interpreting
any result.

Next gates:
1. Run native tests and report failures with exact commit/seed.
2. Compare zero-plasticity and shuffled-target controls.
3. Vary teacher gain and presentations; record train/held-out exact,
   strict rank1, margin, memory footprint and latency.
4. Compare against `ShiftRoleLearner` (hypothesis-aware strong baseline)
   and `HebbianRoleMap` (plain pair learner).
5. Only after a genuine native effect, design compositional role
   transforms with nontrivial relational semantics and multimodal cues.
