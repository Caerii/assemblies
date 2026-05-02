# Volume 2: Memory And Computation

This volume moves from individual assemblies to dynamics over time. The
question changes from "what fired?" to "where does activity go next?"

It covers:

- memorizing short stimulus sequences
- inspecting recall with Long-Range Inhibition (LRI)
- running deterministic finite-state utilities
- sampling probabilistic automata

## Notebooks

- `01_sequence_memory_lri.ipynb`: sequence memorization, overlap diagnostics,
  animated LRI recall, and winner-turnover inspection under one seeded
  parameter setting.
- `02_fsm_and_pfa.ipynb`: parity with `FSMNetwork` and independent samples
  from a small `PFANetwork`.

These notebooks demonstrate package utilities. The full theoretical results
belong to the cited assembly-calculus literature.

Treat the outputs as traces from a small machine: trajectories, overlap
matrices, and sample counts. If a sequence recall run is short or noisy, that
is part of the lesson. The mechanism is visible enough to debug rather than
hidden behind a single success/failure label.
