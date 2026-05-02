# Volume 1: Foundations

Start here if you have not used the package before.

This volume teaches the basic runtime objects:

- `Brain`: owns areas, stimuli, and the compute engine.
- `Area`: a sparse population with `k` active winners.
- `Stimulus`: an external input that can form an assembly.
- `Assembly`: an immutable snapshot of active winners.
- `overlap`: the main diagnostic for comparing assemblies.

## Notebooks

- `01_basic_assembly_calculus.ipynb`: projection, merge, and overlap.
- `02_lexicon_and_readout.ipynb`: build a tiny lexicon and decode a probe
  assembly by overlap.

After this volume, move to `../volume-02-memory-and-computation`.
