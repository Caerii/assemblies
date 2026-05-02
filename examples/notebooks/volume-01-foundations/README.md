# Volume 1: Foundations

Start here if you have not used the package before.

This volume is about learning to see the substrate. Before language,
automata, or research claims, there are only areas, stimuli, winners, and
overlap.

The basic runtime objects are:

- `Brain`: owns areas, stimuli, and the compute engine.
- `Area`: a sparse population with `k` active winners.
- `Stimulus`: an external input that can form an assembly.
- `Assembly`: an immutable snapshot of active winners.
- `AssemblyTrace`: a round-by-round record of how a projection or merge
  settles.
- `overlap`: the main diagnostic for comparing assemblies.

After this volume, you should be able to point at a printed number and say
what it means: assembly size, overlap with another assembly, or readout score.

## Notebooks

- `01_basic_assembly_calculus.ipynb`: projection traces, assembly size,
  merge traces, animations, winner turnover, and source-response diagnostics.
- `02_lexicon_and_readout.ipynb`: build a tiny lexicon and decode a probe
  assembly by overlap, including threshold and damage checks.

Try changing `K`, `N`, or the number of projection rounds. The point is not to
find magic parameters; it is to notice how sparse traces become inspectable
objects.

After this volume, move to `../volume-02-memory-and-computation`.
