# Volume 13: Programming Assemblies

This volume explores the programming-language view of assemblies.

The idea is not that the repo already contains a mature neural compiler. The
idea is that operations, fibers, states, transitions, and traces can be treated
as a small executable language whose behavior is inspectable.

## Notebooks

- `01_transition_objects_and_programs.ipynb`: inspect typed transitions and the
  program-like pieces that connect FSMs, PFAs, fiber schedules, and assembly
  traces.

## Future Notebooks

- operation graphs as programs
- type-like area and role constraints
- compiler phases from symbolic task to assembly protocol
- optimization of operation schedules
- differentiable or trainable assembly programs
- failure traces as runtime errors

This volume should be precise about metaphor: compiler language is a framework
for design and analysis, not a proven cognitive compiler yet.
