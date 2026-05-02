# Visualization

`neural_assemblies.viz` contains small Matplotlib-based helpers for notebooks,
teaching, and diagnostics.

The functions are intentionally lightweight:

- `plot_projection_flow`: draw the shape of a two-source projection/merge
  example.
- `plot_assembly`: show one assembly as active winners on a square neuron grid.
- `plot_assemblies`: compare several assembly grids side by side.
- `plot_overlap_matrix`: show pairwise assembly overlap.
- `plot_recall_trace`: compare recalled assemblies against known references.

These helpers are not scientific evidence by themselves. They make package
state easier to inspect. Use `research/` artifacts for claims about measured
behavior.
