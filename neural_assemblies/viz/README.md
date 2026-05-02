# Visualization

`neural_assemblies.viz` contains small Matplotlib-based helpers for notebooks,
teaching, and diagnostics.

The functions are intentionally lightweight:

- `plot_projection_flow`: draw the shape of a two-source projection/merge
  example.
- `plot_binding_story`: draw a more narrative two-source binding diagram for
  first-contact notebooks.
- `plot_assembly`: show one assembly as active winners on a square neuron grid.
- `plot_assemblies`: compare several assembly grids side by side.
- `animate_assembly_trace`: animate winner turnover across an
  `AssemblyTrace`.
- `plot_trace_metrics`: plot consecutive overlap and new-winner counts from a
  trace.
- `plot_merge_diagnostic`: compare target activity before/after merge and a
  same-area replay-overlap check.
- `plot_response_overlap`: compare same-area response traces against a
  reference assembly and chance overlap.
- `plot_overlap_matrix`: show pairwise assembly overlap.
- `plot_recall_trace`: compare recalled assemblies against known references.

These helpers are not scientific evidence by themselves. They make package
state easier to inspect. Use `research/` artifacts for claims about measured
behavior.
