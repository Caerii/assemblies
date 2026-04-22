# Legacy Layout

This directory archives historical code and artifacts that used to live at the
repository root.

## Structure

- `root_modules/`
  Original repo-root experiment modules. Root-level files like `parser.py` and
  `simulations.py` now act as compatibility shims and import from here.
- `scripts/`
  Archived top-level scripts that predate the current packaged simulation and
  tooling layout.
- `artifacts/image_learning/`
  Tracked GIFs and experiment outputs from the older CIFAR-10 / animation work.
- `experiments/`
  Archived top-level experiment notes that predate the current `research/`
  layout.
- `matlab/`
  MATLAB prototypes retained for historical reference.

## Rule

New reusable code should land under `neural_assemblies/`.
New scientific workflows should land under `research/`.
Nothing new should be added to the repository root unless it is an intentional
top-level compatibility shim or project metadata.
