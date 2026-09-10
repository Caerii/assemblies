# Legacy

`legacy/` keeps historical code and artifacts that used to live at the
repository root.

## Layout

- `root_modules/`
  Old root implementations.
- `root_shims/`
  The re-export files that used to sit at the repository root
  (`parser.py`, `simulations.py`, `learner.py`, `brain_util.py`,
  `image_learner.py`, `recursive_parser.py`). Put the directory on
  `PYTHONPATH` to use the old imports; see its README.
- `scripts/`
  Standalone scripts from the pre-package layout.
- `artifacts/image_learning/`
  GIFs and experiment outputs from older CIFAR-10 and animation work.
- `experiments/`
  Older experiment notes that predate the `research/` layout.
- `matlab/`
  MATLAB prototypes retained for historical reference.

## Rule

Put new runtime code in `neural_assemblies/`.
Put new scientific workflows in `research/`.
Keep the repository root to `brain.py` (the package's entry shim) and
project metadata.
