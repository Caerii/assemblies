# Legacy

`legacy/` keeps historical code and artifacts that used to live at the
repository root.

## Layout

- `root_modules/`
  Old root implementations. Root files such as `parser.py` and
  `simulations.py` now import from here or from maintained package code.
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
Put only compatibility shims and project metadata at the repository root.
