# Examples

This directory contains small runnable examples that exercise the package
surface directly.

## Script Example

- `01_basic_assembly_calculus.py`
  Minimal package-first example covering `Brain`, `project`, `merge`, and
  `overlap`.

Run from the repo root:

```bash
uv run python examples/01_basic_assembly_calculus.py
```

After installing the published package you can also run it with plain Python:

```bash
python examples/01_basic_assembly_calculus.py
```

## Notebook Example

- `notebooks/01_basic_assembly_calculus.ipynb`
  Notebook form of the same basic example.

From the repo root:

```bash
uv run jupyter notebook
```

If Jupyter is not installed in your environment, add it explicitly because it
is not part of the default package test surface.
