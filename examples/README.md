# Examples

Small examples live here. They use the package imports directly.

## Script

`01_basic_assembly_calculus.py` covers `Brain`, `project`, `merge`, and
`overlap`.

Run it from the repo root:

```bash
uv run python examples/01_basic_assembly_calculus.py
```

After installing the package, plain Python works too:

```bash
python examples/01_basic_assembly_calculus.py
```

## Notebook

`notebooks/01_basic_assembly_calculus.ipynb` contains the same basic example in
notebook form.

From the repo root:

```bash
uv run jupyter notebook
```

Jupyter is not part of the default package test environment. Install it
explicitly if your local environment does not already provide it.

The repo pins the default development interpreter in `.python-version`.
