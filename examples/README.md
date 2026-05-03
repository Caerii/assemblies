# Examples

Small examples live here. They use the package imports directly.

## Script

`01_basic_assembly_calculus.py` covers `Brain`, `project`, and `merge`.

Run it from the repo root:

```bash
uv run python examples/01_basic_assembly_calculus.py
```

After installing the package, plain Python works too:

```bash
python examples/01_basic_assembly_calculus.py
```

## Notebook

See `notebooks/README.md` for the notebook curriculum, standards, and reading
order.

`notebooks/volume-01-foundations/01_basic_assembly_calculus.ipynb` contains the
same basic example in notebook form.

The notebook tree is organized as a short course:

- `volume-01-foundations`: projection, merge, lexicons, and readout
- `volume-02-memory-and-computation`: sequence memory, LRI, FSMs, and PFAs
- `volume-03-language`: controlled NEMO-style parsing
- `volume-04-research-workflow`: claims, experiments, and status indexes
- `volume-05-dynamics-and-statistical-mechanics`: order parameters and regimes
- `volume-06-binding-composition-and-interference`: composition diagnostics
- `volume-07-control-circuits-and-runtime`: gating and runtime protocols
- `volume-08-prediction-and-erp-signals`: prediction demos and ERP boundaries
- `volume-09-embodiment-and-grounding`: grounded-task interface maps
- `volume-10-biological-validation-and-connectomes`: biological validation maps
- `volume-11-scaling-engines-and-performance`: backends and winner policies
- `volume-12-systems-paper-and-theory`: paper-claim planning
- `volume-13-programming-assemblies`: transitions and operation programs
- `volume-14-frontiers-and-open-problems`: frontier research triage

From the repo root:

```bash
uv sync --group notebooks
uv run jupyter lab
```

The notebook dependency group adds JupyterLab, widgets, interactive Matplotlib,
canvas drawing, graph widgets, Plotly, and NetworkX. The notebooks should still
avoid making those tools required for normal package imports.

The repo pins the default development interpreter in `.python-version`.
