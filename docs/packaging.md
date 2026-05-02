# Packaging and Release Guide

Use this guide to build, test, and publish the PyPI package
`neural-assemblies`.

The import name remains `neural_assemblies`.

All commands below assume the repository root unless noted otherwise.

## Publishing Reality

- Current PyPI project name: `neural-assemblies`
- Current import name: `neural_assemblies`
- Current package metadata source: `pyproject.toml`
- Current version mirror in code: `neural_assemblies/__init__.py`

If the project is later renamed on PyPI to `assemblies`, this document should
be updated at that time. Until then, do not change these instructions to imply
that `assemblies` is already the published primary package.

## Trusted Publisher Setup

The repo includes [.github/workflows/publish.yml](../.github/workflows/publish.yml)
for GitHub-to-PyPI publishing via OIDC.

One-time PyPI setup:

1. Open PyPI trusted publisher management.
2. Add a pending publisher with:
   - project name: `neural-assemblies`
   - owner: your GitHub user or org
   - repository: `assemblies`
   - workflow: `publish.yml`
3. If you also decide to require a GitHub environment such as `pypi`, make the
   workflow and the PyPI trusted-publisher entry match exactly.

The workflow leaves the environment optional. If you uncomment an
`environment:` line in the workflow later, update the PyPI trusted-publisher
entry to match.

## Local Development

```bash
uv sync
uv run pytest neural_assemblies/tests -q
uv run ruff check neural_assemblies/
```

Optional GPU dependencies:

```bash
uv sync --group gpu
```

## Build

```bash
uv run python -m build
```

Expected artifacts:

- `dist/neural_assemblies-<version>-py3-none-any.whl`
- `dist/neural_assemblies-<version>.tar.gz`

Quick inspection:

```bash
python -m zipfile -l dist/neural_assemblies-*.whl
```

## Publish

### Option A: GitHub Trusted Publisher

- create a GitHub release
- or run the workflow manually from the Actions tab

The publish workflow runs tests, builds the package, downloads the built
artifacts, and publishes them to PyPI through the trusted-publisher action.

### Option B: `twine`

```bash
twine upload dist/neural_assemblies-*
```

For TestPyPI:

```bash
twine upload --repository testpypi dist/neural_assemblies-*
```

### Option C: `uv publish`

```bash
uv publish
```

For TestPyPI:

```bash
uv publish --publish-url https://test.pypi.org/legacy/
```

## Release Checklist

Before publishing:

- `version` in `pyproject.toml` is correct
- `__version__` in `neural_assemblies/__init__.py` matches
- `uv run pytest neural_assemblies/tests -q` passes
- `uv run ruff check neural_assemblies/` passes
- `uv run python -m build` succeeds
- `dist/` contains only the intended `neural_assemblies-*` artifacts

Optional extra wheel check:

```bash
# Fresh-environment install check
# python -m venv .venv-check
# .venv-check\Scripts\pip install dist/neural_assemblies-*.whl
# .venv-check\Scripts\python -c "import neural_assemblies; print(neural_assemblies.__version__)"
```

## Files That Matter For Packaging

| Path | Role |
|------|------|
| `pyproject.toml` | Package metadata, dependencies, version, pytest config. |
| `neural_assemblies/` | Importable package contents. |
| `MANIFEST.in` | Additional source-distribution inclusion rules. |
| `README.md` | Long description used by the package metadata. |
| `LICENSE` | License shipped in the source distribution. |

## Practical Rule

Treat packaging docs as release-contract docs, not aspirational rename docs.
They should describe what maintainers can publish today.
