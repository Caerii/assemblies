# Packaging and release guide

This document is for maintainers who build, test, and publish the **neural-assemblies** package to PyPI. All paths and commands below are from the **repository root** unless noted.

## One-time setup

**Recommended: Trusted Publishers (no API token)**  
Use PyPI’s Trusted Publishers so GitHub Actions can publish via OIDC—no token to store or rotate.

1. In PyPI go to **Account → Publishing → Trusted Publisher Management**.
2. Under **Add a new pending publisher**:
   - **PyPI Project Name:** `neural-assemblies`
   - **Owner:** your GitHub username or org (e.g. `Caerii`)
   - **Repository name:** `assemblies`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi` (optional; create in repo **Settings → Environments** for extra protection)
3. Click **Add**. The workflow [.github/workflows/publish.yml](../.github/workflows/publish.yml) runs on **Release published** or **Run workflow** (Actions tab).

**Alternative: API token**  
Create an API token at [PyPI → API tokens](https://pypi.org/manage/account/token/) with scope for `neural-assemblies`. Use it as `UV_PUBLISH_TOKEN` or `TWINE_PASSWORD` (with `TWINE_USERNAME=__token__`) for local/CI uploads.

## Version and metadata

- **Single source of version**: Update `version` in **`pyproject.toml`** (repo root) for each release. Keep **`neural_assemblies/__init__.py`** `__version__` in sync (same string) so `import neural_assemblies; print(neural_assemblies.__version__)` matches the package.
- **URLs** in `pyproject.toml` point to the canonical repo (e.g. `https://github.com/Caerii/assemblies`). Change if the repo moves.

## Local development

```bash
# From repo root
uv sync                    # install deps + dev deps (dependency-groups.dev)
uv run python -c "from neural_assemblies.core.brain import Brain; print('OK')"
uv run pytest neural_assemblies/tests/ -q   # run tests
uv run ruff check neural_assemblies/       # lint

# GPU extras (on supported CUDA/ROCm machines)
# uv sync --group gpu        # adds dependency-groups.gpu on top
# or: pip install -e ".[gpu]"
```

## Build (no upload)

```bash
pip install build   # or: uv add --dev build
python -m build
```

This produces:

- `dist/neural_assemblies-<version>-py3-none-any.whl`
- `dist/neural_assemblies-<version>.tar.gz`

Check that the wheel contains the package:

```bash
python -m zipfile -l dist/neural_assemblies-*.whl
```

## Upload to PyPI

### Option A: Trusted Publisher (recommended; no token)

After adding the pending publisher on PyPI (see **One-time setup** above):

- **From GitHub:** Create a **Release** (tag e.g. `v0.0.1a1`) or run the workflow manually (**Actions → Publish to PyPI → Run workflow**). The workflow runs tests, builds, and publishes.
- No secrets to configure in the repo.

### Option B: twine (API token)

```bash
# Upload only neural_assemblies artifacts (not legacy assemblies-*)
twine upload dist/neural_assemblies-*
# Or Test PyPI first:
twine upload --repository testpypi dist/neural_assemblies-*
```

Set credentials: `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<pypi-api-token>` (or use keyring). Create the token at [PyPI → API tokens](https://pypi.org/manage/account/token/) with scope for project `neural-assemblies`.

### Option C: uv (API token)

```bash
uv publish
# Or Test PyPI:
uv publish --publish-url https://test.pypi.org/legacy/
```

Set `UV_PUBLISH_TOKEN` to your PyPI API token. Uploads whatever is in `dist/`; ensure only `neural_assemblies-*` files are present.

## Systematic test (before every release)

Run from repo root to verify everything works:

```bash
# 1) Environment
uv sync

# 2) Imports
uv run python -c "import neural_assemblies; from neural_assemblies.core.brain import Brain; from neural_assemblies.assembly_calculus import project, merge; print('OK', neural_assemblies.__version__)"

# 3) Unit tests (package tests only; full suite includes research/experiments)
uv run pytest neural_assemblies/tests/ -v --tb=short

# 4) Build
uv run python -m build

# 5) Optional: install from built wheel in a fresh venv (validates wheel deps)
# python -m venv .venv-check && .venv-check\Scripts\pip install dist/neural_assemblies-*.whl && .venv-check\Scripts\python -c "import neural_assemblies; print(neural_assemblies.__version__)"
```

Only **neural_assemblies** artifacts should be in `dist/` before uploading (remove any old `assemblies-*` if present).

## Pre-commit checklist

Before committing packaging changes:

- [ ] `version` in `pyproject.toml` and `__version__` in `neural_assemblies/__init__.py` match (at repo root).
- [ ] `[project.urls]` in `pyproject.toml` point to the correct repository.
- [ ] `uv sync` and `uv run pytest neural_assemblies/tests/ -q` pass.
- [ ] `python -m build` succeeds and `dist/` contains the expected wheel and sdist.

Before publishing a release:

- [ ] CHANGELOG or release notes updated (if you keep them).
- [ ] Tag the release in git, e.g. `git tag v0.2.0`.
- [ ] Run `python -m build` and `twine upload dist/*` (or `uv publish`).

## Files that affect the package

| File / directory (from repo root) | Role |
|------------------------------------|------|
| `pyproject.toml`                   | Build config, metadata, dependencies, version. |
| `neural_assemblies/`               | The importable package (included via setuptools `packages.find`). |
| `MANIFEST.in`                      | Extra files included in the source distribution (e.g. LICENSE, README). |
| `README.md`                        | Long description on PyPI (`readme = "README.md"`). |
| `LICENSE`                          | Shipped in sdist via MANIFEST.in. |
| `neural_assemblies/py.typed`       | PEP 561 marker for type checkers. |
