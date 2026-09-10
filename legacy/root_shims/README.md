# Root shims

These files used to sit at the repository root so that a historical
checkout could ``import parser`` or ``import simulations`` and reach the
archived implementations under ``legacy/root_modules/``. They moved here
on 2026-09-09 so the root holds only ``brain.py`` (the package's own entry
shim) and project metadata.

To use the old imports, put this directory on the path:

```bash
PYTHONPATH=legacy/root_shims python -c "import simulations; print(simulations.project_sim)"
```

Each shim is a re-export and nothing else; the implementations are in
``legacy/root_modules/``. ``tests/test_legacy_root_shims.py`` covers them.
