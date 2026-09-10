# Root shims

These files used to sit at the repository root so that a historical
checkout could ``import brain``, ``import parser`` or ``import simulations``
and reach the package or the archived implementations under
``legacy/root_modules/``. They moved here on 2026-09-09; the repository
root now holds no Python module, only project metadata.

To use the old imports, put this directory on the path:

```bash
PYTHONPATH=legacy/root_shims python -c "import brain, simulations; print(brain.Brain, simulations.project_sim)"
```

| Shim | Routes to |
|------|-----------|
| `brain.py` | `neural_assemblies.core.brain` (the package) |
| `brain_util.py`, `learner.py`, `parser.py`, `recursive_parser.py`, `simulations.py`, `image_learner.py` | `legacy/root_modules/` (the archived implementations) |

Each shim is a re-export and nothing else. ``tests/test_legacy_root_shims.py``
covers them.
