"""Legacy and performance tests.

This directory exercises the compatibility surfaces: the historical
``import brain`` / ``import simulations`` names live in ``legacy/root_shims``
(the repository root holds no Python module), so that directory joins the
path here, after the repository root.
"""
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_SHIMS = REPO_ROOT / "legacy" / "root_shims"

for entry in (str(ROOT_SHIMS), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
