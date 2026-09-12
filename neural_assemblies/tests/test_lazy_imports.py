"""`import neural_assemblies` must stay cheap, and must not change behaviour.

The package used to re-export eagerly, so importing ANYTHING -- even
`neural_assemblies.core.brain` -- pulled in all eight subpackages: the emergent
parser, the language organ, the reference implementations. 255 ms on top of
numpy, paid by every process, and a sweep spawning a process per trial paid it
per trial.

PEP 562 `__getattr__` defers that. The risk it introduces is not slowness but
SILENCE: a name that quietly stops resolving, a subpackage attribute that
starts raising AttributeError, or a TYPE_CHECKING block that drifts out of step
with the runtime map and takes IDE autocomplete down with it. Nothing here
fails loudly on its own, so it is all asserted.
"""

import subprocess
import sys

import pytest

import neural_assemblies


def _in_fresh_process(code):
    """Run *code* in a new interpreter and return its stdout.

    Import cost and module-set questions are only meaningful before anything
    else has imported the package, so they cannot be asked in-process.
    """
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


def test_every_public_name_resolves():
    """`__all__` is the contract; each entry must actually be gettable."""
    missing = []
    for name in neural_assemblies.__all__:
        try:
            getattr(neural_assemblies, name)
        except AttributeError:
            missing.append(name)
    assert not missing, f"declared in __all__ but unresolvable: {missing}"


def test_star_import_still_works():
    """`from neural_assemblies import *` consults __all__, then __getattr__."""
    ns = {}
    exec("from neural_assemblies import *", ns)
    for name in neural_assemblies.__all__:
        assert name in ns, f"{name} did not survive a star import"


def test_repeated_access_returns_the_same_object():
    """Resolution caches into globals; a second lookup must not re-import.

    If it returned a fresh object each time, `isinstance` checks and identity
    comparisons across modules would silently start failing.
    """
    assert neural_assemblies.Brain is neural_assemblies.Brain
    assert neural_assemblies.merge is neural_assemblies.merge


def test_subpackages_are_reachable_as_attributes():
    """`import neural_assemblies` then `neural_assemblies.core` must work.

    It used to work as a side effect of the eager imports. Losing it is the
    kind of break that surfaces in someone else's script, not in this suite.
    """
    for sub in ("core", "compute", "utils", "assembly_calculus", "constants"):
        mod = getattr(neural_assemblies, sub)
        assert mod.__name__ == f"neural_assemblies.{sub}"


def test_unknown_attribute_still_raises_attributeerror():
    """A typo must not be swallowed by the lazy path."""
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(neural_assemblies, "definitely_not_a_real_export")


def test_dir_lists_the_public_api():
    """`dir()` drives tab-completion; __getattr__ does not populate it."""
    listed = set(dir(neural_assemblies))
    assert set(neural_assemblies.__all__) <= listed
    assert "core" in listed


def test_type_checking_block_matches_the_runtime_map():
    """The TYPE_CHECKING imports exist only to keep type checkers working.

    They are a hand-maintained duplicate of `_LAZY_EXPORTS`, so they will drift.
    When they do, the failure is invisible at runtime and shows up as IDE
    autocomplete quietly dying, which nobody files a bug for.
    """
    import ast
    from pathlib import Path

    src = Path(neural_assemblies.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom):
                    declared.update(a.name for a in sub.names)

    runtime = set(neural_assemblies._LAZY_EXPORTS)
    assert declared == runtime, (
        "the TYPE_CHECKING block and _LAZY_EXPORTS disagree.\n"
        f"  only in TYPE_CHECKING: {sorted(declared - runtime)}\n"
        f"  only in _LAZY_EXPORTS: {sorted(runtime - declared)}")


def test_lazy_exports_covers_all():
    """Every `__all__` entry is either lazy or a real module-level value."""
    eager = {"CUPY_INSTALLED", "GPU_AVAILABLE"}
    unaccounted = (set(neural_assemblies.__all__)
                   - set(neural_assemblies._LAZY_EXPORTS) - eager)
    assert not unaccounted, (
        f"in __all__ but neither lazy nor eagerly defined: {sorted(unaccounted)}")


def test_importing_the_package_pulls_in_no_subpackages():
    """The whole point. Guards against a stray eager import creeping back."""
    out = _in_fresh_process(
        "import sys, neural_assemblies;"
        "print(sorted({m.split('.')[1] for m in sys.modules"
        " if m.startswith('neural_assemblies.')}))"
    )
    assert out == "[]", f"these subpackages loaded eagerly: {out}"


def test_importing_a_submodule_does_not_pull_in_the_rest():
    """`import neural_assemblies.core.brain` must not drag in the parser.

    This is the case that actually bit: the eager __init__ ran first, so even
    the narrowest import bought the entire research stack.
    """
    out = _in_fresh_process(
        "import sys, neural_assemblies.core.brain;"
        "print(sorted({m.split('.')[1] for m in sys.modules"
        " if m.startswith('neural_assemblies.')}))"
    )
    assert "assembly_calculus" not in out, f"language stack came along: {out}"
    assert "language" not in out and "programs" not in out, out


def test_cupy_installation_flag_is_set_without_importing_cupy():
    """Load-bearing, and easy to lose in a rewrite of this file.

    `CUPY_INSTALLED` is computed with `find_spec`, NOT `import cupy`, because on
    Windows whichever of CuPy/torch loads first wins DLL resolution for the
    process -- and importing CuPy at package-import time made later torch CUDA
    calls die with a bare access violation. See the comment in __init__.py.
    """
    out = _in_fresh_process(
        "import sys, neural_assemblies as na;"
        "print(isinstance(na.CUPY_INSTALLED, bool),"
        " na.GPU_AVAILABLE is na.CUPY_INSTALLED, 'cupy' in sys.modules)"
    )
    ok, compatibility_alias, cupy_loaded = out.split()
    assert ok == "True", "CUPY_INSTALLED is not a bool"
    assert compatibility_alias == "True", "legacy GPU_AVAILABLE alias drifted"
    assert cupy_loaded == "False", "package import loaded CuPy -- this breaks torch"


def test_nemo_namespace_does_not_eagerly_require_cupy():
    out = _in_fresh_process(
        "import sys, neural_assemblies.nemo;"
        "print('cupy' in sys.modules)"
    )
    assert out == "False"
