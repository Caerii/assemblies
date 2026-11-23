"""Fail-closed preflight tests: mocked Git/native origin, no fake native results."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.multimodal import native_execution_preflight as P


def test_missing_package_is_not_success(tmp_path, monkeypatch):
    root = tmp_path
    source = root / 'neural_assemblies/core/brain.py'
    source.parent.mkdir(parents=True)
    source.write_text('class Brain: pass\n')
    def git(_, *args):
        return {('rev-parse', '--show-toplevel'): str(root),
                ('branch', '--show-current'): P.EXPECTED_BRANCH,
                ('status', '--porcelain', '--', 'research/multimodal', 'neural_assemblies'): '',
                ('rev-parse', 'HEAD'): 'deadbeef'}[args]
    monkeypatch.setattr(P, '_git', git)
    with pytest.raises((ModuleNotFoundError, ImportError)):
        P.inspect_checkout(root)


def test_wrong_branch_fails_before_import(tmp_path, monkeypatch):
    def git(_, *args):
        return str(tmp_path) if args == ('rev-parse', '--show-toplevel') else 'master'
    monkeypatch.setattr(P, '_git', git)
    with pytest.raises(P.NativePreflightError, match='wrong branch'):
        P.inspect_checkout(tmp_path)


def test_dirty_worktree_rejected(tmp_path, monkeypatch):
    def git(_, *args):
        if args == ('rev-parse', '--show-toplevel'):
            return str(tmp_path)
        if args == ('branch', '--show-current'):
            return P.EXPECTED_BRANCH
        return ' M neural_assemblies/core/brain.py'
    monkeypatch.setattr(P, '_git', git)
    with pytest.raises(P.NativePreflightError, match='uncommitted'):
        P.inspect_checkout(tmp_path)


def test_valid_provenance_and_stale_package_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'neural_assemblies/core/brain.py'
    source.parent.mkdir(parents=True)
    source.write_text('class Brain: pass\n')
    def git(_, *args):
        if args == ('rev-parse', '--show-toplevel'):
            return str(tmp_path)
        if args == ('branch', '--show-current'):
            return P.EXPECTED_BRANCH
        if args == ('rev-parse', 'HEAD'):
            return '1234'
        return ''
    monkeypatch.setattr(P, '_git', git)
    class Brain:
        def project(self, external_drive=None):
            pass
        def engine_for(self): pass
        def read_only(self): pass
        def frozen(self): pass
        def add_area(self): pass
        def clear_activity(self): pass
    monkeypatch.setattr(P.importlib, 'import_module', lambda _: SimpleNamespace(Brain=Brain))
    monkeypatch.setattr(P.inspect, 'getfile', lambda _: str(tmp_path / 'site-packages/brain.py'))
    with pytest.raises(P.NativePreflightError, match='stale Brain'):
        P.inspect_checkout(tmp_path)
    monkeypatch.setattr(P.inspect, 'getfile', lambda _: str(source))
    # package import inside helper is a separate real import; check that it
    # fails rather than silently manufacturing a passing native result.
    with pytest.raises((ModuleNotFoundError, ImportError)):
        P.inspect_checkout(tmp_path)


def test_numeric_smoke_rejects_bad_geometry_without_import():
    with pytest.raises(ValueError):
        P.numeric_smoke(n=8, k=9)
    with pytest.raises(ValueError):
        P.numeric_smoke(seed=-1)


def test_cli_missing_native_exits_nonzero(monkeypatch):
    monkeypatch.setattr(P, 'inspect_checkout', lambda *a, **k: (_ for _ in ()).throw(
        P.NativePreflightError('missing exact checkout')))
    monkeypatch.setattr('sys.argv', ['native_execution_preflight'])
    assert P.main() == 2
