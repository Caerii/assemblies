from pathlib import Path
import ast
import subprocess
import sys

from neural_assemblies.core._torch_ops import (
    TorchOps, _LazyTorchOps, torch_ops,
)


ROOT = Path(__file__).parents[1] / "core" / "torch_engine"
_ALLOWED_DIRECT = {"Tensor", "cuda"}
_EXCLUDED = {"_fused_cuda.py", "_torch_ops.py"}


def test_generated_torch_operators_use_typed_runtime_boundary():
    """Keep generated Torch calls behind ``torch_ops``.

    ``torch.Tensor`` is a type anchor and ``torch.cuda`` is an explicit
    lifecycle boundary.  Generated factories, dtypes, and operators belong to
    the shared protocol so static checks cannot silently lose their surface.
    """
    violations = []
    for path in sorted(ROOT.glob("*.py")):
        if path.name in _EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "torch":
                continue
            if node.attr not in _ALLOWED_DIRECT:
                violations.append(f"{path.name}:{node.lineno}: torch.{node.attr}")
    assert not violations, "generated Torch calls bypass torch_ops:\n" + "\n".join(violations)


def test_torch_runtime_implements_declared_operator_protocol():
    """Fail at import-time validation when a Torch wheel lacks a declared member."""
    missing = [name for name in TorchOps.__annotations__ if not hasattr(torch_ops, name)]
    assert not missing, "TorchOps declares unavailable runtime members: " + ", ".join(missing)


def test_torch_operator_proxy_loads_once_on_first_use(monkeypatch):
    sentinel = object()
    module = type("FakeTorch", (), {"example": sentinel})()
    calls = []

    def fake_import(name):
        calls.append(name)
        return module

    import neural_assemblies.core._torch_ops as boundary
    monkeypatch.setattr(boundary, "import_module", fake_import)
    proxy = _LazyTorchOps()
    assert proxy.example is sentinel
    assert proxy.example is sentinel
    assert calls == ["torch"]


def test_calculus_torch_boundary_imports_without_torch():
    """CPU-only calculus imports must not execute the CUDA engine package."""
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import neural_assemblies.assembly_calculus.batched_trainer"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
