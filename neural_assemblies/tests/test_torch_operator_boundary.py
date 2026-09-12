from pathlib import Path
import ast

from neural_assemblies.core.torch_engine._torch_ops import TorchOps, torch_ops


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
