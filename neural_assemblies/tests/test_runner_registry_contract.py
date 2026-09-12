"""The public experiment registry cannot reintroduce bespoke run entry points."""

import ast
import importlib.util

from research.runner import EXPERIMENTS


def _module_source(name: str) -> str:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, f"unresolvable experiment: {name}"
    with open(spec.origin, encoding="utf-8") as source:
        return source.read()


def _uses_runner_writer(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        names = []
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        if "run_experiment" in names:
            return True
        if any(keyword.arg == "writer" and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "run_experiment" for keyword in node.keywords):
            return True
    return False


def test_registered_experiments_use_the_shared_runner_contract():
    """Registry entries must expose immutable provenance through one path.

    Historical adapters may delegate their parser through ``_historical``;
    current studies import ``experiment_parser`` directly.  Requiring both
    the parser delegation and the writer prevents a new command from looking
    registered while silently writing an untracked result file.
    """
    for command, module in EXPERIMENTS.items():
        source = _module_source(module)
        assert _uses_runner_writer(source), f"{command} bypasses run_experiment"
        assert "experiment_parser" in source or "_historical" in source, (
            f"{command} bypasses the shared experiment parser"
        )


def test_runner_list_is_discoverable_without_importing_experiments(capsys):
    from research import runner

    assert runner.main(["--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(runner.EXPERIMENTS)
    assert all("\t" in line for line in lines)
    assert lines == sorted(lines)
