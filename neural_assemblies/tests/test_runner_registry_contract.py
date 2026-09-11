"""The public experiment registry cannot reintroduce bespoke run entry points."""

import importlib.util

from research.runner import EXPERIMENTS


def _module_source(name: str) -> str:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, f"unresolvable experiment: {name}"
    with open(spec.origin, encoding="utf-8") as source:
        return source.read()


def test_registered_experiments_use_the_shared_runner_contract():
    """Registry entries must expose immutable provenance through one path.

    Historical adapters may delegate their parser through ``_historical``;
    current studies import ``experiment_parser`` directly.  Requiring both
    the parser delegation and the writer prevents a new command from looking
    registered while silently writing an untracked result file.
    """
    for command, module in EXPERIMENTS.items():
        source = _module_source(module)
        assert "run_experiment" in source, f"{command} bypasses run_experiment"
        assert "experiment_parser" in source or "_historical" in source, (
            f"{command} bypasses the shared experiment parser"
        )
