import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIMS = os.path.join(ROOT, "legacy", "root_shims")


def _with_shims_on_path():
    if SHIMS not in sys.path:
        sys.path.insert(0, SHIMS)


def test_brain_shim_points_to_package_brain():
    _with_shims_on_path()
    import brain

    assert brain.Brain.__module__ == "neural_assemblies.core.brain"


def test_legacy_root_shims_expose_archived_modules():
    _with_shims_on_path()
    import brain_util
    import learner
    import parser
    import recursive_parser
    import simulations

    assert brain_util.overlap.__module__ == "legacy.root_modules.brain_util"
    assert simulations.project_sim.__module__ == "legacy.root_modules.simulations"
    assert learner.LearnBrain.__module__ == "legacy.root_modules.learner"
    assert parser.parse.__module__ == "legacy.root_modules.parser"
    assert recursive_parser.parse.__module__ == "legacy.root_modules.recursive_parser"


def test_root_holds_no_python_module():
    """Every historical shim, brain.py included, lives in legacy/root_shims/."""
    for name in ("brain", "brain_util", "image_learner", "learner", "parser",
                 "recursive_parser", "simulations"):
        assert not os.path.exists(os.path.join(ROOT, name + ".py")), name
        assert os.path.exists(os.path.join(SHIMS, name + ".py")), name
    assert not [f for f in os.listdir(ROOT) if f.endswith(".py")]


def test_image_learner_shim_is_optional_on_gpu_stack():
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("torchvision") is None
        or importlib.util.find_spec("sklearn") is None
    ):
        return
    _with_shims_on_path()
    import image_learner

    assert image_learner.CIFAR10Brain.__module__ == "legacy.root_modules.image_learner"
