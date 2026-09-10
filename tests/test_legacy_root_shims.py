import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIMS = os.path.join(ROOT, "legacy", "root_shims")


def test_brain_root_shim_points_to_package_brain():
    import brain

    assert brain.Brain.__module__ == "neural_assemblies.core.brain"


def _with_shims_on_path():
    if SHIMS not in sys.path:
        sys.path.insert(0, SHIMS)


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


def test_root_holds_only_the_package_entry_shim():
    """The six archived-module shims live in legacy/root_shims/, not at the root."""
    for name in ("brain_util", "image_learner", "learner", "parser",
                 "recursive_parser", "simulations"):
        assert not os.path.exists(os.path.join(ROOT, name + ".py")), name
        assert os.path.exists(os.path.join(SHIMS, name + ".py")), name


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
