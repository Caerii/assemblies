import importlib


def test_brain_root_shim_points_to_package_brain():
    import brain

    assert brain.Brain.__module__ == "neural_assemblies.core.brain"


def test_legacy_root_shims_expose_archived_modules():
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


def test_image_learner_shim_is_optional_on_gpu_stack():
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("torchvision") is None
        or importlib.util.find_spec("sklearn") is None
    ):
        return

    import image_learner

    assert image_learner.CIFAR10Brain.__module__ == "legacy.root_modules.image_learner"
