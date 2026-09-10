"""
Neural Assembly Simulation Framework

A modular, mathematically rigorous framework for simulating neural assemblies
based on the Assembly Calculus and NEMO model.

Based on:
- Papadimitriou et al. "Brain Computation by Assemblies of Neurons" (2020)
- Mitropolsky et al. "Architecture of a Biologically Plausible Language Organ" (2023)
"""

# RE-EXPORTS ARE LAZY (PEP 562). `from neural_assemblies import Brain` still
# works, and still returns the same object; only the TIMING moves.
#
# These used to be eager `from .x import y` statements, which meant importing
# anything at all -- even `neural_assemblies.core.brain` -- pulled in all eight
# subpackages: the emergent parser, the language organ, the reference
# implementations, the programs. Measured at 255 ms on top of numpy's 214 ms,
# paid by every process. A script that only wants `Brain` was buying the whole
# research stack, and a sweep that spawns a process per trial bought it per
# trial.
#
# Resolution is cached into the module globals on first access, so the second
# lookup is a plain global and costs nothing.
_LAZY_EXPORTS = {
    name: ".core" for name in (
        "Brain", "Area", "Stimulus", "Connectome",
        "ComputeEngine", "ProjectionResult", "create_engine", "list_engines",
    )
}
_LAZY_EXPORTS["HomeostasisConfig"] = ".core._homeostasis"
_LAZY_EXPORTS.update({
    name: ".compute" for name in (
        "StatisticalEngine", "NeuralComputationEngine",
        "TopKPolicy", "ThresholdPolicy", "RelativeThresholdPolicy",
        "EPercentPolicy", "WinnerPolicy", "WinnerSelector", "PlasticityEngine",
    )
})
_LAZY_EXPORTS.update({name: ".constants" for name in ("DEFAULT_P", "DEFAULT_BETA")})
_LAZY_EXPORTS.update({
    name: ".utils" for name in (
        "normalize_features", "select_top_k_indices", "heapq_select_top_k",
        "binomial_ppf",
    )
})
_LAZY_EXPORTS.update({
    name: ".assembly_calculus" for name in (
        "Assembly", "AssemblyTrace", "PatternCompletionDiagnostic",
        "ResponseDiagnostic", "ResponseTrace", "TraceStep",
        "overlap", "chance_overlap",
        "project", "reciprocal_project", "associate", "merge",
        "pattern_complete", "separate",
        "project_trace", "reciprocal_project_trace", "associate_trace",
        "merge_trace", "pattern_complete_trace", "ordered_recall_trace",
        "snapshot_area", "source_response_traces",
        "FiberCircuit",
    )
})

#: Subpackages reachable as attributes. `import neural_assemblies` followed by
#: `neural_assemblies.core` used to work because the eager imports bound them
#: as a side effect; without this it would raise AttributeError, which is the
#: kind of break that only shows up in someone else's script.
_LAZY_SUBMODULES = (
    "core", "compute", "constants", "utils", "assembly_calculus",
    "language", "nemo", "programs", "interfaces",
)


def __getattr__(name):
    """Resolve a re-export on first access, then cache it as a global."""
    from importlib import import_module

    mod = _LAZY_EXPORTS.get(name)
    if mod is not None:
        value = getattr(import_module(mod, __name__), name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        value = import_module("." + name, __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_LAZY_SUBMODULES) | set(globals()))


# Type checkers and IDEs cannot follow `__getattr__`, so without this block
# every re-export reads as undefined and autocomplete stops working. The
# runtime never executes it, so it costs nothing at import time. It must be
# kept in step with `_LAZY_EXPORTS`; `tests/test_lazy_imports.py` asserts that.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .core._homeostasis import HomeostasisConfig
    from .assembly_calculus import (
        Assembly, AssemblyTrace, FiberCircuit, PatternCompletionDiagnostic,
        ResponseDiagnostic, ResponseTrace, TraceStep, associate,
        associate_trace, chance_overlap, merge, merge_trace,
        ordered_recall_trace, overlap, pattern_complete,
        pattern_complete_trace, project, project_trace, reciprocal_project,
        reciprocal_project_trace, separate, snapshot_area,
        source_response_traces,
    )
    from .compute import (
        EPercentPolicy, NeuralComputationEngine, PlasticityEngine,
        RelativeThresholdPolicy, StatisticalEngine, ThresholdPolicy,
        TopKPolicy, WinnerPolicy, WinnerSelector,
    )
    from .constants import DEFAULT_BETA, DEFAULT_P
    from .core import (
        Area, Brain, ComputeEngine, Connectome, ProjectionResult, Stimulus,
        create_engine, list_engines,
    )
    from .utils import (
        binomial_ppf, heapq_select_top_k, normalize_features,
        select_top_k_indices,
    )

# GPU availability. DETECTED WITHOUT IMPORTING CuPy, and that is load-bearing
# rather than a micro-optimisation.
#
# This line used to be `import cupy`, purely to set the boolean below. Importing
# CuPy loads its copy of the CUDA runtime and cuBLAS, and on Windows whichever
# of CuPy / PyTorch loads first wins DLL resolution for the rest of the process.
# Because this runs at PACKAGE import time, CuPy always won -- so any later
# `import torch` bound against CuPy's libraries and crashed on larger cuBLAS
# calls with a bare `Windows fatal exception: access violation`, no traceback
# and no Python-level error.
#
# Measured, cupy 14.1.1 (CUDA 13.02) against torch 2.12.1+cu130:
#
#     import torch, cupy   -> BatchedSeqTrainer trains fine
#     import cupy, torch   -> access violation in `act @ self.W`
#     import cupy alone, running NO CuPy operations -> same crash
#
# It took down two test files (test_batched_trainer, test_batched_next_token)
# whose own code is correct -- they pass standalone and crash under pytest,
# because conftest imports this package first. `find_spec` answers the same
# question with no CUDA initialisation at all.
from importlib.util import find_spec as _find_spec

GPU_AVAILABLE = _find_spec("cupy") is not None

# Version kept in sync with pyproject.toml for the installed package
__version__ = "0.0.1a1"  # kept in sync with pyproject.toml
__author__ = "Superintelligent Group"

__all__ = [
    # Core classes
    'Brain', 'Area', 'Stimulus', 'Connectome', 'HomeostasisConfig',

    # Compute engine API
    'ComputeEngine', 'ProjectionResult', 'create_engine', 'list_engines',

    # Mathematical engines
    'StatisticalEngine', 'NeuralComputationEngine',
    'TopKPolicy', 'ThresholdPolicy', 'RelativeThresholdPolicy', 'EPercentPolicy', 'WinnerPolicy',
    'WinnerSelector', 'PlasticityEngine',

    # Constants
    'DEFAULT_P', 'DEFAULT_BETA',

    # Utilities
    'normalize_features', 'select_top_k_indices', 'heapq_select_top_k', 'binomial_ppf',

    # Assembly Calculus operations
    'Assembly', 'AssemblyTrace', 'PatternCompletionDiagnostic',
    'ResponseDiagnostic', 'ResponseTrace', 'TraceStep',
    'overlap', 'chance_overlap',
    'project', 'reciprocal_project', 'associate', 'merge', 'pattern_complete', 'separate',
    'project_trace', 'reciprocal_project_trace', 'associate_trace',
    'merge_trace', 'pattern_complete_trace', 'ordered_recall_trace',
    'snapshot_area', 'source_response_traces',
    'FiberCircuit',

    # GPU availability flag
    'GPU_AVAILABLE',
]
