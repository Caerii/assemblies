"""Backward-compatible shim — prefer ``evaluation.erp.calibration``."""
from .._compat import deprecate_shim
deprecate_shim(__name__, "neural_assemblies.assembly_calculus.emergent.evaluation.erp.calibration")
from .erp.calibration import *  # noqa: F401,F403
from .erp.frames import (  # noqa: F401
    DEFAULT_CALIBRATION_FRAMES,
    PositionErpSample,
    collect_frame_samples,
    collect_position_samples,
)
