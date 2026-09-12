"""The index-space TYPES must reject mixing, and this proves they still do.

COMPANION TO `test_index_space_ratchet.py`, NOT A REPLACEMENT. That ratchet
freezes a per-file count of risky lines and says so in its own docstring: *"The
defect is MIXING the two spaces, which needs dataflow analysis to detect
properly."* A type checker IS that dataflow analysis. The ratchet contains the
legacy sites; these types stop new ones being written.

WHY THIS TEST ALSO RUNS THE API. The branded arrays retain their index-space
identity at runtime, so the public overlap boundary can reject mixed values
before computing a meaningless number. Pyright checks the static contract as
well, and the runtime test constructs both cases explicitly:

  * three calls that MUST error (one of each space, and Assembly vs raw array)
  * three calls that MUST NOT error (same space on both sides)

If pyright is unavailable the test SKIPS rather than passing vacuously -- a
silent pass here would be indistinguishable from a working guard.
"""
from __future__ import annotations

import json

import numpy as np
import os
import shutil
import subprocess
import tempfile

import pytest

from neural_assemblies.assembly_calculus.assembly import overlap
from neural_assemblies.core.index_spaces import CompactIdx, NeuronIds

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Line numbers are asserted, so keep the layout stable when editing.
PROBE = '''\
import numpy as np
from neural_assemblies.assembly_calculus.assembly import Assembly, overlap
from neural_assemblies.core.index_spaces import CompactIdx, NeuronIds

compact = CompactIdx(np.array([1, 2, 3], dtype=np.uint32))
neurons = NeuronIds(np.array([77, 88, 99], dtype=np.uint32))
asm = Assembly(area="A", winners=neurons)

overlap(compact, neurons)   # line 9  MUST ERROR
overlap(neurons, compact)   # line 10 MUST ERROR
overlap(asm, compact)       # line 11 MUST ERROR
overlap(compact, compact)   # line 12 must be clean
overlap(neurons, neurons)   # line 13 must be clean
overlap(asm, asm)           # line 14 must be clean
'''

MUST_ERROR = {9, 10, 11}
MUST_BE_CLEAN = {12, 13, 14}


def test_mixing_index_spaces_is_rejected_at_runtime():
    compact = CompactIdx(np.array([1, 2, 3], dtype=np.uint32))
    neurons = NeuronIds(np.array([77, 88, 99], dtype=np.uint32))
    with pytest.raises(TypeError, match="same index space"):
        overlap(compact, neurons)
    assert overlap(compact, compact) == 1.0
    assert overlap(neurons, neurons) == 1.0


def _pyright_error_lines(source: str) -> set:
    """Run pyright on *source* placed inside the repo, return 1-based lines."""
    fd, path = tempfile.mkstemp(suffix=".py", dir=REPO, prefix="_idxspace_probe_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(source)
        proc = subprocess.run(
            ["pyright", "--outputjson", path],
            capture_output=True, text=True, cwd=REPO, timeout=300,
        )
        payload = json.loads(proc.stdout)
        return {
            d["range"]["start"]["line"] + 1
            for d in payload.get("generalDiagnostics", [])
            if d.get("severity") == "error"
        }
    finally:
        os.unlink(path)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("pyright") is None,
                    reason="pyright not installed; the guard cannot be verified")
def test_mixing_index_spaces_is_a_type_error():
    """Both halves asserted: it fires on the defect AND stays quiet otherwise.

    Asserting only the first half would pass for a checker that rejects
    everything, which is not a guard but a wall.
    """
    errored = _pyright_error_lines(PROBE)

    missed = MUST_ERROR - errored
    assert not missed, (
        f"index-space mixing was NOT flagged on lines {sorted(missed)}. The "
        f"types have lost their power -- check the `overlap` overloads and that "
        f"`Area.winners`/`Assembly.winners` still carry CompactIdx/NeuronIds."
    )

    false_alarms = MUST_BE_CLEAN & errored
    assert not false_alarms, (
        f"same-space calls were rejected on lines {sorted(false_alarms)}. The "
        f"guard is over-firing, which pushes callers to delete the annotations "
        f"and is worse than no guard."
    )
