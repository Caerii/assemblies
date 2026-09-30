"""The refracted arc-and-state core of the sequence organs, once.

`NemoArcFSM` (state ASSIGNED) and `SequenceTransducer` (state INDUCED) share
exactly this: a STATE area, a REFRACTED arc area, and -- when the organ
carries its own density -- the two structural connectivities between them
at `organ_p`, set before any traffic. What differs (symbols vs words, the
teacher-forced write vs the induced one, LEX and OUT) stays in each class.
This retires the duplication the transducer's docstring left standing "for
one commit". The hashed substrate's counterpart is
`core/torch_engine/_arc_core.py`.

Area creation order is preserved from both classes (STATE, then ARC).
"""
from __future__ import annotations

from typing import Tuple


def add_arc_state_core(brain, prefix: str, *, n_arc: int, n_state: int, k: int,
                       beta: float, refracted_strength: float,
                       state_refracted_strength: float = 0.0,
                       organ_p: float | None = None) -> Tuple[str, str]:
    """Add ``{prefix}_state`` and ``{prefix}_arc`` (refracted) to `brain`,
    with STATE <-> ARC at `organ_p` when given. Returns the two names."""
    state_area, arc_area = f"{prefix}_state", f"{prefix}_arc"
    # THE STATE MAY BE REFRACTED TOO, and by default is not (the
    # configuration A3 measured collapsing); turning it on is a
    # re-measurement, PREREG_state_refraction.md.
    if state_refracted_strength > 0:
        brain.add_area(state_area, n_state, k, beta, refracted=True,
                       refracted_strength=state_refracted_strength)
    else:
        brain.add_area(state_area, n_state, k, beta)
    brain.add_area(arc_area, n_arc, k, beta, refracted=True,
                   refracted_strength=refracted_strength)
    # LOCAL REGIME: connectivity is structural, set before any traffic
    # ([[SEQ-REGIME]], [[SEQ-ORGAN-EMBEDS]]).
    if organ_p is not None:
        brain.add_connectivity(state_area, arc_area, organ_p)
        brain.add_connectivity(arc_area, state_area, organ_p)
    return state_area, arc_area
