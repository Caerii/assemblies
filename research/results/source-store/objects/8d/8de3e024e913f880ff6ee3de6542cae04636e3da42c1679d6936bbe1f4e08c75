"""HashedArcCore: the refracted arc-and-state core, once.

`HashedTransducer` (state INDUCED) and `HashedArcFSM` (state ASSIGNED)
share exactly this and nothing else: a refracted ARC area, a STATE area,
the two organ fibers between them at the organ's density, and the two
moves of the clock --

    conjoin   the arc recomputed from (state, inputs): ONE round, the
              state fiber priced from STATE's current winners;
    advance   STATE recomputed from the arc, feed-forward, ONE round.

What differs stays outside: the transducer's LEX/OUT and its repeated
writes onto a fixed arc; the machine's assigned blocks and teacher-forced
write. The duplication the numpy transducer's docstring left standing
"for one commit" is retired here and in `programs/arc_core.py`.

FIBER ORDER IS THE CALLER'S. Drives sum in the order fibers are given and
float addition is not associative ([[exact-tables-are-tie-fragile]]); each
organ passes its own ordered list, so its measured tables do not move.
"""
from __future__ import annotations

from typing import Any

from ._hashed import DenseOrganFiber, HashedArea
from ._hashed_aligner import pair_seeds


class HashedArcCore:
    def __init__(self, brain_seeds, *, prefix: str, n_arc: int, n_state: int, k: int,
                 p: float, beta: float, refracted_strength: float,
                 state_refracted_strength: float = 0.0,
                 w_max: float | None = 20.0,
                 norm_init: bool = True, max_potentiations: int = 4096,
                 tie_jitter: float = 0.0, device: str = "cuda"):
        S = [int(s) for s in brain_seeds]
        self.B, self.k, self.n_arc, self.n_state = len(S), int(k), int(n_arc), int(n_state)
        self.p, self.device = float(p), device
        self.arc_area, self.state_area = f"{prefix}_arc", f"{prefix}_state"
        self.arc = HashedArea(self.n_arc, k, pair_seeds(S, self.arc_area, self.arc_area),
                              device=device, refracted_strength=refracted_strength,
                              tie_jitter=tie_jitter)
        self.state = HashedArea(self.n_state, k, pair_seeds(S, self.state_area, self.state_area),
                                device=device, refracted_strength=state_refracted_strength,
                                tie_jitter=tie_jitter)
        self.state_arc = DenseOrganFiber(pair_seeds(S, self.state_area, self.arc_area),
                                         self.n_state, self.n_arc, p, beta=beta, w_max=w_max,
                                         norm_init=norm_init, max_rounds=max_potentiations,
                                         device=device)
        self.arc_state = DenseOrganFiber(pair_seeds(S, self.arc_area, self.state_area),
                                         self.n_arc, self.n_state, p, beta=beta, w_max=w_max,
                                         norm_init=norm_init, max_rounds=max_potentiations,
                                         device=device)

    def rows(self):
        """Source winners per core fiber, for `rows_for`."""
        return {id(self.state_arc): self.state.winners,
                id(self.arc_state): self.arc.winners}

    def conjoin(self, fibers, freeze: bool = False, rows_for=None):
        """Recompute ARC from `fibers` (the caller's ORDERED afferents, the
        core's `state_arc` among them), one round."""
        rf = dict(self.rows())
        if rows_for:
            rf.update(rows_for)
        return self.arc.project(1, fibers, rows_for=rf, freeze=freeze)

    def advance(self, freeze: bool = False):
        """Recompute STATE from ARC, one round, feed-forward."""
        return self.state.project(1, [self.arc_state], rows_for=self.rows(), freeze=freeze)

    def teach(self, target: Any) -> None:
        """The machine's teacher-forced write: ARC -> STATE onto a PINNED
        `target` [B, k]; no drive is computed (the engine's fixed-target
        plasticity)."""
        self.state.winners = target
        self.arc_state.begin_episode()
        self.arc_state.observe(self.arc.winners, target)
        self.arc_state.end_episode()
        self.state.ever.scatter_(1, target, True)

    def inhibit(self, mask=None) -> None:
        for a in (self.arc, self.state):
            if mask is None:
                a.inhibit()
            else:
                a.inhibit_rows(mask)

    def check(self) -> None:
        self.state_arc.check()
        self.arc_state.check()
