"""
FiberCircuit — declarative gating of projection channels.

In the Assembly Calculus, a "fiber" is a directed projection channel
between two brain areas (or from a stimulus to an area). The parser
(Mitropolsky et al. 2023) uses INHIBIT/DISINHIBIT rules to gate
which fibers are active at each parsing step.

FiberCircuit generalizes this pattern: declare the possible connections,
then inhibit/disinhibit them as needed. Each ``step()`` call translates
the current fiber state into a single ``brain.project()`` call.
"""

from collections import defaultdict


class FiberCircuit:
    """Manages a set of fibers (projection channels) with gating.

    A fiber is a directed connection from a source to a target.
    Sources can be brain areas or stimuli. Fibers can be inhibited
    (disabled) or disinhibited (enabled) to control information flow.

    Example::

        circuit = FiberCircuit(brain)
        circuit.add("A", "B")
        circuit.add("B", "C")
        circuit.add("C", "A")

        circuit.step()  # Projects along all three fibers

        circuit.inhibit("C", "A")
        circuit.step()  # Only A→B and B→C are active

        circuit.disinhibit("C", "A")
        circuit.step()  # All three active again
    """

    def __init__(self, brain):
        self.brain = brain
        self._fibers = {}       # (source_area, target_area) -> active
        self._stim_fibers = {}  # (stimulus, target_area) -> active

    def add(self, source, target, active=True):
        """Declare an area-to-area fiber.

        Args:
            source: Source area name.
            target: Target area name.
            active: Whether the fiber starts active (default True).
        """
        self._fibers[(source, target)] = active

    def add_stim(self, stimulus, target, active=True):
        """Declare a stimulus-to-area fiber.

        Args:
            stimulus: Stimulus name.
            target: Target area name.
            active: Whether the fiber starts active (default True).
        """
        self._stim_fibers[(stimulus, target)] = active

    def inhibit(self, source, target):
        """Disable a fiber. Subsequent step() calls will skip it.

        Raises KeyError if the fiber was not previously declared.
        """
        if (source, target) in self._fibers:
            self._fibers[(source, target)] = False
        elif (source, target) in self._stim_fibers:
            self._stim_fibers[(source, target)] = False
        else:
            raise KeyError(f"No fiber declared from {source!r} to {target!r}")

    def disinhibit(self, source, target):
        """Re-enable a fiber.

        Raises KeyError if the fiber was not previously declared.
        """
        if (source, target) in self._fibers:
            self._fibers[(source, target)] = True
        elif (source, target) in self._stim_fibers:
            self._stim_fibers[(source, target)] = True
        else:
            raise KeyError(f"No fiber declared from {source!r} to {target!r}")

    def is_active(self, source, target) -> bool:
        """Query whether a fiber is currently active."""
        if (source, target) in self._fibers:
            return self._fibers[(source, target)]
        if (source, target) in self._stim_fibers:
            return self._stim_fibers[(source, target)]
        raise KeyError(f"No fiber declared from {source!r} to {target!r}")

    def active_area_projections(self) -> dict:
        """Returns dst_areas_by_src_area for active area fibers.

        Format: {source_area: [target_area, ...]} with only active fibers.
        """
        result = defaultdict(list)
        for (src, tgt), active in self._fibers.items():
            if active:
                result[src].append(tgt)
        return dict(result)

    def active_stim_projections(self) -> dict:
        """Returns areas_by_stim for active stimulus fibers.

        Format: {stimulus: [target_area, ...]} with only active fibers.
        """
        result = defaultdict(list)
        for (stim, tgt), active in self._stim_fibers.items():
            if active:
                result[stim].append(tgt)
        return dict(result)

    def step(self):
        """Execute one projection step using only active fibers.

        Builds areas_by_stim and dst_areas_by_src_area from the
        current fiber state, then calls brain.project().
        """
        areas_by_stim = self.active_stim_projections()
        dst_areas_by_src_area = self.active_area_projections()
        self.brain.project(areas_by_stim, dst_areas_by_src_area)

    def autonomous_step(self, n: int = 1):
        """Execute *n* projection steps using only area-to-area fibers.

        Temporarily inhibits all stimulus fibers so that only
        area-to-area (recurrent/feedforward) projections execute.
        Stimulus fiber states are restored afterward.

        Args:
            n: Number of autonomous steps (default 1).
        """
        saved_stim = {k: v for k, v in self._stim_fibers.items()}
        for key in self._stim_fibers:
            self._stim_fibers[key] = False

        for _ in range(n):
            self.step()

        self._stim_fibers.update(saved_stim)
