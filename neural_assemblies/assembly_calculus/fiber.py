"""
FiberCircuit — declarative gating of projection channels.

In the Assembly Calculus, a "fiber" is a directed projection channel
between two brain areas (or from a stimulus to an area). The parser
(Mitropolsky et al. 2023) uses INHIBIT/DISINHIBIT rules to gate
which fibers are active at each parsing step.

FiberCircuit generalizes this pattern: declare the possible connections,
then inhibit/disinhibit them as needed. Each ``step()`` call translates
the current fiber state into a single ``brain.project()`` call.

Why gating is the whole control story.  NEMO has no program counter, no
registers, and no way to address an assembly by name; the only thing an outer
controller may do is open and close fibers.  So "control flow" in this model
IS fiber state, and the biological claim is modest and specific: a small
number of disinhibitory interneuron populations can switch cortico-cortical
pathways on and off far faster than synaptic weights change.  Everything the
parser does -- selecting a syntactic role, discarding a completed phrase,
routing a word to one area rather than another -- is expressed as which
fibers are open on that time step.

Two consequences that surprise readers:

* A fiber's *weights persist while it is inhibited*.  Inhibiting is muting a
  channel for a time step, not deleting what was learned through it.  Reopen
  it and the previously potentiated synapses are all still there.
* An inhibited fiber also stops LEARNING, because no input flows through it,
  so nothing is co-active to potentiate.  Gating therefore controls plasticity
  as well as activation, which is how the parser keeps unrelated word pairs
  from silently associating.
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

        This is how you ask what the network does ON ITS OWN.  With stimuli
        clamped off, nothing external constrains the winners, so the areas
        settle purely on their learned weights -- which is the setting in
        which pattern completion, sequence recall, and prediction are read
        out.  Any measurement that leaves a stimulus firing is measuring the
        input as much as the network.

        The restore runs in a ``finally``: a projection that raises mid-loop
        must not leave every stimulus fiber inhibited, because the exception is
        usually caught somewhere above and the circuit is then silently deaf to
        all input for the rest of the session -- a failure that shows up as
        wrong results much later, far from its cause.  Successful runs are
        unaffected.

        Args:
            n: Number of autonomous steps (default 1).
        """
        saved_stim = {k: v for k, v in self._stim_fibers.items()}
        for key in self._stim_fibers:
            self._stim_fibers[key] = False

        try:
            for _ in range(n):
                self.step()
        finally:
            self._stim_fibers.update(saved_stim)
