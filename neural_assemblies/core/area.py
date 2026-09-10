# area.py
"""
Neural Area Simulation

This module implements the Area class for simulating individual brain areas
within the neural assembly framework. Each area represents a distinct brain
region with its own neural population and assembly dynamics.

Biological Context:
- Models cortical columns, brain regions, or functional areas
- Implements sparse neural coding: only k neurons fire per timestep
- Tracks neural activity patterns and assembly formation
- Supports both explicit (full simulation) and sparse (statistical) modes

Assembly Calculus Context:
- Each area can contain multiple neural assemblies
- Assemblies are sets of k co-active neurons representing concepts
- Assembly formation follows winner-take-all competition
- Supports assembly fixation for stable representations

Mathematical Foundation:
- Sparse coding: k/n ratio determines representation sparsity
- Winner-take-all: Only top-k neurons with highest inputs fire
- Hebbian plasticity: Synaptic weights strengthen with co-activation
- Statistical efficiency: Sparse simulation for large populations
"""

import numpy as np
from typing import Dict, List, Optional

from .backend import get_xp, xp_by_name, xp_name
from .index_spaces import CompactIdx, validated_indices
from .activity import ActivityState


class Area(ActivityState):
    """
    Neural Area for Assembly Simulation
    
    Represents a brain area containing a population of neurons that can form
    neural assemblies through co-activation and learning. Each area implements
    sparse neural coding where only a small fraction of neurons fire per timestep.
    
    This class models the fundamental unit of neural computation in the Assembly
    Calculus framework, where assemblies of co-active neurons represent concepts
    and perform computations through their interactions.
    
    Biological Principles:
    - Sparse coding: Only k neurons fire per timestep (k << n)
    - Assembly formation: Co-active neurons form stable assemblies
    - Plasticity: Synaptic weights adapt through Hebbian learning
    - Hierarchical processing: Areas can project to other areas
    
    Assembly Calculus Operations:
    - Assembly creation: Winner-take-all selection forms new assemblies
    - Assembly fixation: Stable assemblies can be frozen for reuse
    - Assembly projection: Assemblies can project to other areas
    - Assembly association: Assemblies can become more similar through co-activation
    
    References:
    - Papadimitriou, C. H., et al. "Brain Computation by Assemblies of Neurons." 
      Proceedings of the National Academy of Sciences 117.25 (2020): 14464-14472.
    - Mitropolsky, D., et al. "The Architecture of a Biologically Plausible 
      Language Organ." 2023.
    """

    _activity_fields = ("_winners", "w", "_num_ever_fired", "_new_winners", "_new_w",
                        "num_first_winners", "fixed_assembly", "ever_fired", "num_ever_fired")
    _activity_history_fields = ("saved_winners", "saved_w")

    def __init__(
        self,
        name: str,
        n: int,
        k: int,
        beta: float = 0.05,
        explicit: bool = False,
        refractory_period: int = 0,
        inhibition_strength: float = 0.0,
        refracted: bool = False,
        refracted_strength: float = 0.0,
        winner_policy=None,
        input_noise_std: float = 0.0,
        slot_count: int = 0,
    ):
        """
        Initializes the Area.

        Args:
            name (str): Name of the area.
            n (int): Number of neurons in the area.
            k (int): Number of neurons that can fire at any time step.
            beta (float): Default synaptic plasticity parameter.
            explicit (bool): Whether the area is fully simulated (explicit).
            refractory_period (int): Number of steps of LRI suppression
                (0 = disabled).  When > 0, recently-fired neurons receive
                a penalty during winner selection.
            inhibition_strength (float): Magnitude of the LRI penalty.
            refracted (bool): Whether refracted mode is enabled.
                When True, a cumulative bias grows each time a neuron
                fires, making repeated firing progressively harder.
            refracted_strength (float): Magnitude of the per-firing
                bias increment in refracted mode.
        """
        from .registration import validate_area_registration, validate_slot_configuration
        n, k = validate_area_registration(name, n, k)
        from ._homeostasis import validate_lri_parameters
        refractory_period, inhibition_strength = validate_lri_parameters(
            refractory_period, inhibition_strength)
        self.name = name
        self.n = n
        self.k = k
        self.beta = beta
        self.explicit = explicit
        self.refractory_period = refractory_period
        self.inhibition_strength = inhibition_strength
        self.refracted = refracted
        self.refracted_strength = refracted_strength
        self.winner_policy = winner_policy
        self.input_noise_std = input_noise_std
        self.slot_count = validate_slot_configuration(n, slot_count, winner_policy)

        # Captured once, like the engine's `_xp`. Re-reading the global here
        # meant the winners SETTER converted into whatever backend was selected
        # most recently in the process, so an Area belonging to a numpy Brain
        # started storing CuPy arrays the moment a GPU engine was built
        # anywhere. The values stayed correct; the container did not, and
        # `isinstance(area.winners, np.ndarray)` began failing far from the
        # cause.
        xp = get_xp()
        self._xp_name = xp_name(xp)
        self._winners = xp.array([], dtype=xp.uint32)
        # `w` MEANS TWO THINGS. Between projections the engine syncs it to
        # num-ever-fired; the `winners` setter then overwrites it with
        # `len(winners)`. Both readings are load-bearing, so `_num_ever_fired`
        # tracks the recruitment one separately and survives a winners
        # assignment -- read it via `get_num_ever_fired()`, and use
        # `active_count` for the other meaning.
        self.w = 0
        self._num_ever_fired = 0
        self.fixed_assembly = False

        # Temporary state for projection updates (matches brain.py)
        self._new_winners = xp.array([], dtype=xp.uint32)
        self._new_w = 0
        self.num_first_winners = -1

        if explicit:
            self.ever_fired = xp.zeros(self.n, dtype=bool)
            self.num_ever_fired = 0

        self.beta_by_stimulus: Dict[str, float] = {}
        self.beta_by_area: Dict[str, float] = {}
        self.saved_winners: List[np.ndarray] = []
        self.saved_w: List[int] = []

        # Sparse mapping: compact winner index -> actual neuron id
        # And a pre-shuffled pool of neuron ids for assigning to new winners
        self.compact_to_neuron_id: List[int] = []
        self.neuron_id_pool: Optional[np.ndarray] = None  # set by Brain.add_area
        self.neuron_id_pool_ptr: int = 0

    @property
    def winners(self) -> CompactIdx:
        """COMPACT ENGINE INDICES, ``0..w-1`` -- NOT neuron IDs.

        The return type is what stops this being confused with
        ``Assembly.winners``, which holds stable neuron IDs in ``0..n-1``. The
        two are both uint32 arrays and are COMPLETELY DISJOINT as measured, so
        mixing them returns a number that reads as exactly chance. Convert with
        ``index_spaces.to_neuron_ids`` (or use ``diagnostics.read_assembly``,
        the sanctioned readout) before comparing against anything stored.

        Compact indices are also NOT STABLE: they are reassigned as the area
        materializes more neurons, so one saved across a projection is a bug
        waiting to be dereferenced.
        """
        return CompactIdx(self._winners)

    @property
    def _xp(self):
        """This area's array module, pinned at construction.

        A NAME is stored rather than the module, because Areas are pickled and
        deep-copied constantly (fork, checkpoint, disk cache, probes) and a
        module cannot be pickled.
        """
        return xp_by_name(self._xp_name)

    @property
    def active_count(self) -> int:
        """How many neurons are firing NOW.

        One of the two things `w` means. The other is `get_num_ever_fired()`.
        Say which you mean; `w` alone does not.
        """
        return len(self._winners)

    @winners.setter
    def winners(self, value):
        """Specification: docs/reviews/whole-codebase/SEMANTIC_CARDS.md#contract-mixed-drive-indices

        Validate compact positions before conversion or activity-count mutation.
        """
        xp = self._xp
        self._winners = validated_indices(value, upper=self.n,
                                          label=f"{self.name} compact winners", xp=xp, unique=True)
        # CLOBBERS the num-ever-fired meaning of `w`. Preserved because the
        # projection loop and several callers depend on `w` tracking the cap
        # between engine syncs; use `num_ever_fired` / `active_count` to say
        # which one you mean.
        self.w = len(self._winners)

    def fix_assembly(self):
        """Freezes the current assembly, preventing it from changing."""
        if len(self.winners) == 0:
            raise ValueError(f"Area {self.name} has no winners to fix.")
        self.fixed_assembly = True

    def unfix_assembly(self):
        """Allows the assembly to change in future simulations."""
        self.fixed_assembly = False

    def update_beta_by_stimulus(self, stimulus_name: str, new_beta: float):
        """DEAD ROUTE. Use `Brain.update_plasticities(stim_update_map=...)`.

        See `update_beta_by_area` for why this raises instead of working.
        """
        raise NotImplementedError(
            f"Area.update_beta_by_stimulus({stimulus_name!r}, {new_beta}) does "
            f"not reach the engine, so it would change nothing. Use\n"
            f"    brain.update_plasticities(stim_update_map="
            f"{{{self.name!r}: [({stimulus_name!r}, {new_beta})]}})"
        )

    def update_beta_by_area(self, area_name: str, new_beta: float):
        """DEAD ROUTE. Use `Brain.update_plasticity(from_area, to_area, beta)`.

        Writing `self.beta_by_area` alone is a silent no-op on every engine
        except the legacy dense `compute.explicit_projection` path: the sparse,
        torch and cuda engines all read their own `AreaState.beta_by_source`,
        which only `engine.set_beta` writes. An `Area` cannot forward to the
        engine itself -- it holds no engine handle on purpose, because Areas are
        pickled and deep-copied constantly (that is also why `_xp` stores a
        name, not a module).

        `Brain.update_plasticity` writes BOTH, which is what keeps the two
        readers agreeing. This raises rather than no-ops because the silent
        version cost a full 10-seed run that reported two identical arms as a
        negative result.
        """
        raise NotImplementedError(
            f"Area.update_beta_by_area({area_name!r}, {new_beta}) does not "
            f"reach the engine, so it would change nothing. Use\n"
            f"    brain.update_plasticity({area_name!r}, {self.name!r}, {new_beta})"
        )

    def get_num_ever_fired(self) -> int:
        """Neurons that have EVER fired -- recruitment, not current activity.

        Reads `_num_ever_fired`, NOT `w`. `w` carries this value only until
        something assigns to `winners`, after which it is `len(winners)` --
        `k` by construction for a k-cap, 0 after a clear. Returning `w` here
        made this accessor report `k` for any area whose winners had been set
        directly (probes, `inhibit_areas`, fixed assemblies), which reads as a
        sealed area. See [[fake-perfect-probe-signatures]].
        """
        if self.explicit:
            return self.num_ever_fired
        # `getattr` default, not attribute access: an Area unpickled from a
        # checkpoint written before `_num_ever_fired` existed does not have
        # the field, and `checkpoint.py` loads exactly such files from disk.
        # Falling back to `w` restores the OLD semantics for those -- stale
        # after a winners assignment, but that is what they were recorded
        # with, and it beats an AttributeError on load.
        #
        # `max` rather than either alone because the two mirror the same
        # quantity with different failure modes: `w` is fresh but clobberable,
        # `_num_ever_fired` survives the setter but is absent on old pickles.
        return max(int(getattr(self, "_num_ever_fired", 0)), int(self.w))

    def _update_winners(self, new_winners):
        """
        Updates the winners and records the state.

        Args:
            new_winners: The new winners to set.
        """
        xp = self._xp
        self.winners = new_winners
        if self.explicit:
            self.ever_fired[new_winners] = True
            self.num_ever_fired = int(xp.sum(self.ever_fired))
        if self.saved_winners is not None:
            self.saved_winners.append(new_winners)
        if self.saved_w is not None:
            self.saved_w.append(self.w)
