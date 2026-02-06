"""
Assembly Calculus Brain Simulation (Root Implementation)
=========================================================

Reference implementation of the Assembly Calculus framework for neural
computation, as described in:

  Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., &
  Maass, W. "Brain Computation by Assemblies of Neurons."
  PNAS 117(25): 14464-14472, 2020.

This module implements the core Brain and Area classes for simulating neural
assemblies. The key operations are:

  - **Projection** (A → B): Assembly in area A activates neurons in area B;
    top-k winners become the new assembly. Hebbian plasticity strengthens
    connections between co-active neurons.

  - **Reciprocal Projection** (A ↔ B): Bidirectional projection that tests
    whether B can restore A's assembly after learning.

  - **Association**: Simultaneous projection of two assemblies into a shared
    area increases their overlap, linking the represented concepts.

  - **Merge** (A + B → C): Two assemblies project simultaneously into a
    downstream area, forming a combined representation.

Mathematical details:
  - Winner selection: top-k by accumulated synaptic input
  - Plasticity rule: w_new = w_old * (1 + β), with optional saturation at w_max
  - Sparse simulation: new winner inputs sampled from a truncated normal
    approximation of Binomial(total_k, p)
  - Connectome initialization: Bernoulli(p) per synapse

Original author: Daniel Mitropolsky, 2018
"""

import numpy as np
import heapq
import collections
from scipy.stats import binom, truncnorm
import math
import types

# Import extracted math primitives
try:
    from src.math_primitives.explicit_projection import (
        validate_source_winners,
        accumulate_prev_inputs_explicit,
        apply_area_to_area_plasticity
    )
    from src.math_primitives.image_activation import (
        preprocess_image,
        normalize_and_select_topk
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from math_primitives.explicit_projection import (
        validate_source_winners,
        accumulate_prev_inputs_explicit,
        apply_area_to_area_plasticity
    )
    from math_primitives.image_activation import (
        preprocess_image,
        normalize_and_select_topk
    )

EMPTY_MAPPING = types.MappingProxyType({})

# Default weight ceiling for Hebbian plasticity.
# Prevents unbounded weight growth over long simulations.
# See: Dabagia et al. "Coin-Flipping in the Brain" (2024), Section 3.
DEFAULT_W_MAX = 20.0

class Area:
  """A brain area representing a neural population with winner-take-all dynamics.

  Each area contains n neurons, of which k fire per timestep (the "assembly").
  The assembly evolves through projection operations governed by Hebbian
  plasticity. In sparse mode, only neurons that have ever fired are tracked
  explicitly; in explicit mode, all n neurons are simulated.

  Attributes:
    name: Symbolic tag identifying this area (must be unique per Brain).
    n: Total number of neurons in the area.
    k: Assembly size — number of neurons that fire per timestep.
    beta: Default Hebbian plasticity coefficient (weight multiplier = 1 + beta).
    beta_by_stimulus: Per-stimulus plasticity overrides {stim_name: beta}.
    beta_by_area: Per-source-area plasticity overrides {area_name: beta}.
    w: Support size — number of neurons that have ever fired in this area.
        For explicit areas, w == n from initialization.
    saved_w: History of w values after each projection round.
    winners: Current assembly (indices of firing neurons), as np.uint32 array.
    saved_winners: History of winner arrays after each projection round.
    num_first_winners: Count of first-time winners from the last projection.
    fixed_assembly: If True, winners are frozen and projection skips this area.
    explicit: If True, simulate all n neurons; if False, use sparse mode.
  """
  def __init__(self, name, n, k, *,
               beta=0.05, w=0, explicit=False):
    """Initializes a brain area.

    Args:
      name: Area name (symbolic tag), must be unique within a Brain.
      n: Total number of neurons in the area.
      k: Number of neurons that fire per timestep (assembly size).
           Typically k = sqrt(n) following the convention in Papadimitriou et al.
      beta: Default Hebbian plasticity coefficient. Connection weights are
            multiplied by (1 + beta) when pre- and post-synaptic neurons co-fire.
      w: Initial support size (number of neurons that have ever fired).
         Set to n for explicit areas, 0 for sparse areas.
      explicit: If True, all n neurons are explicitly simulated with full
                weight matrices. If False, sparse simulation tracks only
                neurons that have fired, using statistical sampling for new
                winner candidates.
    """
    self.name = name
    self.n = n
    self.k = k
    self.beta = beta
    self.beta_by_stimulus = {}
    self.beta_by_area = {}
    self.w = w
    self._new_w = 0
    self.saved_w = []
    self.winners = np.array([], dtype=np.uint32)
    self._new_winners = []
    self.saved_winners = []
    self.num_first_winners = -1
    self.fixed_assembly = False
    self.explicit = explicit

    if explicit:
      self.ever_fired = np.zeros(self.n, dtype=bool)
      self.num_ever_fired = 0

  def _update_winners(self):
    """Updates `winners` and `w` after a projection step.

    If `explicit` is `False`, the `w` attribute is updated to the value of
    `_new_w`. The `winners` attribute is updated to the value of `_new_winners`.
    """
    # self.winners = self._new_winners
    self.winners = np.array(self._new_winners, dtype=np.uint32)

    if not self.explicit:
      self.w = self._new_w

  def update_beta_by_stimulus(self, name, new_beta):
    """Updates the synaptic plasticity parameter for synapses from a specific stimulus.

    Args:
      name: The name of the stimulus that the synapses come from.
      new_beta: The new synaptic plasticity parameter.
    """
    self.beta_by_stimulus[name] = new_beta

  def update_area_beta(self, name, new_beta):
    """Updates the synaptic plasticity parameter for synapses from a specific area.

    Args:
      name: The name of the area that the synapses come from.
      new_beta: The new synaptic plasticity parameter.
    """
    self.beta_by_area[name] = new_beta

  def fix_assembly(self):
    """Freezes the current assembly so it persists unchanged through projections.

    When an area's assembly is fixed, ``project_into`` skips winner selection
    and returns the existing winners unchanged. This is used for:
      - Creating stable "concept" representations for testing retrieval.
      - Holding a stimulus pattern fixed while projecting to downstream areas.

    Raises:
      ValueError: If no assembly exists in this area (winners is empty).
    """
    if self.winners.size == 0:
      raise ValueError(f'Area {self.name!r} does not have assembly; cannot fix.')
    self.fixed_assembly = True

  def unfix_assembly(self):
    """
    Allows the assembly to change in future simulations.

    This is the opposite of `.fix_assembly()`.
    """
    
    self.fixed_assembly = False

  def get_num_ever_fired(self):
    """
    Returns the total number of neurons that have ever fired in this area.

    Returns:
      int: The number of neurons that have ever fired in this area.
    """
    
    if self.explicit:
      return self.num_ever_fired 
    else:
      return self.w

class Brain:
  """A model brain implementing the Assembly Calculus framework.

  Orchestrates neural simulation across multiple brain areas connected by
  synaptic weight matrices (connectomes). Each projection step:
    1. Accumulates synaptic input to each neuron in the target area.
    2. Selects the top-k neurons (winners) as the new assembly.
    3. Applies Hebbian plasticity: w *= (1 + β), clamped at w_max.
    4. Expands connectomes if new neurons fire for the first time (sparse mode).

  Two simulation modes are supported:
    - **Sparse** (default): Only tracks neurons that have ever fired. New winner
      candidates are sampled from a truncated normal approximation of the
      binomial distribution. Memory: O(w²) where w << n.
    - **Explicit**: All n neurons simulated with full weight matrices.
      Memory: O(n²) per connectome.

  Attributes:
    area_by_name: {name: Area} mapping for all brain areas.
    areas: Alias for area_by_name (backward compatibility).
    stimulus_size_by_name: {stim_name: neuron_count} for stimuli.
    connectomes_by_stimulus: {stim: {area: weight_vector}} stimulus→area weights.
    connectomes: {src_area: {dst_area: weight_matrix}} area→area weights.
    p: Connection probability (Bernoulli parameter for random connectivity).
    w_max: Weight ceiling for Hebbian plasticity (prevents unbounded growth).
    save_size: Whether to record support size history.
    save_winners: Whether to record winner history.
    disable_plasticity: If True, skip all weight updates (for testing).
  """
  def __init__(self, p, save_size=True, save_winners=False, seed=0, w_max=DEFAULT_W_MAX):
    self.area_by_name = {}
    self.areas = {}
    self.stimulus_size_by_name = {}
    self.connectomes_by_stimulus = {}
    self.connectomes = {}
    self.p = p
    self.w_max = w_max
    self.save_size = save_size
    self.save_winners = save_winners
    self.disable_plasticity = False
    self._rng = np.random.default_rng(seed=seed)

  def add_stimulus(self, stimulus_name, size):
    """Add a stimulus to the current instance.

    Args:
      stimulus_name: The name with which the stimulus will be registered.
      size: Number of firing neurons in this stimulus(?).
    """
    self.stimulus_size_by_name[stimulus_name] = size
    this_stimulus_connectomes = {}
    for area_name in self.area_by_name:
      if self.area_by_name[area_name].explicit:
        this_stimulus_connectomes[area_name] = self._rng.binomial(
            size, self.p,
            size=self.area_by_name[area_name].n).astype(np.float32)
      else:
        this_stimulus_connectomes[area_name] = np.empty(0, dtype=np.float32)
      self.area_by_name[area_name].beta_by_stimulus[stimulus_name] = (
        self.area_by_name[area_name].beta)
    self.connectomes_by_stimulus[stimulus_name] = this_stimulus_connectomes

  def add_area(self, area_name, n, k, beta, explicit=False):
    """Add a brain area (sparse by default, explicit if specified).

    Sparse areas track only neurons that have ever fired, with connectomes
    grown dynamically. Explicit areas allocate full n×n weight matrices
    initialized with Bernoulli(p) random connections.

    Args:
      area_name: Unique name for the area.
      n: Total neuron count.
      k: Assembly size (winners per timestep).
      beta: Hebbian plasticity coefficient.
      explicit: If True, use full simulation with n×n weight matrices.
    """
    self.area_by_name[area_name] = the_area = Area(area_name, n, k, beta=beta, explicit=explicit)
    self.areas[area_name] = the_area

    for stim_name, stim_connectomes in self.connectomes_by_stimulus.items():
        if explicit:
            stim_connectomes[area_name] = self._rng.binomial(
                1, self.p, size=n).astype(np.float32)
        else:
            stim_connectomes[area_name] = np.empty(0, dtype=np.float32)
        the_area.beta_by_stimulus[stim_name] = beta

    new_connectomes = {}
    for other_area_name in self.area_by_name:
        other_area = self.area_by_name[other_area_name]
        if explicit:
            other_area_size = other_area.n if other_area.explicit else other_area.w
            new_connectomes[other_area_name] = self._rng.binomial(
                1, self.p, size=(n, other_area_size)).astype(np.float32)
            if other_area_name != area_name:
                self.connectomes[other_area_name][area_name] = self._rng.binomial(
                    1, self.p, size=(other_area.n, n)).astype(np.float32)
        else:
            other_area_size = other_area.n if other_area.explicit else 0
            new_connectomes[other_area_name] = np.empty((0, other_area_size), dtype=np.float32)
            if other_area_name != area_name:
                self.connectomes[other_area_name][area_name] = np.empty(
                    (other_area_size, 0), dtype=np.float32)
        other_area.beta_by_area[area_name] = other_area.beta
        the_area.beta_by_area[other_area_name] = beta
    self.connectomes[area_name] = new_connectomes

  def add_explicit_area(self, area_name, n, k, beta, *,
                      custom_inner_p=None,
                      custom_out_p=None,
                      custom_in_p=None):
    """Add an explicit (fully-simulated) area with full n×n weight matrices.

    Unlike sparse areas, explicit areas set w=n from the start, meaning all
    neurons are candidates for winner selection from the first projection.
    This enables detailed analysis of assembly dynamics but requires O(n²)
    memory per inter-area connectome.

    Args:
      area_name: Unique name for the area.
      n: Total neuron count.
      k: Assembly size (winners per timestep).
      beta: Hebbian plasticity coefficient.
      custom_inner_p: Self-connection probability (default: self.p).
      custom_out_p: Outgoing connection probability (default: self.p).
      custom_in_p: Incoming connection probability (default: self.p).
    """
    self.area_by_name[area_name] = the_area = Area(
        area_name, n, k, beta=beta, w=n, explicit=True)
    self.areas[area_name] = the_area
    the_area.ever_fired = np.zeros(n, dtype=bool)
    the_area.num_ever_fired = 0

    for stim_name, stim_connectomes in self.connectomes_by_stimulus.items():
        stim_connectomes[area_name] = self._rng.binomial(
            1, self.p, size=n).astype(np.float32)
        the_area.beta_by_stimulus[stim_name] = beta

    inner_p = custom_inner_p if custom_inner_p is not None else self.p
    in_p = custom_in_p if custom_in_p is not None else self.p
    out_p = custom_out_p if custom_out_p is not None else self.p

    new_connectomes = {}
    for other_area_name in self.area_by_name:
        if other_area_name == area_name:
            new_connectomes[other_area_name] = self._rng.binomial(
                1, inner_p, size=(n, n)).astype(np.float32)
        else:
            other_area = self.area_by_name[other_area_name]
            if other_area.explicit:
                other_n = self.area_by_name[other_area_name].n
                new_connectomes[other_area_name] = self._rng.binomial(
                    1, out_p, size=(n, other_n)).astype(np.float32)
                self.connectomes[other_area_name][area_name] = self._rng.binomial(
                    1, in_p, size=(other_n, n)).astype(np.float32)
            else:
                new_connectomes[other_area_name] = np.empty((n, 0), dtype=np.float32)
                self.connectomes[other_area_name][area_name] = np.empty((0, n), dtype=np.float32)
        self.area_by_name[other_area_name].beta_by_area[area_name] = (
            self.area_by_name[other_area_name].beta)
        self.area_by_name[area_name].beta_by_area[other_area_name] = beta
    self.connectomes[area_name] = new_connectomes

  def update_plasticity(self, from_area, to_area, new_beta):
    self.area_by_name[to_area].beta_by_area[from_area] = new_beta

  def update_plasticities(self,
                          area_update_map=EMPTY_MAPPING,
                          stim_update_map=EMPTY_MAPPING):
    # area_update_map consists of area1: list[ (area2, new_beta) ]
    # represents new plasticity FROM area2 INTO area1
    for to_area, update_rules in area_update_map.items():
      for from_area, new_beta in update_rules:
        self.update_plasticity(from_area, to_area, new_beta)

    # stim_update_map consists of area: list[ (stim, new_beta) ]f
    # represents new plasticity FROM stim INTO area
    for area, update_rules in stim_update_map.items():
      the_area = self.area_by_name[area]
      for stim, new_beta in update_rules:
        the_area.beta_by_stimulus[stim] = new_beta

  def activate(self, area_name, index):
    area = self.area_by_name[area_name]
    k = area.k
    assembly_start = k * index
    area.winners = list(range(assembly_start, assembly_start + k))
    area.fix_assembly()

  def activate_with_image(self, area_name, image):
    """
    Activates neurons in the given area using raw image data.
    
    Args:
        area_name (str): The name of the brain area to activate.
        image (np.ndarray): The raw image data (flattened or 2D).
    """
    area = self.area_by_name[area_name]

    # Use extracted image activation engine
    image_flat = preprocess_image(image, area.n)
    winners = normalize_and_select_topk(image_flat, area.k, area.n)

    area.winners = winners
    area.w = len(area.winners)

  def project(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
    # Validate stim_area, area_area well defined
    # areas_by_stim: {"stim1":["A"], "stim2":["C","A"]}
    # dst_areas_by_src_area: {"A":["A","B"],"C":["C","A"]}
    
    """
    Simulates neural projections from stimuli and areas into target areas.

    Args:
      areas_by_stim (Dict[str, List[str]]): Mapping from stimuli to target areas.
      dst_areas_by_src_area (Dict[str, List[str]]): Mapping from source areas to target areas.
      verbose (int): Verbosity level.

    Raises:
      IndexError: If a stimulus or area is not in the brain's dictionaries.

    Side effects:

      - Updates the winners of each target area.
      - If `save_winners` is True, saves the new winners in each area's `saved_winners`.
      - If `save_size` is True, saves the new size of each area in each area's `saved_w`.
    """
    stim_in = collections.defaultdict(list)
    area_in = collections.defaultdict(list)

    for stim, areas in areas_by_stim.items():
      if stim not in self.stimulus_size_by_name:
        raise IndexError(f"Not in brain.stimulus_size_by_name: {stim}")
      for area_name in areas:
        if area_name not in self.area_by_name:
          raise IndexError(f"Not in brain.area_by_name: {area_name}")
        stim_in[area_name].append(stim)
    for from_area_name, to_area_names in dst_areas_by_src_area.items():
      if from_area_name not in self.area_by_name:
        raise IndexError(from_area_name + " not in brain.area_by_name")
      for to_area_name in to_area_names:
        if to_area_name not in self.area_by_name:
          raise IndexError(f"Not in brain.area_by_name: {to_area_name}")
        area_in[to_area_name].append(from_area_name)

    to_update_area_names = stim_in.keys() | area_in.keys()

    for area_name in to_update_area_names:
      area = self.area_by_name[area_name]
      num_first_winners = self.project_into(
        area, stim_in[area_name], area_in[area_name], verbose)
      area.num_first_winners = num_first_winners
      if self.save_winners:
        area.saved_winners.append(area._new_winners)

    # once everything is done, for each area in to_update: area.update_winners()
    for area_name in to_update_area_names:
      area = self.area_by_name[area_name]
      area._update_winners()
      if self.save_size:
        area.saved_w.append(area.w)

  def project_with_image(self, image, areas_by_stim, dst_areas_by_src_area, input_area, verbose=0):
    """
    Projects raw image data into the neural network as the initial activation.

    Args:
        image (np.ndarray): Raw input image data.
        areas_by_stim (dict): Mapping from stimuli to target areas.
        dst_areas_by_src_area (dict): Mapping from source areas to target areas.
        input_area (str): The name of the input area for the image.
        verbose (int): Verbosity level.
    """
    # Activate the input area using the raw image data
    self.activate_with_image(input_area, image)

    # Project activations as usual
    self.project(areas_by_stim, dst_areas_by_src_area, verbose)

  def project_into(self, target_area, from_stimuli, from_areas, verbose=0):
    """Project stimuli and source areas into a single target area.

    This is the core computational step of the Assembly Calculus.  For each
    target area the method:

      1. **Accumulates synaptic input** from every firing stimulus and every
         source-area assembly into the target's existing neurons (indices
         ``0 .. target_area.w-1``).

      2. **Samples candidate new neurons** (sparse mode only).  Each of the
         ``k`` candidates receives a random input drawn from a truncated-
         normal approximation of ``Binomial(total_k, p)`` conditioned on
         being in the top-``(k / effective_n)`` quantile of the full
         ``n - w`` unexplored population.

      3. **Selects the top-k winners** (by ``heapq.nlargest``) from the
         union of existing and candidate neurons.

      4. **Applies Hebbian plasticity**:  ``w_ij *= (1 + β)`` for every
         synapse from a firing pre-synaptic neuron *j* to a winning
         post-synaptic neuron *i*, clamped at ``w_max``.

      5. **Expands connectomes** to accommodate first-time winners by
         adding rows/columns initialized with ``Bernoulli(p)`` random
         weights.

    Args:
      target_area: The Area object to project into.
      from_stimuli: List of stimulus names providing input this round.
      from_areas: List of source area names providing input this round.
      verbose: 0 = silent, 1 = summary, 2 = detailed arrays.

    Returns:
      int: Number of first-time winners (neurons that fired for the first
           time in this area during this projection step).

    References:
      Papadimitriou et al., PNAS 117(25):14464-14472, 2020 — Algorithm 1.
      Dabagia et al., "Coin-Flipping in the Brain", 2024 — weight ceiling.
    """
    rng = self._rng
    area_by_name = self.area_by_name
    target_area_name = target_area.name

    if verbose >= 1:
      print(f"Projecting {', '.join(from_stimuli)} "
            f" and {', '.join(from_areas)} into {target_area.name}")

    # If projecting from area with no assembly, complain.
    # Use extracted explicit projection engine for validation
    validate_source_winners(from_areas, self.connectomes, self.area_by_name, target_area_name)
    
    for from_area_name in from_areas:
      from_area = area_by_name[from_area_name]
      if from_area.winners.size == 0 or from_area.w == 0: # add not back in if not using numpy array
        raise ValueError(f"Projecting from area with no assembly: {from_area}")

    # For experiments with a "fixed" assembly in some area.
    if target_area.fixed_assembly:
      target_area_name = target_area.name
      target_area._new_winners = target_area.winners
      target_area._new_w = target_area.w
      first_winner_inputs = []
      num_first_winners_processed = 0

    else:
        # target_area_name = target_area.name

        if target_area.explicit: # Since target_area.w is now equal to target_area.n for explicit areas, the shapes will align when performing prev_winner_inputs += connectome[w].
            # Use extracted explicit projection engine for input accumulation
            prev_winner_inputs = accumulate_prev_inputs_explicit(
                target_area.n, 
                from_stimuli, 
                from_areas, 
                self.connectomes_by_stimulus, 
                self.connectomes, 
                self.area_by_name, 
                target_area_name
            )
        else:
            prev_winner_inputs = np.zeros(target_area.w, dtype=np.float32)

        if not target_area.explicit:
            for stim in from_stimuli:
              stim_inputs = self.connectomes_by_stimulus[stim][target_area_name]
              prev_winner_inputs += stim_inputs
            for from_area_name in from_areas:
              connectome = self.connectomes[from_area_name][target_area_name]
              for w in self.area_by_name[from_area_name].winners:
                prev_winner_inputs += connectome[w]

        if verbose >= 2:
          print("prev_winner_inputs:", prev_winner_inputs)

        # simulate area.k potential new winners if the area is not explicit
        if not target_area.explicit:
          input_size_by_from_area_index = []
          num_inputs = 0
          for stim in from_stimuli:
            local_k = self.stimulus_size_by_name[stim]
            input_size_by_from_area_index.append(local_k)
            num_inputs += 1
          for from_area_name in from_areas:
            effective_k = len(self.area_by_name[from_area_name].winners)
            input_size_by_from_area_index.append(effective_k)
            num_inputs += 1

          total_k = sum(input_size_by_from_area_index)
          if verbose >= 2:
            print(f"{total_k=} and {input_size_by_from_area_index=}")

          effective_n = target_area.n - target_area.w
          if effective_n <= target_area.k:
            raise RuntimeError(
                f'Remaining size of area "{target_area_name}" too small to sample k new winners.')
          # Threshold: inputs in the top-(k / effective_n) quantile.
          quantile = (effective_n - target_area.k) / effective_n
          alpha = binom.ppf(quantile, total_k, self.p)
          if verbose >= 2:
            print(f"Alpha = {alpha}")
          # Truncated-normal approximation to Binom(total_k, p) conditioned
          # on exceeding alpha.  Upper tail is unbounded (np.inf) rather than
          # total_k to better match the true binomial tail — values that
          # overshoot are clamped below.
          mu = total_k * self.p
          std = math.sqrt(total_k * self.p * (1.0 - self.p))
          a = (alpha - mu) / std
          potential_new_winner_inputs = (
              mu + truncnorm.rvs(a, np.inf, scale=std, size=target_area.k)
          ).round(0)
          np.clip(potential_new_winner_inputs, 0, total_k,
                  out=potential_new_winner_inputs)

          if verbose >= 2:
            print(f"potential_new_winner_inputs: {potential_new_winner_inputs}")

          # Combine existing and candidate inputs for winner selection.
          all_potential_winner_inputs = np.concatenate(
              [prev_winner_inputs, potential_new_winner_inputs])
        else:  # Case: Area is explicit.
          all_potential_winner_inputs = prev_winner_inputs

        new_winner_indices = heapq.nlargest(target_area.k,
                                            range(len(all_potential_winner_inputs)),
                                            all_potential_winner_inputs.__getitem__)
        if target_area.explicit:
            for winner in new_winner_indices:
                if not target_area.ever_fired[winner]:
                    target_area.ever_fired[winner] = True 
                    target_area.num_ever_fired += 1

        num_first_winners_processed = 0

        if not target_area.explicit:
          first_winner_inputs = []
          for i in range(target_area.k):
            if new_winner_indices[i] >= target_area.w:
              # Winner-index larger than `w` means that this winner was
              # first-activated here.
              first_winner_inputs.append(
                  all_potential_winner_inputs[new_winner_indices[i]])
              new_winner_indices[i] = target_area.w + num_first_winners_processed
              num_first_winners_processed += 1
        target_area._new_winners = new_winner_indices
        target_area._new_w = target_area.w + num_first_winners_processed

        if verbose >= 2:
          print(f"new_winners: {target_area._new_winners}")

        # Update connectomes for explicit areas
        if target_area.explicit:
            # Apply plasticity for explicit areas using extracted engine
            apply_area_to_area_plasticity(
                target_area._new_winners,
                from_areas,
                self.connectomes,
                self.area_by_name,
                target_area_name,
                self.disable_plasticity
            )

        # Distribute each first-time winner's total input across sources.
        # For each new winner, sample its total_input indices uniformly from
        # [0, total_k) and count how many fall in each source's range.
        inputs_by_first_winner_index = [None] * num_first_winners_processed
        for i in range(num_first_winners_processed):
          input_indices = rng.choice(range(total_k),
                                     int(first_winner_inputs[i]),
                                     replace=False)
          num_connections_by_input_index = np.zeros(num_inputs)
          total_so_far = 0
          for j in range(num_inputs):
            num_connections_by_input_index[j] = sum(
              total_so_far + input_size_by_from_area_index[j] > w >= total_so_far
              for w in input_indices)
            total_so_far += input_size_by_from_area_index[j]
          inputs_by_first_winner_index[i] = num_connections_by_input_index
          if verbose >= 2:
            print(f"For first_winner # {i} with input "
                  f"{first_winner_inputs[i]} split as so: "
                  f"{num_connections_by_input_index}")

    # --- Stimulus → target area connectome updates ---
    # Expand each firing-stimulus connectome by num_first_winners_processed,
    # assign sampled input counts, then apply Hebbian plasticity.
    num_inputs_processed = 0
    for stim in from_stimuli:
      connectomes = self.connectomes_by_stimulus[stim]
      if num_first_winners_processed > 0:
        connectomes[target_area_name] = target_connectome = np.resize(
            connectomes[target_area_name],
            target_area._new_w)
      else:
        target_connectome = connectomes[target_area_name]
      first_winner_synapses = target_connectome[target_area.w:]
      for i in range(num_first_winners_processed):
        first_winner_synapses[i] = (
            inputs_by_first_winner_index[i][num_inputs_processed])
      stim_to_area_beta = target_area.beta_by_stimulus[stim]
      if self.disable_plasticity:
        stim_to_area_beta = 0.0
      # Hebbian plasticity: w *= (1 + β), clamped at w_max to prevent
      # unbounded weight growth. See Dabagia et al. (2024) for saturation.
      for i in target_area._new_winners:
        target_connectome[i] *= 1 + stim_to_area_beta
      if self.w_max is not None:
        np.clip(target_connectome, 0, self.w_max, out=target_connectome)
      if verbose >= 2:
        print(f"{stim} now looks like: ")
        print(self.connectomes_by_stimulus[stim][target_area_name])
      num_inputs_processed += 1

    # update connectomes from stimuli that were not fired this round into the area.
    if (not target_area.explicit) and (num_first_winners_processed > 0):
        for stim_name, connectomes in self.connectomes_by_stimulus.items():
            if stim_name in from_stimuli:
                continue
            connectomes[target_area_name] = the_connectome = np.resize(
                connectomes[target_area_name],
                target_area._new_w)
            the_connectome[target_area.w:] = rng.binomial(
                self.stimulus_size_by_name[stim_name], self.p,
                size=(num_first_winners_processed))

    # --- Source area → target area connectome updates ---
    # Expand each source-area connectome by num_first_winners_processed columns,
    # assign sampled synapses for new winners, then apply Hebbian plasticity.
    for from_area_name in from_areas:
      from_area_w = self.area_by_name[from_area_name].w
      from_area_winners = self.area_by_name[from_area_name].winners
      from_area_winners_set = set(from_area_winners)
      from_area_connectomes = self.connectomes[from_area_name]
      the_connectome = from_area_connectomes[target_area_name] = np.pad(
          from_area_connectomes[target_area_name],
          ((0, 0), (0, num_first_winners_processed)))
      for i in range(num_first_winners_processed):
        total_in = inputs_by_first_winner_index[i][num_inputs_processed]
        sample_indices = rng.choice(from_area_winners, int(total_in), replace=False)
        for j in sample_indices:
          the_connectome[j, target_area.w + i] = 1.0
        for j in range(from_area_w):
          if j not in from_area_winners_set:
            the_connectome[j, target_area.w + i] = rng.binomial(1, self.p)
      # Apply Hebbian plasticity for sparse areas (explicit handled by engine).
      # Saturate at w_max to prevent unbounded weight growth.
      if not target_area.explicit:
        area_to_area_beta = (
          0 if self.disable_plasticity
          else target_area.beta_by_area[from_area_name])
        for i in target_area._new_winners:
          for j in from_area_winners:
            the_connectome[j, i] *= 1.0 + area_to_area_beta
            if self.w_max is not None and the_connectome[j, i] > self.w_max:
              the_connectome[j, i] = self.w_max
      if verbose >= 2:
        print(f"Connectome of {from_area_name} to {target_area_name} is now:",
              the_connectome)
      num_inputs_processed += 1

    # Expand connectomes for non-firing areas and reverse connections.
    # When new neurons fire for the first time, their connections to/from
    # all other areas must be initialized with Bernoulli(p) random weights.
    for other_area_name, other_area in self.area_by_name.items():
      other_area_connectomes = self.connectomes[other_area_name]
      if other_area_name not in from_areas:
          # Pad the connectome to add columns for new winners
          the_other_area_connectome = other_area_connectomes[target_area_name] = np.pad(
              other_area_connectomes[target_area_name],
              ((0, 0), (0, num_first_winners_processed))
          )

          # Only perform the assignment if there are new winners
          num_new_winners = target_area._new_w - target_area.w
          if num_new_winners > 0:
              the_other_area_connectome[:, target_area.w:] = rng.binomial(
                  1, self.p,
                  size=(the_other_area_connectome.shape[0], num_new_winners)
              )

      # Add rows for new winners
      target_area_connectomes = self.connectomes[target_area_name]
      the_target_area_connectome = target_area_connectomes[other_area_name] = np.pad(
          target_area_connectomes[other_area_name],
          ((0, num_first_winners_processed), (0, 0))
      )

      # Only perform the assignment if there are new winners
      num_new_winners = target_area._new_w - target_area.w
      if num_new_winners > 0:
          the_target_area_connectome[target_area.w:, :] = rng.binomial(
              1, self.p,
              size=(num_new_winners, the_target_area_connectome.shape[1])
          )

      if verbose >= 2:
          print(f"Connectome of {target_area_name!r} to {other_area_name!r} "
                "is now:", self.connectomes[target_area_name][other_area_name])

    return num_first_winners_processed