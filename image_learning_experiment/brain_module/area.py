import numpy as np

class Area:
  """A brain area.

  Attributes:
    name: the area's name (symbolic tag).
    n: number of neurons in the area.
    k: number of neurons that fire in this area.
    beta: Default value for activation-`beta`.
    beta_by_stimulus: Mapping from stimulus-name to corresponding beta.
      (In original code: `.stimulus_beta`).
    beta_by_stimulus: Mapping from area-name to corresponding beta.
      (In original code: `.area_beta`).
    w: Number of neurons that has ever fired in this area.
    saved_w: List of per-round size-of-support.
    winners: List of winners, as set by previous action.
    saved_winners: List of lists of all winners, per-round.
    num_first_winners: ??? TODO(tfish): Clarify.
    fixed_assembly: Whether the assembly (of winners) in this area
      is considered frozen.
    explicit: Whether to fully simulate this area (rather than performing
      a sparse-only simulation).
  """
  def __init__(self, name, n, k, *,
               beta=0.05, w=0, explicit=False):
    """Initializes the instance.

    Args:
      name: Area name (symbolic tag), must be unique.
      n: number of neurons(?)
      k: number of firing neurons when activated.
      beta: default activation-beta.
      w: initial 'winner' set-size.
      explicit: boolean indicating whether the area is 'explicit'
        (fully-simulated).
    """
    self.name = name
    self.n = n
    self.k = k
    self.beta = beta
    self.beta_by_stimulus = {}
    self.beta_by_area = {}
    self.w = w
    # Value of `w` since the last time that `.project()` was called.
    self._new_w = 0
    self.saved_w = []
    # self.winners = []
    self.winners = np.array([], dtype=np.uint32)  # Initialize as empty NumPy array
    # Value of `winners` since the last time that `.project()` was called.
    # only to be used inside `.project()` method.
    self._new_winners = []
    self.saved_winners = []
    self.num_first_winners = -1
    self.fixed_assembly = False
    self.explicit = explicit

    # FOR DEBUGGING
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
    """
    Freezes the current assembly, preventing it from changing in future simulations.

    Raises:
      ValueError: if no assembly exists in this area.
    """
    print(f"[DEBUG] In fix_assembly: winners size: {self.winners.size}")
    print(f"[DEBUG] winners: {self.winners}")
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