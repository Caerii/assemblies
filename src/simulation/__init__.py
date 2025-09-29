"""
Simulation engines and runners.

This module contains the core simulation engines for running neural
assembly simulations, including projection, association, and merge simulations.
"""

from .projection_simulator import project_sim, project_beta_sim, assembly_only_sim
from .association_simulator import associate, association_sim, association_grand_sim
from .merge_simulator import merge_sim, merge_beta_sim
from .pattern_completion import (pattern_com, pattern_com_repeated, 
                                pattern_com_alphas, pattern_com_iterations)
from .density_simulator import density, density_sim
from .advanced_simulations import (fixed_assembly_recip_proj, fixed_assembly_merge, separate)
from .turing_simulations import larger_k, turing_erase
from .plotting_utils import (plot_project_sim, plot_merge_sim, plot_association,
                            plot_pattern_com, plot_overlap, plot_density_ee)

__all__ = ['project_sim', 'project_beta_sim', 'assembly_only_sim',
           'associate', 'association_sim', 'association_grand_sim',
           'merge_sim', 'merge_beta_sim',
           'pattern_com', 'pattern_com_repeated', 'pattern_com_alphas', 'pattern_com_iterations',
           'density', 'density_sim',
           'fixed_assembly_recip_proj', 'fixed_assembly_merge', 'separate',
           'larger_k', 'turing_erase',
           'plot_project_sim', 'plot_merge_sim', 'plot_association',
           'plot_pattern_com', 'plot_overlap', 'plot_density_ee']
