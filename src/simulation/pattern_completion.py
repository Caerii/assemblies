"""
Pattern completion simulation module.

This module contains simulation functions for studying pattern completion,
including basic pattern completion and parameter sweeps.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from core.brain import Brain
    import brain_util as bu
except ImportError:
    # Fallback for when running from root directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    import brain
    import brain_util as bu
    Brain = brain.Brain
import random
import copy

def pattern_com(n=100000, k=317, p=0.05, beta=0.05, project_iter=10, alpha=0.5, comp_iter=1):
    """
    Simulates pattern completion by randomly reactivating a subset of neurons in an assembly and projecting multiple times.

    Parameters:
    n (int): Number of neurons in the area.
    k (int): Number of active neurons in the stimulus.
    p (float): Probability parameter for the brain area.
    beta (float): Connectivity probability between neurons.
    project_iter (int): Number of initial projection iterations to establish the assembly.
    alpha (float): Fraction of assembly neurons randomly chosen to fire in the completion phase.
    comp_iter (int): Number of completion iterations to project the partial pattern.

    Returns:
    tuple: Contains two elements; the first is the final weights of the neural connections,
           and the second is the list of winners (active neurons) after completion.
    """
    b = Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter-1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    # pick random subset of the neurons to fire
    subsample_size = int(k*alpha)
    subsample = random.sample(list(b.areas["A"].winners), subsample_size)
    b.areas["A"].winners = subsample
    for i in range(comp_iter):
        b.project({}, {"A": ["A"]})
    return b.areas["A"].saved_w, b.areas["A"].saved_winners

def pattern_com_repeated(n=100000, k=317, p=0.05, beta=0.05, project_iter=12, alpha=0.4,
                         trials=3, max_recurrent_iter=10, resample=False):
    """
    Repeatedly simulates pattern completion with potentially different subsets of neurons firing each time,
    to assess the robustness of pattern reinstatement.

    Parameters:
    n (int): Number of neurons in the area.
    k (int): Number of active neurons in the stimulus.
    p (float): Probability parameter for the brain area.
    beta (float): Connectivity probability between neurons.
    project_iter (int): Number of initial projection iterations to establish the assembly.
    alpha (float): Fraction of assembly neurons randomly chosen to fire.
    trials (int): Number of times the pattern completion is attempted.
    max_recurrent_iter (int): Maximum number of projections during a single completion trial.
    resample (bool): If True, resample the subset of firing neurons for each trial.

    Returns:
    tuple: Contains overlaps of winners and the number of iterations to reach completion for each trial.
    """
    b = Brain(p, save_winners=True)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter-1):
        b.project({"stim": ["A"]}, {"A": ["A"]})

    subsample_size = int(k*alpha)
    rounds_to_completion = []
    # pick random subset of the neurons to fire
    subsample = random.sample(list(b.areas["A"].winners), subsample_size)
    for trail in range(trials):
        if resample:
            subsample = random.sample(list(b.areas["A"].winners), subsample_size)
        b.areas["A"].winners = subsample
        rounds = 0
        while True:
            rounds += 1
            b.project({}, {"A": ["A"]})
            if (b.areas["A"].num_first_winners == 0) or (rounds == max_recurrent_iter):
                break
        rounds_to_completion.append(rounds)
    saved_winners = b.areas["A"].saved_winners
    overlaps = bu.get_overlaps(saved_winners, project_iter-1, percentage=True)
    return overlaps, rounds_to_completion

def pattern_com_alphas(n=100000, k=317, p=0.01, beta=0.05,
                       alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                       project_iter=25, comp_iter=5):
    """
    Simulates pattern completion with varying percentages of active neurons in an established neural assembly.

    Parameters:
    n (int): Total number of neurons in the area.
    k (int): Number of neurons initially activated by the stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural area.
    alphas (list of float): List of proportions of the assembly to reactivate in each trial.
    project_iter (int): Number of iterations to establish the initial neural assembly.
    comp_iter (int): Number of iterations for each completion attempt.

    Returns:
    dict: Dictionary where keys are the alpha values and values are the overlap ratios of reactivated assembly with the initial winners.
    """
    b = Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter-1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    results = {}
    A_winners = b.areas["A"].winners
    for alpha in alphas:
        # pick random subset of the neurons to fire
        subsample_size = int(k*alpha)
        b_copy = copy.deepcopy(b)
        subsample = random.sample(b_copy.areas["A"].winners, subsample_size)
        b_copy.areas["A"].winners = subsample
        for i in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        final_winners = b_copy.areas["A"].winners
        o = bu.overlap(final_winners, A_winners)
        results[alpha] = float(o)/float(k)
    return results

def pattern_com_iterations(n=100000, k=317, p=0.01, beta=0.05, alpha=0.4, comp_iter=8,
                           min_iter=20, max_iter=30):
    """
    Investigates the effect of varying the number of projection iterations on the stability of pattern completion.

    Parameters:
    n (int): Total number of neurons in the neural area.
    k (int): Number of neurons initially activated by the stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural area.
    alpha (float): Fraction of the assembly to reactivate.
    comp_iter (int): Number of iterations for completion projections.
    min_iter (int): Minimum number of projection iterations to perform.
    max_iter (int): Maximum number of projection iterations to perform.

    Returns:
    dict: Dictionary where keys are the number of iterations, and values are the overlap ratios of reactivated assembly with the initial winners.
    """
    b = Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(min_iter-2):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    results = {}
    subsample_size = int(k*alpha)
    subsample = random.sample(list(b.areas["A"].winners), subsample_size)
    for i in range(min_iter, max_iter+1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        b_copy = copy.deepcopy(b)
        b_copy.areas["A"].winners = subsample
        for j in range(comp_iter):
            b_copy.project({}, {"A": ["A"]})
        o = bu.overlap(b_copy.areas["A"].winners, b.areas["A"].winners)
        results[i] = float(o)/float(k)
    return results
