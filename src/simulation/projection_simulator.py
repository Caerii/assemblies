"""
Projection simulation module.

This module contains simulation functions for studying neural projections,
including basic projection simulations and beta parameter sweeps.
"""

try:
    from src.core.brain import Brain
except ImportError:
    import brain
    Brain = brain.Brain

def project_sim(n=1000000, k=1000, p=0.01, beta=0.05, t=50):
    """
    Simulates the projection of stimuli into an area over a number of iterations.

    Parameters:
    n (int): The number of neurons in the area.
    k (int): The number of neurons in the stimulus.
    p (float): The probability associated with the brain area setup.
    beta (float): The connectivity probability between neurons.
    t (int): The number of time steps to simulate.

    Returns:
    list: A list of synaptic weights saved during the projection.
    """
    b = Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(t-1):
        b.project({}, {"A": ["A"]})
    return b.areas["A"].saved_w

def project_beta_sim(n=100000, k=317, p=0.01, t=100):
    """
    Simulates projections for various beta values to study their effects on synaptic weights.

    Parameters:
    n (int): The number of neurons in the area.
    k (int): The number of neurons in the stimulus.
    p (float): The probability parameter for the brain area.
    t (int): The number of time steps for each simulation.

    Returns:
    dict: A dictionary where keys are beta values and values are the resulting synaptic weights.
    """
    results = {}
    for beta in [0.25, 0.1, 0.075, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001]:
        print("Working on " + str(beta) + "\n")
        out = project_sim(n, k, p, beta, t)
        results[beta] = out
    return results

def assembly_only_sim(n=100000, k=317, p=0.05, beta=0.05, project_iter=10):
    """
    Simulates the stabilization of an assembly within a neural area through recurrent projections.

    Parameters:
    n (int): Number of neurons in the area.
    k (int): Number of active neurons in the stimulus.
    p (float): Probability associated with initial brain setup.
    beta (float): Connectivity probability between neurons.
    project_iter (int): Number of projection iterations to run before stabilization.

    Returns:
    list: List of synaptic weights of the assembly after simulation.
    """
    b = Brain(p)
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    for i in range(project_iter-1):
        b.project({"stim": ["A"]}, {"A": ["A"]})
    for i in range(5):
        b.project({}, {"A": ["A"]})
    return b.areas["A"].saved_w
