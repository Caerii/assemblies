"""
Density simulation module.

This module contains simulation functions for studying neural density,
including basic density simulations and parameter sweeps.
"""

try:
    from src.core.brain import Brain
except ImportError:
    import brain
    Brain = brain.Brain

def density(n=100000, k=317, p=0.01, beta=0.05, rounds=20):
    """
    Simulates and calculates the density of connections within a neural network after several rounds of projections.

    Parameters:
    n (int): Number of neurons in the network.
    k (int): Number of connections per neuron.
    p (float): Initial probability of a connection.
    beta (float): Learning rate or adjustment factor in the neural update rules.
    rounds (int): Number of simulation rounds to execute.

    Returns:
    tuple: A tuple containing the density of connections within the final winner assembly, and a list of synaptic strengths over all rounds.

    Description:
    Executes a series of projections to develop neural assemblies, then calculates the density of connections among the final set of active neurons.
    """
    b = Brain(p, engine="numpy_sparse")
    b.add_stimulus("stim", k)
    b.add_area("A", n, k, beta)
    b.project({"stim": ["A"]}, {})
    saved_w = []
    for i in range(rounds):
        b.project({"stim": ["A"]}, {"A": ["A"]})
        saved_w.append(b.areas["A"].w)
    conn = b.connectomes["A"]["A"]
    final_winners = b.areas["A"].winners
    edges = 0
    for i in final_winners:
        for j in final_winners:
            if conn[i][j] != 0:
                edges += 1
    return float(edges)/float(k**2), saved_w

def density_sim(n=100000, k=317, p=0.01, beta_values=[0, 0.025, 0.05, 0.075, 0.1]):
    """
    Runs a series of density simulations over a range of beta values to determine how learning rates affect connectivity.

    Parameters:
    n (int): Number of neurons.
    k (int): Number of connections per neuron.
    p (float): Probability of initial connections.
    beta_values (list): List of beta values to simulate.

    Returns:
    dict: A dictionary with beta values as keys and density simulation results as values.

    Description:
    This function iterates over a list of beta values, runs density calculations for each, and collects the results for analysis.
    """
    results = {}
    for beta in beta_values:
        print("Working on " + str(beta) + "\n")
        out = density(n, k, p, beta)
        results[beta] = out
    return results
