"""
Merge simulation module.

This module contains simulation functions for studying neural merging,
including basic merge simulations and beta parameter sweeps.
"""

try:
    from src.core.brain import Brain
except ImportError:
    import brain
    Brain = brain.Brain

def merge_sim(n=100000, k=317, p=0.01, beta=0.05, max_t=50):
    """
    Simulates the merging of neural activities from two different stimuli into a shared and separate neural areas over multiple time steps.

    Parameters:
    n (int): Number of neurons in each neural area.
    k (int): Number of active neurons initially stimulated by each stimulus.
    p (float): Base probability parameter for the brain's setup.
    beta (float): Probability of connectivity within each neural area.
    max_t (int): Maximum number of time steps to run the merge simulation.

    Returns:
    tuple: Saved weights of neural areas 'A', 'B', and 'C' after merging simulation.
    """
    b = Brain(p)
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)

    b.project({"stimA": ["A"]}, {})
    b.project({"stimB": ["B"]}, {})
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    b.project({}, {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]})
    for i in range(max_t-1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]})
    return b.areas["A"].saved_w, b.areas["B"].saved_w, b.areas["C"].saved_w

def merge_beta_sim(n=100000, k=317, p=0.01, t=100):
    """
    Executes the merge simulation across a range of beta values to evaluate the effects of varying connectivity probabilities on neural merging.

    Parameters:
    n (int): Number of neurons in each neural area.
    k (int): Number of active neurons initially stimulated by each stimulus.
    p (float): Base probability parameter for the brain's setup.
    t (int): Number of time steps each simulation will run for.

    Returns:
    dict: A dictionary where keys are beta values and values are the results of the merge simulation for those beta values.
    """
    results = {}
    for beta in [0.3, 0.2, 0.1, 0.075, 0.05]:
        print("Working on " + str(beta) + "\n")
        out = merge_sim(n, k, p, beta=beta, max_t=t)
        results[beta] = out
    return results
