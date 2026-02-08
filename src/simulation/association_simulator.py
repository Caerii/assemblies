"""
Association simulation module.

This module contains simulation functions for studying neural associations,
including basic association simulations and parameter sweeps.
"""

import os
try:
    from src.core.brain import Brain
except ImportError:
    import brain
    Brain = brain.Brain

import copy

def associate(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    """
    Simulates the association of two neural stimuli into a third neural area through sequential and concurrent projections.

    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    overlap_iter (int): Number of projection iterations where both stimuli influence the third area.

    Returns:
    brain.Brain: The brain object after executing the association simulation.
    """
    b = Brain(p, save_winners=True, engine="numpy_sparse")
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    VERBOSE = os.environ.get("ASSEMBLIES_VERBOSE") == "1"
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    # Create assemblies A and B to stability
    for i in range(9):
        if VERBOSE:
            print(f"Stabilizing A/B {i+1}/9")
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]}, verbose=1 if VERBOSE else 0)
    b.project({"stimA": ["A"]}, {"A": ["A", "C"]}, verbose=1 if VERBOSE else 0)
    # Project A->C
    for i in range(9):
        if VERBOSE:
            print(f"A->C {i+1}/10")
        b.project({"stimA": ["A"]},
                  {"A": ["A", "C"], "C": ["C"]}, verbose=1 if VERBOSE else 0)
    # Project B->C
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        if VERBOSE:
            print(f"B->C {i+1}/10")
        b.project({"stimB": ["B"]},
                  {"B": ["B", "C"], "C": ["C"]}, verbose=1 if VERBOSE else 0)
    # Project both A,B to C
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]}, verbose=1 if VERBOSE else 0)
    for i in range(overlap_iter-1):
        if VERBOSE:
            print(f"A,B->C overlap {i+2}/{overlap_iter}")
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]}, verbose=1 if VERBOSE else 0)
    # Project just B
        b.project({"stimB": ["B"]}, {"B": ["B", "C"]}, verbose=1 if VERBOSE else 0)
    for i in range(9):
        if VERBOSE:
            print(f"Final B-only {i+1}/10")
        b.project({"stimB": ["B"]}, {"B": ["B", "C"], "C": ["C"]}, verbose=1 if VERBOSE else 0)
    return b

def association_sim(n=100000, k=317, p=0.05, beta=0.1, overlap_iter=10):
    """
    Wrapper function to execute an association simulation and retrieve the saved weights and winner neurons of area 'C'.

    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    overlap_iter (int): Number of projection iterations where both stimuli influence the third area.

    Returns:
    tuple: Tuple containing the saved weights and winner neurons from area 'C' after the simulation.
    """
    b = associate(n, k, p, beta, overlap_iter)
    return b.areas["C"].saved_w, b.areas["C"].saved_winners

def association_grand_sim(n=100000, k=317, p=0.01, beta=0.05, min_iter=10, max_iter=20):
    """
    Conducts a comprehensive association simulation with varying numbers of iterations to examine the effects on the stability and overlap of assemblies in a third neural area.

    Parameters:
    n (int): Total number of neurons in each neural area.
    k (int): Number of neurons initially activated by each stimulus.
    p (float): Probability parameter for the brain setup.
    beta (float): Connectivity probability within the neural areas.
    min_iter (int): Minimum number of iterations for associative projections.
    max_iter (int): Maximum number of iterations for associative projections.

    Returns:
    dict: Dictionary where keys are the number of iterations and values are the overlap ratios of assemblies in the third area as influenced by two distinct stimuli.
    """
    b = Brain(p, save_winners=True, engine="numpy_sparse")
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    b.add_stimulus("stimB", k)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    # Create assemblies A and B to stability
    for i in range(9):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]})
    b.project({"stimA": ["A"]}, {"A": ["A", "C"]})
    # Project A->C
    for i in range(9):
        b.project({"stimA": ["A"]},
                  {"A": ["A", "C"], "C": ["C"]})
    # Project B->C
    b.project({"stimB": ["B"]}, {"B": ["B", "C"]})
    for i in range(9):
        b.project({"stimB": ["B"]},
                  {"B": ["B", "C"], "C": ["C"]})
    # Project both A,B to C
    b.project({"stimA": ["A"], "stimB": ["B"]},
              {"A": ["A", "C"], "B": ["B", "C"]})
    for i in range(min_iter-2):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
    results = {}
    for i in range(min_iter, max_iter+1):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]})
        b_copy1 = copy.deepcopy(b)
        b_copy2 = copy.deepcopy(b)
        # in copy 1, project just A
        b_copy1.project({"stimA": ["A"]}, {})
        b_copy1.project({}, {"A": ["C"]})
        # in copy 2, project just B
        b_copy2.project({"stimB": ["B"]}, {})
        b_copy2.project({}, {"B": ["C"]})
        o = overlap(b_copy1.areas["C"].winners, b_copy2.areas["C"].winners)
        results[i] = float(o)/float(k)
    return results

def overlap(assembly1, assembly2):
    """Calculate overlap between two assemblies."""
    return len(set(assembly1) & set(assembly2))
