"""
Advanced simulation module.

This module contains advanced simulation functions that were in the original
simulations.py but not yet modularized, including fixed assembly simulations
and specialized experimental setups.
"""

try:
    from src.core.brain import Brain
    import brain_util as bu
except ImportError:
    import brain
    import brain_util as bu
    Brain = brain.Brain

def fixed_assembly_recip_proj(n=100000, k=317, p=0.01, beta=0.05):
    """
    Simulates reciprocal projections between two neural assemblies A and B to test the restoration capabilities of the neural network.
    
    Parameters:
    n (int): Number of neurons in each area.
    k (int): Number of initial active neurons (winners) in the stimulation pattern.
    p (float): Initial probability of connectivity.
    beta (float): Synaptic modification rate.
    
    Description:
    Initiates a brain model, creates two areas A and B, and applies stimulation to A. After stabilizing the response in A, it projects responses to B and then reciprocally from B to A to test if the original pattern in A can be restored. The results show how effectively the assembly in B can influence A after stabilization.
    """
    b = Brain(p, save_winners=True, engine="numpy_sparse")
    b.add_stimulus("stimA", k)
    b.add_area("A", n, k, beta)
    # Will project fixes A into B
    b.add_area("B", n, k, beta)
    b.project({"stimA": ["A"]}, {})
    print("A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({"stimA": ["A"]}, {"A": ["A"]})
        print("A.w=" + str(b.areas["A"].w))
    # Freeze assembly in A and start projecting A <-> B
    b.areas["A"].fix_assembly()
    b.project({}, {"A": ["B"]})
    for i in range(20):
        b.project({}, {"A": ["B"], "B": ["A", "B"]})
        print("B.w=" + str(b.areas["B"].w))
    # If B has stabilized, this implies that the A->B direction is stable.
    # Therefore to test that this "worked" we should check that B->A restores A
    print("Before B->A, A.w=" + str(b.areas["A"].w))
    b.areas["A"].unfix_assembly()
    b.project({}, {"B": ["A"]})
    print("After B->A, A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({}, {"B": ["A"], "A": ["A"]})
        print("A.w=" + str(b.areas["A"].w))
    overlaps = bu.get_overlaps(b.areas["A"].saved_winners[-22:], 0, percentage=True)
    print(overlaps)

def fixed_assembly_merge(n=100000, k=317, p=0.01, beta=0.05):
    """
    Prepares a setup for merging neural activities between multiple assemblies using fixed patterns to observe the effects of merging in a controlled environment.

    Parameters:
    n (int): Number of neurons in each neural area.
    k (int): Number of initial active neurons in the stimulus.
    p (float): Initial probability of connectivity.
    beta (float): Synaptic modification rate.

    Description:
    Creates a neural setup with three areas, A, B, and C, and provides stimuli to A and B. The method then fixes the assemblies in A and B to observe their influence without further modifications.
    """
    b = Brain(p, engine="numpy_sparse")
    b.add_stimulus("stimA", k)
    b.add_stimulus("stimB", k)
    b.add_area("A", n, k, beta)
    b.add_area("B", n, k, beta)
    b.add_area("C", n, k, beta)
    b.project({"stimA": ["A"], "stimB": ["B"]}, {})
    for i in range(20):
        b.project({"stimA": ["A"], "stimB": ["B"]},
                  {"A": ["A"], "B": ["B"]})
    b.areas["A"].fix_assembly()
    b.areas["B"].fix_assembly()

def separate(n=10000, k=100, p=0.01, beta=0.05, rounds=10, overlap=0):
    """
    Simulates the effect of projecting separate stimuli into a neural area to examine the interaction and overlap between the resulting neural assemblies.

    Parameters:
    n (int): Total number of neurons in area A.
    k (int): Number of active neurons in each stimulus.
    p (float): Initial probability of connectivity.
    beta (float): Synaptic modification rate.
    rounds (int): Number of projection rounds for each stimulus.
    overlap (int): Number of overlapping neurons between the two stimuli.

    Description:
    Introduces two distinct stimuli into a neural area and measures the overlap between the resulting assemblies after a series of projections. It also tests the ability to restore initial assemblies after multiple rounds of stimulation.
    """
    b = Brain(p, engine="numpy_sparse")
    b.add_explicit_area("EXP", 2*k, k, beta)
    b.add_area("A", n, k, beta)

    b.areas["EXP"].winners = list(range(0, k))
    b.areas["EXP"].fix_assembly()

    print("PROJECTION STIM_1 INTO A....")
    b.project({}, {"EXP": ["A"]})
    prev_w = k
    print(prev_w)
    for i in range(rounds):
        b.project({}, {"EXP": ["A"], "A": ["A"]})
        new_w = b.areas["A"].w - prev_w
        print(new_w)
        prev_w = b.areas["A"].w
    stim1_assembly = b.areas["A"].winners

    print("PROJECTION STIM_2 INTO A....")

    b.areas["EXP"].winners = list(range(k-overlap, 2*k-overlap))
    b.areas["EXP"].fix_assembly()

    b.project({}, {"EXP": ["A"]})
    new_w = b.areas["A"].w - prev_w
    print(new_w)
    prev_w = b.areas["A"].w

    for i in range(rounds):
        b.project({}, {"EXP": ["A"], "A": ["A"]})
        new_w = b.areas["A"].w - prev_w
        print(new_w)
        prev_w = b.areas["A"].w
    stim2_assembly = b.areas["A"].winners
    o = bu.overlap(stim1_assembly, stim2_assembly)
    print("Got overlap of " + str(o) + " / " + str(k))
    
    b.no_plasticity = True
    b.areas["EXP"].winners = list(range(0, k))
    b.areas["EXP"].fix_assembly()
    b.project({}, {"EXP": ["A"]})
    o = bu.overlap(b.areas["A"].winners, stim1_assembly)
    print("Restored " + str(o) + " / " + str(k) + " of Assembly 1")

    b.areas["EXP"].winners = list(range(k-overlap, 2*k-overlap))
    b.project({}, {"EXP": ["A"]})
    o = bu.overlap(b.areas["A"].winners, stim2_assembly)
    print("Restored " + str(o) + " / " + str(k) + " of Assembly 2")
