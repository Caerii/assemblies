# Default simulation library containing:
# - Basic [project]ion simulations (convergence for different beta, etc)
# - Merge simulations (different betas)
# - Pattern completion simulations
# - Association simulations
# - simulations studying density in assemblies (higher than ambient p)

# Also contains methods for plotting saved results from some of these simulations
# (for figures).

import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict

def project_sim(n=1000000,k=1000,p=0.01,beta=0.05,t=50):
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
    b = brain.Brain(p)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(t-1):
        b.project({"stim":["A"]},{"A":["A"]})
    return b.areas["A"].saved_w

def project_beta_sim(n=100000,k=317,p=0.01,t=100):
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
    for beta in [0.25,0.1,0.075,0.05,0.03,0.01,0.007,0.005,0.003,0.001]:
        print("Working on " + str(beta) + "\n")
        out = project_sim(n,k,p,beta,t)
        results[beta] = out
    return results

def assembly_only_sim(n=100000,k=317,p=0.05,beta=0.05,project_iter=10):
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
    b = brain.Brain(p)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(project_iter-1):
        b.project({"stim":["A"]},{"A":["A"]})
    for i in range(5):
        b.project({},{"A":["A"]})
    return b.areas["A"].saved_w

# alpha = percentage of (random) final assembly neurons to try firing
def pattern_com(n=100000,k=317,p=0.05,beta=0.05,project_iter=10,alpha=0.5,comp_iter=1):
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
    b = brain.Brain(p,save_winners=True)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(project_iter-1):
        b.project({"stim":["A"]},{"A":["A"]})
    # pick random subset of the neurons to fire
    subsample_size = int(k*alpha)
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    b.areas["A"].winners = subsample
    for i in range(comp_iter):
        b.project({},{"A":["A"]})
    return b.areas["A"].saved_w,b.areas["A"].saved_winners

def pattern_com_repeated(n=100000,k=317,p=0.05,beta=0.05,project_iter=12,alpha=0.4,
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
    b = brain.Brain(p,save_winners=True)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(project_iter-1):
        b.project({"stim":["A"]},{"A":["A"]})

    subsample_size = int(k*alpha)
    rounds_to_completion = []
    # pick random subset of the neurons to fire
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    for trail in range(trials):
        if resample:
            subsample = random.sample(b.areas["A"].winners, subsample_size)
        b.areas["A"].winners = subsample
        rounds = 0
        while True:
            rounds += 1
            b.project({},{"A":["A"]})
            if (b.areas["A"].num_first_winners == 0) or (rounds == max_recurrent_iter):
                break
        rounds_to_completion.append(rounds)
    saved_winners = b.areas["A"].saved_winners
    overlaps = bu.get_overlaps(saved_winners,project_iter-1,percentage=True)
    return overlaps, rounds_to_completion

def pattern_com_alphas(n=100000,k=317,p=0.01,beta=0.05,
    alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],project_iter=25,comp_iter=5):
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
    b = brain.Brain(p)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(project_iter-1):
        b.project({"stim":["A"]},{"A":["A"]})
    results = {}
    A_winners = b.areas["A"].winners
    for alpha in alphas:
        # pick random subset of the neurons to fire
        subsample_size = int(k*alpha)
        b_copy = copy.deepcopy(b)
        subsample = random.sample(b_copy.areas["A"].winners, subsample_size)
        b_copy.areas["A"].winners = subsample
        for i in range(comp_iter):
            b_copy.project({},{"A":["A"]})
        final_winners = b_copy.areas["A"].winners
        o = bu.overlap(final_winners, A_winners)
        results[alpha] = float(o)/float(k)
    return results

def pattern_com_iterations(n=100000,k=317,p=0.01,beta=0.05,alpha=0.4,comp_iter=8,
    min_iter=20,max_iter=30):
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
    b = brain.Brain(p)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    for i in range(min_iter-2):
        b.project({"stim":["A"]},{"A":["A"]})
    results = {}
    subsample_size = int(k*alpha)
    subsample = random.sample(b.areas["A"].winners, subsample_size)
    for i in range(min_iter,max_iter+1):
        b.project({"stim":["A"]},{"A":["A"]})
        b_copy = copy.deepcopy(b)
        b_copy.areas["A"].winners = subsample
        for j in range(comp_iter):
            b_copy.project({},{"A":["A"]})
        o = bu.overlap(b_copy.areas["A"].winners, b.areas["A"].winners)
        results[i] = float(o)/float(k)
    return results

# Sample command c_w,c_winners = bu.association_sim()
def associate(n=100000,k=317,p=0.05,beta=0.1,overlap_iter=10):
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
    b = brain.Brain(p,save_winners=True)
    b.add_stimulus("stimA",k)
    b.add_area("A",n,k,beta)
    b.add_stimulus("stimB",k)
    b.add_area("B",n,k,beta)
    b.add_area("C",n,k,beta)
    b.project({"stimA":["A"],"stimB":["B"]},{})
    # Create assemblies A and B to stability
    for i in range(9):
        b.project({"stimA":["A"],"stimB":["B"]},
            {"A":["A"],"B":["B"]})
    b.project({"stimA":["A"]},{"A":["A","C"]})
    # Project A->C
    for i in range(9):
        b.project({"stimA":["A"]},
            {"A":["A","C"],"C":["C"]})
    # Project B->C
    b.project({"stimB":["B"]},{"B":["B","C"]})
    for i in range(9):
        b.project({"stimB":["B"]},
            {"B":["B","C"],"C":["C"]})
    # Project both A,B to C
    b.project({"stimA":["A"],"stimB":["B"]},
        {"A":["A","C"],"B":["B","C"]})
    for i in range(overlap_iter-1):
        b.project({"stimA":["A"],"stimB":["B"]},
                {"A":["A","C"],"B":["B","C"],"C":["C"]})
    # Project just B
    b.project({"stimB":["B"]},{"B":["B","C"]})
    for i in range(9):
        b.project({"stimB":["B"]},{"B":["B","C"],"C":["C"]})
    return b

def association_sim(n=100000,k=317,p=0.05,beta=0.1,overlap_iter=10):
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
    b = associate(n,k,p,beta,overlap_iter)
    return b.areas["C"].saved_w,b.areas["C"].saved_winners

def association_grand_sim(n=100000,k=317,p=0.01,beta=0.05,min_iter=10,max_iter=20):
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
    b = brain.Brain(p,save_winners=True)
    b.add_stimulus("stimA",k)
    b.add_area("A",n,k,beta)
    b.add_stimulus("stimB",k)
    b.add_area("B",n,k,beta)
    b.add_area("C",n,k,beta)
    b.project({"stimA":["A"],"stimB":["B"]},{})
    # Create assemblies A and B to stability
    for i in range(9):
        b.project({"stimA":["A"],"stimB":["B"]},
            {"A":["A"],"B":["B"]})
    b.project({"stimA":["A"]},{"A":["A","C"]})
    # Project A->C
    for i in range(9):
        b.project({"stimA":["A"]},
            {"A":["A","C"],"C":["C"]})
    # Project B->C
    b.project({"stimB":["B"]},{"B":["B","C"]})
    for i in range(9):
        b.project({"stimB":["B"]},
            {"B":["B","C"],"C":["C"]})
    # Project both A,B to C
    b.project({"stimA":["A"],"stimB":["B"]},
        {"A":["A","C"],"B":["B","C"]})
    for i in range(min_iter-2):
        b.project({"stimA":["A"],"stimB":["B"]},
                {"A":["A","C"],"B":["B","C"],"C":["C"]})
    results = {}
    for i in range(min_iter,max_iter+1):
        b.project({"stimA":["A"],"stimB":["B"]},
                {"A":["A","C"],"B":["B","C"],"C":["C"]})
        b_copy1 = copy.deepcopy(b)
        b_copy2 = copy.deepcopy(b)
        # in copy 1, project just A
        b_copy1.project({"stimA":["A"]},{})
        b_copy1.project({},{"A":["C"]})
        # in copy 2, project just B
        b_copy2.project({"stimB":["B"]},{})
        b_copy2.project({},{"B":["C"]})
        o = bu.overlap(b_copy1.areas["C"].winners, b_copy2.areas["C"].winners)
        results[i] = float(o)/float(k)
    return results

def merge_sim(n=100000,k=317,p=0.01,beta=0.05,max_t=50):
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
    b = brain.Brain(p)
    b.add_stimulus("stimA",k)
    b.add_stimulus("stimB",k)
    b.add_area("A",n,k,beta)
    b.add_area("B",n,k,beta)
    b.add_area("C",n,k,beta)

    b.project({"stimA":["A"]},{})
    b.project({"stimB":["B"]},{})
    b.project({"stimA":["A"],"stimB":["B"]},
        {"A":["A","C"],"B":["B","C"]})
    b.project({"stimA":["A"],"stimB":["B"]},
        {"A":["A","C"],"B":["B","C"],"C":["C","A","B"]})
    for i in range(max_t-1):
        b.project({"stimA":["A"],"stimB":["B"]},
            {"A":["A","C"],"B":["B","C"],"C":["C","A","B"]})
    return b.areas["A"].saved_w, b.areas["B"].saved_w, b.areas["C"].saved_w

def merge_beta_sim(n=100000,k=317,p=0.01,t=100):
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
    for beta in [0.3,0.2,0.1,0.075,0.05]:
        print("Working on " + str(beta) + "\n")
        out = merge_sim(n,k,p,beta=beta,max_t=t)
        results[beta] = out
    return results
# UTILS FOR EVAL

def plot_project_sim(show=True, save="", show_legend=False, use_text_font=True):
    """
    Plots the results of a projection simulation, displaying the development of neural weights over time with the option to customize the display.

    Parameters:
    show (bool): If True, display the plot; otherwise, do not show.
    save (str): Path to save the plot file. If empty, the plot is not saved.
    show_legend (bool): If True, include a legend in the plot.
    use_text_font (bool): If True, use specific font settings for the plot.

    Notes:
    The function retrieves the simulation results using a utility method, formats them, and then plots using matplotlib.
    """
    results = bu.sim_load('project_results')
    # fonts
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    # 0.05 and 0.07 overlap almost exactly, pop 0.07
    results.pop(0.007)
    od = OrderedDict(sorted(results.items()))
    x = np.arange(100)
    print(x)
    for key,val in od.iteritems():
        plt.plot(x,val,linewidth=0.7)
    if show_legend:
        plt.legend(od.keys(), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0,10,20,50,100])
    k = 317
    plt.yticks([k,2*k,5*k,10*k,13*k],["k","2k","5k","10k","13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, od.keys()):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
                        xycoords = ax.get_yaxis_transform(), textcoords="offset points",
                        size=10, va="center")
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def plot_merge_sim(show=True, save="", show_legend=False, use_text_font=True):
    """
    Plots the results of merge simulations across different beta values, showing the neural activities over time.

    Parameters:
    show (bool): If True, displays the plot on the screen.
    save (str): Path to save the plot image file. If empty, the plot is not saved.
    show_legend (bool): If True, includes a legend in the plot.
    use_text_font (bool): If True, sets the font style to be suitable for mathematical notation.

    Notes:
    Retrieves the results from a utility function, then formats and displays them using matplotlib.
    """
    results = bu.sim_load('merge_betas')
    # fonts
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    x = np.arange(101)
    for key,val in od.iteritems():
        plt.plot(x,val,linewidth=0.7)
    if show_legend:
        plt.legend(od.keys(), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0,10,20,50,100])
    k = 317
    plt.yticks([k,2*k,5*k,10*k,13*k],["k","2k","5k","10k","13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, od.keys()):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
                        xycoords = ax.get_yaxis_transform(), textcoords="offset points",
                        size=10, va="center")
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def plot_association(show=True, save="", use_text_font=True):
    """
    Plots the results of neural association simulations, highlighting the degree of overlap between neural assemblies over time.

    Parameters:
    show (bool): If True, displays the plot on the screen.
    save (str): Path to save the plot image file. If empty, the plot is not saved.
    use_text_font (bool): If True, sets the font style to be suitable for mathematical notation.

    Notes:
    Retrieves the results from a utility function, then formats and displays them using matplotlib. This plot helps in understanding the association dynamics in neural simulations.
    """
    results = bu.sim_load('association_results')
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(od.keys(),od.values(),linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0.1,0.2,0.3,0.4,0.5],["10%","20%","30%","40%","50%"])
    plt.xlabel(r'$t$')
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def plot_pattern_com(show=True, save="", use_text_font=True):
    """
    Plots the results of pattern completion simulations, showing the effectiveness of pattern completion across different iterations.

    Parameters:
    show (bool): If True, displays the plot on the screen.
    save (str): Path to save the plot image file. If empty, the plot is not saved.
    use_text_font (bool): If True, sets the font style to be suitable for mathematical notation.

    Notes:
    Retrieves the simulation data from a utility function and plots the degree of pattern completion, providing insights into the stability and recovery of neural patterns.
    """
    results = bu.sim_load('pattern_com_iterations')
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(od.keys(),od.values(),linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0,0.25,0.5,0.75,1],["0%","25%","50%","75%","100%"])
    plt.xlabel(r'$t$')
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def plot_overlap(show=True, save="", use_text_font=True):
    """
    Plots the overlap of neural assemblies and projections over simulation iterations, showcasing the relationship between assemblies.

    Parameters:
    show (bool): If True, the plot is displayed on the screen.
    save (str): Path where the plot image is saved. If empty, the plot is not saved.
    use_text_font (bool): If True, mathematical fonts are used for text in the plot.

    Description:
    This function loads simulation results, sets the plotting parameters for mathematical expressions, and visualizes the overlap between neural assemblies and projections.
    """
    results = bu.sim_load('overlap_results')
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

    od = OrderedDict(sorted(results.items()))
    plt.plot(od.keys(),od.values(),linewidth=0.7)
    ax = plt.axes()
    plt.xticks([0,0.2,0.4,0.6,0.8],["","20%","40%","60%","80%"])
    plt.xlabel('overlap (assemblies)')
    plt.yticks([0,0.05,0.1,0.15,0.2,0.25,0.3],["","5%","10%","15%","20%","25%","30%"])
    plt.ylabel('overlap (projections)')
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def density(n=100000,k=317,p=0.01,beta=0.05,rounds=20):
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
    b = brain.Brain(p)
    b.add_stimulus("stim",k)
    b.add_area("A",n,k,beta)
    b.project({"stim":["A"]},{})
    saved_w = []
    for i in range(rounds):
        b.project({"stim":["A"]},{"A":["A"]})
        saved_w.append(b.areas["A"].w)
    conn = b.connectomes["A"]["A"]
    final_winners = b.areas["A"].winners
    edges = 0
    for i in final_winners:
        for j in final_winners:
            if conn[i][j] != 0:
                edges += 1
    return float(edges)/float(k**2), saved_w

def density_sim(n=100000,k=317,p=0.01,beta_values=[0,0.025,0.05,0.075,0.1]):
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
        out = density(n,k,p,beta)
        results[beta] = out
    return results

def plot_density_ee(show=True,save="",use_text_font=True):
    """
    Plots the results of density simulations, showing the effective assembly connectivity probability as a function of the learning rate beta.

    Parameters:
    show (bool): If True, displays the plot on the screen.
    save (str): Path to save the plot image file. If empty, the plot is not saved.
    use_text_font (bool): If True, sets the font style to be suitable for mathematical notation.

    Description:
    Loads results from density simulations and plots the effective assembly connectivity probability against varying beta values.
    """
    if(use_text_font):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    od = bu.sim_load('density_results')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'assembly $p$')
    plt.plot(od.keys(),od.values(),linewidth=0.7)
    plt.plot([0,0.06],[0.01,0.01],color='red',linestyle='dashed',linewidth=0.7)
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

# For default values, first B->A gets only 25% of A's original assembly
# After subsequent recurrent firings restore up to 42% 
# With artificially high beta, can get 100% restoration.
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
    b = brain.Brain(p, save_winners=True)
    b.add_stimulus("stimA",k)
    b.add_area("A",n,k,beta)
    # Will project fixes A into B
    b.add_area("B",n,k,beta)
    b.project({"stimA":["A"]},{})
    print("A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({"stimA":["A"]}, {"A":["A"]})
        print("A.w=" + str(b.areas["A"].w))
    # Freeze assembly in A and start projecting A <-> B
    b.areas["A"].fix_assembly()
    b.project({}, {"A":["B"]})
    for i in range(20):
        b.project({}, {"A":["B"], "B":["A","B"]})
        print("B.w=" + str(b.areas["B"].w))
    # If B has stabilized, this implies that the A->B direction is stable.
    # Therefore to test that this "worked" we should check that B->A restores A
    print("Before B->A, A.w=" + str(b.areas["A"].w))
    b.areas["A"].unfix_assembly()
    b.project({},{"B":["A"]})
    print("After B->A, A.w=" + str(b.areas["A"].w))
    for i in range(20):
        b.project({}, {"B":["A"],"A":["A"]})
        print("A.w=" + str(b.areas["A"].w))
    overlaps = bu.get_overlaps(b.areas["A"].saved_winners[-22:],0,percentage=True)
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
    b = brain.Brain(p)
    b.add_stimulus("stimA",k)
    b.add_stimulus("stimB",k)
    b.add_area("A",n,k,beta)
    b.add_area("B",n,k,beta)
    b.add_area("C",n,k,beta)
    b.project({"stimA":["A"], "stimB":["B"]},{})
    for i in range(20):
        b.project({"stimA":["A"], "stimB":["B"]},
            {"A":["A"], "B":["B"]})
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
    b = brain.Brain(p)
    b.add_explicit_area("EXP", 2*k, k, beta)
    b.add_area("A",n,k,beta)

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