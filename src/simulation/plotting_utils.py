"""
Plotting utilities for simulation results.

This module contains plotting functions for visualizing simulation results,
including projection, merge, association, and pattern completion plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import brain_util as bu

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
    for key, val in od.items():
        plt.plot(x, val, linewidth=0.7)
    if show_legend:
        plt.legend(od.keys(), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0, 10, 20, 50, 100])
    k = 317
    plt.yticks([k, 2*k, 5*k, 10*k, 13*k], ["k", "2k", "5k", "10k", "13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, od.keys()):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(), 
                        xycoords=ax.get_yaxis_transform(), textcoords="offset points",
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
    for key, val in od.items():
        plt.plot(x, val, linewidth=0.7)
    if show_legend:
        plt.legend(od.keys(), loc='upper left')
    ax = plt.axes()
    ax.set_xticks([0, 10, 20, 50, 100])
    k = 317
    plt.yticks([k, 2*k, 5*k, 10*k, 13*k], ["k", "2k", "5k", "10k", "13k"])
    plt.xlabel(r'$t$')

    if not show_legend:
        for line, name in zip(ax.lines, od.keys()):
            y = line.get_ydata()[-1]
            ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(), 
                        xycoords=ax.get_yaxis_transform(), textcoords="offset points",
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
    plt.plot(od.keys(), od.values(), linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["10%", "20%", "30%", "40%", "50%"])
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
    plt.plot(od.keys(), od.values(), linewidth=0.7)
    ax = plt.axes()
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
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
    plt.plot(od.keys(), od.values(), linewidth=0.7)
    ax = plt.axes()
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8], ["", "20%", "40%", "60%", "80%"])
    plt.xlabel('overlap (assemblies)')
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], ["", "5%", "10%", "15%", "20%", "25%", "30%"])
    plt.ylabel('overlap (projections)')
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)

def plot_density_ee(show=True, save="", use_text_font=True):
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
    plt.plot(od.keys(), od.values(), linewidth=0.7)
    plt.plot([0, 0.06], [0.01, 0.01], color='red', linestyle='dashed', linewidth=0.7)
    if show:
        plt.show()
    if not show and save != "":
        plt.savefig(save)
