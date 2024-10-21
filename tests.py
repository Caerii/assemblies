import brain
import brain_util as bu
import numpy as np
import random
import copy
import pickle
import matplotlib.pyplot as plt

from collections import OrderedDict

def fixed_assembly_test(n=100000,k=317,p=0.01,beta=0.01):
	"""
    Tests the ability of an area within a brain model to fix and subsequently unfix an assembly,
    assessing the stability and plasticity control features of the 'Brain' class.
    
    Parameters:
    - n (int): The number of neurons in the area.
    - k (int): The size of the assembly to be stimulated.
    - p (float): The probability of connection in the brain network.
    - beta (float): The learning rate or plasticity coefficient.

    The function performs initial projections to form an assembly, fixes the assembly to prevent
    further modifications during additional projections, and finally unfreezes it to test if
    modifications resume as expected.
    """
	b = brain.Brain(p)
	b.add_stimulus("stim",k)
	b.add_area("A",n,k,beta)
	b.project({"stim":["A"]},{})
	for i in xrange(3):
		b.project({"stim":["A"]},{"A":["A"]})
		print(b.areas["A"].w)
	b.areas["A"].fix_assembly()
	for i in xrange(5):
		b.project({"stim":["A"]},{"A":["A"]})
		print(b.areas["A"].w)
	b.areas["A"].unfix_assembly()
	for i in xrange(5):
		b.project({"stim":["A"]},{"A":["A"]})
		print(b.areas["A"].w)

def explicit_assembly_test():
	"""
    Tests the creation, connection, and projection functionality in a brain network with explicit areas.
    This function specifically tests projections from a stimulus to an explicit area, and then between areas.
    
    No parameters are used as this function sets up a predefined small network with hardcoded configurations.
    
    Demonstrates the utilization of both explicit and normal areas within the same model and tests the
    interactions through projections.
    """
	b = brain.Brain(0.5)
	b.add_stimulus("stim",3)
	b.add_explicit_area("A",10,3,beta=0.5)
	b.add_area("B",10,3,beta=0.5)

	print(b.stimuli_connectomes["stim"]["A"])
	print(b.connectomes["A"]["A"])
	print(b.connectomes["A"]["B"].shape)
	print(b.connectomes["B"]["A"].shape)

	# Now test projection stimulus -> explicit area
	print("Project stim->A")
	b.project({"stim":["A"]},{})
	print(b.areas["A"].winners)
	print(b.stimuli_connectomes["stim"]["A"])
	# Now test projection stimulus, area -> area
	b.project({"stim":["A"]},{"A":["A"]})
	print(b.areas["A"].winners)
	print(b.stimuli_connectomes["stim"]["A"])
	print(b.connectomes["A"]["A"])

	# project explicit A -> B
	print("Project explicit A -> normal B")
	b.project({},{"A":["B"]})
	print(b.areas["B"].winners)
	print(b.connectomes["A"]["B"])
	print(b.connectomes["B"]["A"])
	print(b.stimuli_connectomes["stim"]["B"])

def explicit_assembly_test2(rounds=20):
	"""
    Explores the interaction and the impact of fixing and unfixing an assembly on subsequent projections
    in a network with both explicit and regular brain areas.
    
    Parameters:
    - rounds (int): The number of projection rounds to run after setting up initial conditions.

    The test involves fixing an assembly in an explicit area, projecting to a normal area, and
    observing the behavior of projections back to the original area after unfixing the assembly.
    """
	b = brain.Brain(0.1)
	b.add_explicit_area("A",100,10,beta=0.5)
	b.add_area("B",10000,100,beta=0.5)

	b.areas["A"].winners = list(range(10,20))
	b.areas["A"].fix_assembly()
	b.project({}, {"A": ["B"]})

	# Test that if we fire back from B->A now, we don't recover the fixed assembly
	b.areas["A"].unfix_assembly()
	b.project({}, {"B": ["A"]})
	print(b.areas["A"].winners)

	b.areas["A"].winners = list(range(10,20))
	b.areas["A"].fix_assembly()
	b.project({}, {"A": ["B"]})
	for _ in range(rounds):
		b.project({}, {"A": ["B"], "B": ["A", "B"]})
		print(b.areas["B"].w)

	b.areas["A"].unfix_assembly()
	b.project({}, {"B": ["A"]})
	print("After 1 B->A, got A winners:")
	print(b.areas["A"].winners)

	for _ in range(4):
		b.project({}, {"B": ["A"], "A": ["A"]})
	print("After 5 B->A, got A winners:")
	print(b.areas["A"].winners)

def explicit_assembly_recurrent():
	"""
    Tests recurrent projection dynamics within a brain model that includes explicit assembly areas.
    
    This function sets up a brain with one explicit area and tests the behavior of recurrent projections
    within this setup. It examines how the brain model maintains assembly configurations over several
    rounds of projections, particularly focusing on stability and changes in the assembly population.
    """
	b = brain.Brain(0.1)
	b.add_explicit_area("A",100,10,beta=0.5)

	b.areas["A"].winners = list(range(60,70))


