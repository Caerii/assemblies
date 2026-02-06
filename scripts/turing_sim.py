"""
Module: turing_sim.py

This module contains simulation functions designed to test and explore the dynamics of neural assemblies
in a brain-like structure modeled through the `brain` and `brain_util` libraries. The simulations
include tests for fixed and dynamic assembly behaviors under various conditions, exploring how
neural connections stabilize, interact, and change in response to stimuli and internal projections.

The functions are intended for use in experimental neuroscience simulations where understanding the
plasticity and stability of neural circuits is crucial. They mimic the behavior of neural networks by
using artificial stimuli to influence neural assemblies and observing the outcomes in controlled scenarios.
"""

import brain
import brain_util as bu
import numpy as np
import time

def larger_k(n=10000,k=100,p=0.01,beta=0.05, bigger_factor=10):
	"""
    Simulates the interaction between two neural areas with differing assembly sizes.
    Tests how a larger assembly area (Area B) influences a smaller assembly area (Area A)
    when stimulated repeatedly and how they stabilize over iterations.

    Parameters:
    - n (int): Number of neurons in each area.
    - k (int): Base number of neurons in the smaller assembly.
    - p (float): Probability of connections in the brain model.
    - beta (float): Learning rate or plasticity coefficient.
    - bigger_factor (int): Multiplier for the size of the larger assembly compared to the smaller one.

    The function projects stimuli to Area A and tracks changes in synaptic weights over time, providing
    output on stabilization and final synaptic weights in both areas.
    """
	start_time = time.time()

	b = brain.Brain(p, save_winners=True)
	b.add_stimulus("stim", k)
	b.add_area("A", n, k, beta)
	b.add_area("B", n, bigger_factor * k, beta)
	b.update_plasticities(area_update_map={"A": [("B", 0.8), ("A", 0.0)],
                                           "B": [("A", 0.8), ("B", 0.8)]})
	b.project({"stim": ["A"]}, {})
	t = 1
	while True:
		b.project({"stim": ["A"]}, {"A": ["A"]})
		print("A total w is " + str(b.area_by_name["A"].w))
		if (b.area_by_name["B"].num_first_winners <= 1) and (b.area_by_name["A"].num_first_winners <= 1):
			print("proj(stim, A) stabilized after " + str(t) + " rounds")
			break
		t += 1
	print(f"Stabilization time: {time.time() - start_time} seconds")
	A_after_proj = b.area_by_name["A"].winners

	b.project({"stim": ["A"]}, {"A": ["A", "B"]})
	t = 1
	while True:
		b.project({"stim": ["A"]}, {"A": ["A", "B"], "B": ["B", "A"]})
		print("Num new winners in A " + str(b.area_by_name["A"].num_first_winners))
		print("Num new winners in B " + str(b.area_by_name["B"].num_first_winners))
		if (b.area_by_name["B"].num_first_winners <= 1) and (b.area_by_name["A"].num_first_winners <= 1):
			print("recip_project(A, B) stabilized after " + str(t) + " rounds")
			break
		t += 1
	print("Final statistics")
	print("A.w = " + str(b.area_by_name["A"].w))
	print("B.w = " + str(b.area_by_name["B"].w))
	A_after_B = b.area_by_name["A"].saved_winners[-1]
	o = bu.overlap(A_after_proj, A_after_B)
	print("Overlap is " + str(o))

	print(f"Total time: {time.time() - start_time} seconds")

def turing_erase(n=50000,k=100,p=0.01,beta=0.05, r=1.0, bigger_factor=20):
	"""
    Simulates the erasing process in a Turing machine model by selectively reducing the influence of one area on another.
    This test explores the dynamics of 'erasing' an assembly's influence on another by adjusting plasticities and connectivity.

    Parameters:
    - n (int): Number of neurons in each area.
    - k (int): Number of neurons in the base assembly.
    - p (float): Probability of connections in the brain model.
    - beta (float): Learning rate or plasticity coefficient.
    - r (float): Ratio to determine the size of the stimulus compared to the base assembly.
    - bigger_factor (int): Factor by which the neural areas are larger than the base assembly size.

    The function demonstrates how changes in synaptic weights affect the memory and recovery of neural patterns
    when subjected to competing influences from other neural assemblies.
    """
	start_time = time.time()

	b = brain.Brain(p, save_winners=True)
    # Much smaller stimulus, similar to lower p from stimulus into A
	smaller_k = int(r * k)
	b.add_stimulus("stim", smaller_k)
	b.add_area("A", n, bigger_factor * k, beta)
	b.add_area("B", n, bigger_factor * k, beta)
	b.add_area("C", n, bigger_factor * k, beta)
	b.update_plasticities(area_update_map={"A": [("B", 0.8), ("C", 0.8), ("A", 0.01)],
                                           "B": [("A", 0.8), ("B", 0.8)],
                                           "C": [("A", 0.8), ("C", 0.8)]},
                         stim_update_map={"A": [("stim", 0.05)]})
	b.project({"stim": ["A"]}, {})
	t = 1
	while True:
		b.project({"stim": ["A"]}, {"A": ["A"]})
		if (b.area_by_name["B"].num_first_winners <= 1) and (b.area_by_name["A"].num_first_winners <= 1):
			print("proj(stim, A) stabilized after " + str(t) + " rounds")
			break
		t += 1

	print(f"Stabilization time: {time.time() - start_time} seconds")

	b.project({"stim": ["A"]}, {"A": ["A", "B"]})
	t = 1
	while True:
		b.project({"stim": ["A"]}, {"A": ["A", "B"], "B": ["B", "A"]})
		print("Num new winners in A " + str(b.area_by_name["A"].num_first_winners))
		if (b.area_by_name["B"].num_first_winners <= 1) and (b.area_by_name["A"].num_first_winners <= 1):
			print("recip_project(A, B) stabilized after " + str(t) + " rounds")
			break
		t += 1
	A_after_proj_B = b.area_by_name["A"].winners

	b.project({"stim": ["A"]}, {"A": ["A", "C"]})
	t = 1
	while True:
		b.project({"stim": ["A"]}, {"A": ["A", "C"], "C": ["C", "A"]})
		print("Num new winners in A " + str(b.area_by_name["A"].num_first_winners))
		if (b.area_by_name["C"].num_first_winners <= 1) and (b.area_by_name["A"].num_first_winners <= 1):
			print("recip_project(A, C) stabilized after " + str(t) + " rounds")
			break
		t += 1

	print(f"Total time: {time.time() - start_time} seconds")

	A_after_proj_C = b.area_by_name["A"].winners

    # Check final conditions
	b.project({}, {"A": ["B"]})
	B_after_erase = b.area_by_name["B"].saved_winners[-1]
	B_before_erase = b.area_by_name["B"].saved_winners[-2]
	B_overlap = bu.overlap(B_after_erase, B_before_erase)
	print("Overlap of B after erase and with y is " + str(B_overlap) + "\n")
	A_overlap = bu.overlap(A_after_proj_B, A_after_proj_C)
	print("Overlap of A after proj(B) vs after proj(C) is " + str(A_overlap) + "\n")