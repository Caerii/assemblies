import numpy as np
import pickle

# Save obj (could be Brain object, list of saved winners, etc) as file_name
def sim_save(file_name, obj):
	"""Saves an object to a file using pickle."""
	with open(file_name,'wb') as f:
		pickle.dump(obj, f)

def sim_load(file_name):
	"""Loads an object from a file using pickle."""
	with open(file_name,'rb') as f:
		return pickle.load(f)

# Compute item overlap between two lists viewed as sets.
def overlap(a, b, percentage=False):
	"""Computes the number or percentage of overlapping items between two iterables."""
	set_a = set(a)
	set_b = set(b)
	o = len(set_a & set_b)
	if percentage:
		len_b = len(set_b)
		return (float(o)/float(len_b)) if len_b > 0 else 0.0
	else:
		return o

# Compute overlap of each list of winners in winners_list 
# with respect to a specific winners set, namely winners_list[base]
def get_overlaps(winners_list, base_index, percentage=False):
	"""Computes overlap of each list in winners_list with the list at base_index."""
	if not winners_list or base_index >= len(winners_list):
		return []
	overlaps = []
	base_winners = winners_list[base_index]
	k = len(base_winners) # Assuming base_winners is the reference size
	for i in range(len(winners_list)):
		o = overlap(winners_list[i], base_winners) # Use the overlap function defined above
		if percentage:
			overlaps.append(float(o)/float(k) if k > 0 else 0.0)
		else:
			overlaps.append(o)
	return overlaps 